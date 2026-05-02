use serde_json::{json, Value};
use wasm_bindgen::prelude::*;

pub mod actions;
pub mod context;
pub mod extensions;
pub mod policy;
pub mod schemas;

use crate::actions::{compact_whitespace, ActionKind, ActionLedgerRecord};
use crate::context::{compile_context_packet, prompt_from_context_packet, rank_evidence};
use crate::extensions::{parse_manifest, parse_receipt, ApprovalPolicy, ExtensionManifest};
use crate::policy::parse_model_decision;
use crate::schemas::{
    ActionDecisionLite, ContextPacketLite, EpisodeRecordLite, EvidenceRecord, MessageRecord,
    StepRecordLite, TurnRecord,
};

const DEFAULT_MAX_STEPS: usize = 8;
const MAX_SNIPPET_CHARS: usize = 900;
const MAX_TITLE_CHARS: usize = 180;
const CORE_VERSION: &str = env!("CARGO_PKG_VERSION");

#[wasm_bindgen]
pub struct AgentLiteCore {
    session_id: String,
    mode: String,
    max_context_items: usize,
    max_steps: usize,
    request_counter: usize,
    messages: Vec<MessageRecord>,
    turns: Vec<TurnRecord>,
    extension_manifests: Vec<ExtensionManifest>,
    action_ledger: Vec<ActionLedgerRecord>,
    last_context_packet: Option<ContextPacketLite>,
    last_decision: Option<ActionDecisionLite>,
}

#[wasm_bindgen]
impl AgentLiteCore {
    #[wasm_bindgen(constructor)]
    pub fn new(session_id: String, mode: String, max_context_items: usize) -> AgentLiteCore {
        AgentLiteCore {
            session_id: nonempty(session_id, "browser-session"),
            mode: normalize_mode(&mode),
            max_context_items: max_context_items.clamp(1, 12),
            max_steps: DEFAULT_MAX_STEPS,
            request_counter: 0,
            messages: Vec::new(),
            turns: Vec::new(),
            extension_manifests: Vec::new(),
            action_ledger: Vec::new(),
            last_context_packet: None,
            last_decision: None,
        }
    }

    pub fn set_mode(&mut self, mode: String) {
        self.mode = normalize_mode(&mode);
    }

    pub fn reset(&mut self) {
        self.request_counter = 0;
        self.messages.clear();
        self.turns.clear();
        self.action_ledger.clear();
        self.last_context_packet = None;
        self.last_decision = None;
    }

    pub fn step_count(&self) -> usize {
        self.turns.len()
    }

    pub fn can_continue(&self) -> bool {
        self.turns.len() < self.max_steps
    }

    pub fn start_turn(
        &mut self,
        user_text: String,
        context_rows_json: String,
        language: String,
        max_new_tokens: u32,
    ) -> String {
        self.request_counter = self.request_counter.saturating_add(1);
        let step_index = self.turns.len().saturating_add(1);
        let normalized_user_text = compact_whitespace(&user_text);
        let normalized_language = nonempty(language, "Auto");
        let evidence = parse_evidence_rows(&context_rows_json, self.max_context_items);
        let request_id = format!("{}-{}", self.session_id, self.request_counter);
        let task_json = json!({
            "task_id": request_id,
            "prompt": normalized_user_text,
            "workspace_subdir": "browser",
            "max_steps": self.max_steps,
        })
        .to_string();
        let packet = compile_context_packet(
            &request_id,
            &task_json,
            &normalized_user_text,
            &context_rows_json,
            "[]",
            self.max_context_items,
            &self.mode,
        );
        let prompt = prompt_from_context_packet(
            &self.mode,
            &normalized_language,
            &normalized_user_text,
            &packet,
            step_index,
            self.max_steps,
        );
        self.last_context_packet = Some(packet.clone());
        self.messages.push(MessageRecord {
            role: "user".to_string(),
            text: normalized_user_text.clone(),
        });
        self.turns.push(TurnRecord {
            step_index,
            mode: self.mode.clone(),
            language: normalized_language.clone(),
            user_text: normalized_user_text,
            evidence_count: evidence.len(),
            max_new_tokens,
            completed: false,
        });
        json!({
            "request_id": request_id,
            "step_index": step_index,
            "max_steps": self.max_steps,
            "mode": self.mode,
            "language": normalized_language,
            "prompt": prompt,
            "evidence": evidence,
            "context_packet": packet,
            "can_continue": self.can_continue(),
        })
        .to_string()
    }

    pub fn finish_turn(&mut self, assistant_text: String) -> String {
        let text = trim_generated_text(&assistant_text);
        if let Some(turn) = self.turns.last_mut() {
            turn.completed = true;
        }
        self.messages.push(MessageRecord {
            role: "assistant".to_string(),
            text: text.clone(),
        });
        json!({
            "step_index": self.turns.len(),
            "completed_steps": self.turns.iter().filter(|turn| turn.completed).count(),
            "can_continue": self.can_continue(),
            "assistant_text": text,
            "snapshot": self.snapshot_value(),
        })
        .to_string()
    }

    pub fn finish_model_reply(&mut self, model_text: String) -> String {
        let packet = parse_model_decision(
            &model_text,
            &self.mode,
            r#"{"code_execution_enabled": false}"#,
        );
        self.last_decision = Some(packet.decision.clone());
        let text = if packet.decision.content.trim().is_empty() {
            model_text
        } else {
            packet.decision.content.clone()
        };
        let receipt = self.finish_turn(text);
        json!({
            "decision_packet": packet,
            "turn_receipt": serde_json::from_str::<Value>(&receipt).unwrap_or(Value::Null),
            "snapshot": self.snapshot_value(),
        })
        .to_string()
    }

    pub fn parse_model_decision(&self, model_text: String, options_json: String) -> String {
        serde_json::to_string(&parse_model_decision(
            &model_text,
            &self.mode,
            &options_json,
        ))
        .unwrap_or_else(|error| {
            json!({
                "parse_status": "error",
                "error": format!("decision serialization failed: {error}"),
            })
            .to_string()
        })
    }

    pub fn plan_lite_turn(
        &self,
        user_text: String,
        history_json: String,
        options_json: String,
    ) -> String {
        let user_text = compact_whitespace(&user_text);
        let history = serde_json::from_str::<Value>(&history_json).unwrap_or(Value::Null);
        let options = serde_json::from_str::<Value>(&options_json).unwrap_or(Value::Null);
        let selected_context_count = options
            .get("selected_context_count")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let plan = plan_lite_action(&user_text, &history, selected_context_count as usize);
        json!({
            "action": plan.0.as_str(),
            "query": user_text,
            "reason": plan.1,
            "selected_context_count": selected_context_count,
            "loop_contract": {
                "commands": ["respond", "gather_context"],
                "always_final_response": true,
                "gather_context": "rank and add relevant papers/context, then respond",
                "respond": "answer directly without adding new papers"
            }
        })
        .to_string()
    }

    pub fn rank_evidence(
        &self,
        query: String,
        candidate_rows_json: String,
        limit: usize,
    ) -> String {
        serde_json::to_string(&rank_evidence(&query, &candidate_rows_json, limit)).unwrap_or_else(
            |error| {
                json!({
                    "status": "error",
                    "error": format!("ranked evidence serialization failed: {error}"),
                })
                .to_string()
            },
        )
    }

    pub fn compile_context_packet(
        &mut self,
        task_json: String,
        evidence_json: String,
        history_json: String,
    ) -> String {
        let task_value = serde_json::from_str::<Value>(&task_json).unwrap_or(Value::Null);
        let query = first_string(&task_value, &["prompt", "user_text", "task"]);
        let request_id = format!(
            "{}-ctx-{}",
            self.session_id,
            self.request_counter.saturating_add(1)
        );
        let packet = compile_context_packet(
            &request_id,
            &task_json,
            &query,
            &evidence_json,
            &history_json,
            self.max_context_items,
            &self.mode,
        );
        self.last_context_packet = Some(packet.clone());
        serde_json::to_string(&packet).unwrap_or_else(|error| {
            json!({
                "status": "error",
                "error": format!("context packet serialization failed: {error}"),
            })
            .to_string()
        })
    }

    pub fn start_turn_with_context(
        &mut self,
        task_json: String,
        retrieval_candidates_json: String,
        history_json: String,
        options_json: String,
    ) -> String {
        self.request_counter = self.request_counter.saturating_add(1);
        let step_index = self.turns.len().saturating_add(1);
        let options = serde_json::from_str::<Value>(&options_json).unwrap_or(Value::Null);
        let normalized_language = nonempty(
            options
                .get("language")
                .and_then(Value::as_str)
                .unwrap_or("Auto")
                .to_string(),
            "Auto",
        );
        let max_new_tokens = options
            .get("max_new_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(560) as u32;
        let task_value = serde_json::from_str::<Value>(&task_json).unwrap_or(Value::Null);
        let user_text = nonempty(
            first_string(&task_value, &["prompt", "user_text", "task"]),
            "Continue.",
        );
        let request_id = format!("{}-{}", self.session_id, self.request_counter);
        let requested_context_items = options
            .get("max_context_items")
            .and_then(Value::as_u64)
            .map(|value| value as usize)
            .unwrap_or(self.max_context_items)
            .clamp(1, self.max_context_items.max(1));
        let packet = compile_context_packet(
            &request_id,
            &task_json,
            &user_text,
            &retrieval_candidates_json,
            &history_json,
            requested_context_items,
            &self.mode,
        );
        let prompt = prompt_from_context_packet(
            &self.mode,
            &normalized_language,
            &user_text,
            &packet,
            step_index,
            self.max_steps,
        );
        let evidence = parse_evidence_rows(&retrieval_candidates_json, requested_context_items);
        self.last_context_packet = Some(packet.clone());
        self.messages.push(MessageRecord {
            role: "user".to_string(),
            text: user_text.clone(),
        });
        self.turns.push(TurnRecord {
            step_index,
            mode: self.mode.clone(),
            language: normalized_language.clone(),
            user_text,
            evidence_count: evidence.len(),
            max_new_tokens,
            completed: false,
        });
        json!({
            "request_id": request_id,
            "step_index": step_index,
            "max_steps": self.max_steps,
            "mode": self.mode,
            "language": normalized_language,
            "prompt": prompt,
            "evidence": evidence,
            "context_packet": packet,
            "can_continue": self.can_continue(),
        })
        .to_string()
    }

    pub fn register_extension_manifest(&mut self, manifest_json: String) -> String {
        match parse_manifest(&manifest_json) {
            Ok(mut manifest) => {
                manifest.disable_on_register();
                let id = manifest.id.clone();
                self.extension_manifests
                    .retain(|existing| existing.id != id);
                self.extension_manifests.push(manifest);
                json!({
                    "status": "installed",
                    "extension_id": id,
                    "enabled": false,
                    "installed_extension_count": self.extension_manifests.len(),
                    "registered_extension_count": self.extension_manifests.len(),
                })
                .to_string()
            }
            Err(error) => json!({
                "status": "error",
                "error": error,
            })
            .to_string(),
        }
    }

    pub fn install_extension_manifest(&mut self, manifest_json: String) -> String {
        self.register_extension_manifest(manifest_json)
    }

    pub fn uninstall_extension(&mut self, extension_id: String) -> String {
        let normalized_extension_id = compact_whitespace(&extension_id);
        let before = self.extension_manifests.len();
        self.extension_manifests
            .retain(|manifest| manifest.id != normalized_extension_id);
        if self.extension_manifests.len() == before {
            return json!({
                "status": "error",
                "error": "extension is not installed",
                "extension_id": normalized_extension_id,
            })
            .to_string();
        }
        json!({
            "status": "uninstalled",
            "extension_id": normalized_extension_id,
            "installed_extension_count": self.extension_manifests.len(),
        })
        .to_string()
    }

    pub fn set_extension_enabled(&mut self, extension_id: String, enabled: bool) -> String {
        let normalized_extension_id = compact_whitespace(&extension_id);
        let Some(manifest) = self
            .extension_manifests
            .iter_mut()
            .find(|manifest| manifest.id == normalized_extension_id)
        else {
            return json!({
                "status": "error",
                "error": "extension is not registered",
                "extension_id": normalized_extension_id,
            })
            .to_string();
        };
        if manifest.approval_policy == ApprovalPolicy::Disabled && enabled {
            return json!({
                "status": "disabled",
                "error": "extension approval policy is disabled",
                "extension_id": normalized_extension_id,
                "enabled": false,
            })
            .to_string();
        }
        manifest.set_enabled(enabled);
        json!({
            "status": if enabled { "enabled" } else { "disabled" },
            "extension_id": normalized_extension_id,
            "enabled": manifest.enabled,
        })
        .to_string()
    }

    pub fn list_extension_manifests(&self) -> String {
        json!({
            "status": "ok",
            "extensions": self.extension_manifests,
            "installed_extension_count": self.extension_manifests.len(),
            "registered_extension_count": self.extension_manifests.len(),
            "enabled_extension_count": self.extension_manifests.iter().filter(|manifest| manifest.enabled).count(),
        })
        .to_string()
    }

    pub fn propose_extension_action(
        &mut self,
        extension_id: String,
        capability_id: String,
        input_json: String,
    ) -> String {
        let normalized_extension_id = compact_whitespace(&extension_id);
        let normalized_capability_id = compact_whitespace(&capability_id);
        let Some(manifest) = self
            .extension_manifests
            .iter()
            .find(|manifest| manifest.id == normalized_extension_id)
        else {
            return json!({
                "status": "error",
                "error": "extension is not registered",
                "extension_id": normalized_extension_id,
            })
            .to_string();
        };
        if !manifest.enabled {
            return json!({
                "status": "disabled",
                "error": "extension is not enabled",
                "extension_id": normalized_extension_id,
            })
            .to_string();
        }
        if manifest.approval_policy == ApprovalPolicy::Disabled {
            return json!({
                "status": "disabled",
                "error": "extension approval policy is disabled",
                "extension_id": normalized_extension_id,
            })
            .to_string();
        }
        if !manifest.supports_capability(&normalized_capability_id) {
            return json!({
                "status": "error",
                "error": "extension does not declare requested capability",
                "extension_id": normalized_extension_id,
                "capability_id": normalized_capability_id,
            })
            .to_string();
        }
        let input = serde_json::from_str::<Value>(&input_json).unwrap_or_else(|_| {
            json!({
                "raw": input_json,
                "parse_warning": "input_json was not valid JSON; stored as raw text"
            })
        });
        let action_id = format!(
            "act_{}_{}",
            self.session_id
                .replace(|ch: char| !ch.is_ascii_alphanumeric(), "_"),
            self.action_ledger.len().saturating_add(1)
        );
        let record = ActionLedgerRecord::pending_extension(
            action_id.clone(),
            normalized_capability_id.clone(),
            input,
            self.turns.len(),
            format!("turn-{}", self.turns.len()),
        );
        self.action_ledger.push(record);
        json!({
            "status": "pending_user_approval",
            "action_id": action_id,
            "extension_id": normalized_extension_id,
            "capability_id": normalized_capability_id,
            "requires_user_approval": true,
            "ledger_size": self.action_ledger.len(),
        })
        .to_string()
    }

    pub fn record_extension_result(&mut self, action_id: String, receipt_json: String) -> String {
        let normalized_action_id = compact_whitespace(&action_id);
        match parse_receipt(&receipt_json) {
            Ok(receipt) => {
                if receipt.action_id != normalized_action_id {
                    return json!({
                        "status": "error",
                        "error": "receipt action_id does not match requested action_id",
                        "action_id": normalized_action_id,
                        "receipt_action_id": receipt.action_id,
                    })
                    .to_string();
                }
                let Some(record) = self
                    .action_ledger
                    .iter_mut()
                    .find(|record| record.action_id == normalized_action_id)
                else {
                    return json!({
                        "status": "error",
                        "error": "action_id was not found in the ledger",
                        "action_id": normalized_action_id,
                    })
                    .to_string();
                };
                record.policy_state = receipt.status.clone();
                record.receipt = Some(serde_json::to_value(&receipt).unwrap_or(Value::Null));
                json!({
                    "status": "recorded",
                    "action_id": normalized_action_id,
                    "policy_state": record.policy_state,
                    "ledger_size": self.action_ledger.len(),
                })
                .to_string()
            }
            Err(error) => json!({
                "status": "error",
                "error": error,
                "action_id": normalized_action_id,
            })
            .to_string(),
        }
    }

    pub fn export_episode_json(&self) -> String {
        let steps = self
            .turns
            .iter()
            .filter(|turn| turn.completed)
            .map(|turn| StepRecordLite {
                index: turn.step_index,
                action: if turn.mode == "code" {
                    ActionKind::ProposeCode
                } else {
                    ActionKind::Respond
                },
                content: turn.user_text.clone(),
                verification: json!({
                    "passed": true,
                    "reasons": ["browser turn recorded"]
                }),
                runtime_attestation: self.runtime_attestation_value(),
            })
            .collect::<Vec<_>>();
        let episode = EpisodeRecordLite {
            task_id: format!("browser-session-{}", self.session_id),
            prompt: self
                .messages
                .iter()
                .find(|message| message.role == "user")
                .map(|message| message.text.clone())
                .unwrap_or_default(),
            workspace: "browser".to_string(),
            success: !self.messages.is_empty(),
            task_metadata: json!({
                "benchmark_family": "agentkernel_lite_browser",
                "runtime": "rust_wasm",
                "code_execution": "disabled",
            }),
            task_contract: json!({
                "workspace_subdir": "browser",
                "max_steps": self.max_steps,
            }),
            termination_reason: if self.can_continue() {
                "active".to_string()
            } else {
                "max_steps_reached".to_string()
            },
            steps,
            messages: self.messages.clone(),
            action_ledger: serde_json::to_value(&self.action_ledger).unwrap_or(Value::Null),
        };
        serde_json::to_string(&episode).unwrap_or_else(|error| {
            json!({
                "status": "error",
                "error": format!("episode serialization failed: {error}")
            })
            .to_string()
        })
    }

    pub fn runtime_attestation(&self) -> String {
        self.runtime_attestation_value().to_string()
    }

    pub fn snapshot_json(&self) -> String {
        self.snapshot_value().to_string()
    }
}

fn parse_evidence_rows(raw_json: &str, limit: usize) -> Vec<EvidenceRecord> {
    let Ok(value) = serde_json::from_str::<Value>(raw_json) else {
        return Vec::new();
    };
    let rows = value.as_array().cloned().unwrap_or_default();
    rows.iter()
        .take(limit)
        .enumerate()
        .map(|(idx, row)| evidence_from_row(idx + 1, row))
        .collect()
}

fn evidence_from_row(index: usize, row: &Value) -> EvidenceRecord {
    let title = truncate_chars(
        &nonempty(first_string(row, &["title", "name"]), "Untitled paper"),
        MAX_TITLE_CHARS,
    );
    let paper_id = first_string(row, &["paper_id", "canonical_paper_id", "arxiv_id", "id"]);
    let snippet = truncate_chars(
        &nonempty(
            compact_whitespace(&first_string(
                row,
                &["context_text", "abstract", "summary", "text", "full_text"],
            )),
            "No abstract or text snippet was available in the loaded row.",
        ),
        MAX_SNIPPET_CHARS,
    );
    EvidenceRecord {
        index,
        title,
        paper_id: paper_id.clone(),
        primary_category: first_string(row, &["primary_category", "category", "category_list"]),
        year: first_string(row, &["year", "published_year"]),
        source: first_string(row, &["source", "dataset", "pack"]),
        snippet,
        pdf_url: arxiv_pdf_url(&paper_id),
        score: first_number(row, &["retrieval_score", "score"]),
    }
}

fn first_string(row: &Value, keys: &[&str]) -> String {
    for key in keys {
        let Some(value) = row.get(*key) else {
            continue;
        };
        if let Some(text) = value.as_str() {
            let normalized = compact_whitespace(text);
            if !normalized.is_empty() {
                return normalized;
            }
        }
        if value.is_number() {
            return value.to_string();
        }
    }
    String::new()
}

fn first_number(row: &Value, keys: &[&str]) -> f64 {
    for key in keys {
        if let Some(value) = row.get(*key).and_then(Value::as_f64) {
            return value;
        }
    }
    0.0
}

fn arxiv_pdf_url(raw: &str) -> String {
    let id = raw
        .trim()
        .trim_start_matches("arXiv:")
        .trim_start_matches("arxiv:")
        .split_whitespace()
        .next()
        .unwrap_or("")
        .trim_end_matches('.');
    if id.is_empty() {
        return String::new();
    }
    let id = strip_arxiv_version(id)
        .trim_matches(|ch: char| ch == '[' || ch == ']' || ch == '(' || ch == ')');
    let modern = id.len() >= 9
        && id.as_bytes().get(4) == Some(&b'.')
        && id
            .chars()
            .enumerate()
            .all(|(idx, ch)| idx == 4 && ch == '.' || ch.is_ascii_digit());
    let legacy = id.contains('/')
        && id
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-' | '/'));
    if modern || legacy {
        format!("https://arxiv.org/pdf/{id}")
    } else {
        String::new()
    }
}

fn strip_arxiv_version(id: &str) -> &str {
    let Some(version_index) = id.rfind('v') else {
        return id;
    };
    let suffix = &id[version_index + 1..];
    if suffix.is_empty() || !suffix.chars().all(|ch| ch.is_ascii_digit()) {
        return id;
    }
    if !id[..version_index]
        .chars()
        .last()
        .is_some_and(|ch| ch.is_ascii_digit())
    {
        return id;
    }
    &id[..version_index]
}

fn plan_lite_action(
    user_text: &str,
    _history: &Value,
    selected_context_count: usize,
) -> (ActionKind, String) {
    let normalized = user_text.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return (
            ActionKind::Respond,
            "empty prompt; respond directly".to_string(),
        );
    }

    let conversational = [
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank you",
        "who are you",
        "what can you do",
    ];
    if conversational
        .iter()
        .any(|phrase| normalized == *phrase || normalized.starts_with(&format!("{phrase} ")))
    {
        return (
            ActionKind::Respond,
            "conversational prompt does not need paper retrieval".to_string(),
        );
    }

    let retrieval_terms = [
        "paper",
        "papers",
        "research",
        "study",
        "studies",
        "literature",
        "arxiv",
        "citation",
        "evidence",
        "source",
        "sources",
        "retrieve",
        "find",
        "look up",
        "search",
        "survey",
        "compare",
        "summarize",
        "explain this paper",
        "what does this paper",
        "according to",
    ];
    if retrieval_terms.iter().any(|term| normalized.contains(term)) {
        return (
            ActionKind::GatherContext,
            "prompt asks for research-backed context".to_string(),
        );
    }

    if selected_context_count > 0
        && ["this", "that", "it", "paper", "above"]
            .iter()
            .any(|term| normalized.contains(term))
    {
        return (
            ActionKind::GatherContext,
            "selected paper context is relevant to the prompt".to_string(),
        );
    }

    if normalized.ends_with('?') && normalized.split_whitespace().count() >= 7 {
        return (
            ActionKind::GatherContext,
            "substantive question may benefit from ranked context".to_string(),
        );
    }

    if normalized.split_whitespace().count() >= 5 {
        return (
            ActionKind::GatherContext,
            "substantive topic can benefit from ranked context".to_string(),
        );
    }

    (
        ActionKind::Respond,
        "general chat prompt; respond without adding new papers".to_string(),
    )
}

fn trim_generated_text(text: &str) -> String {
    let mut out = text.trim().to_string();
    if let Some(index) = out.rfind("Assistant:") {
        out = out[index + "Assistant:".len()..].trim().to_string();
    }
    out
}

fn normalize_mode(mode: &str) -> String {
    match mode.trim().to_ascii_lowercase().as_str() {
        "think" => "think".to_string(),
        "deep_research" | "deep-research" | "deep research" | "research" => {
            "deep_research".to_string()
        }
        _ => "chat".to_string(),
    }
}

fn nonempty(value: impl Into<String>, fallback: &str) -> String {
    let normalized = compact_whitespace(&value.into());
    if normalized.is_empty() {
        fallback.to_string()
    } else {
        normalized
    }
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    let mut out = String::new();
    for ch in value.chars().take(max_chars) {
        out.push(ch);
    }
    if value.chars().count() > max_chars {
        out.push_str("...");
    }
    out
}

impl AgentLiteCore {
    fn snapshot_value(&self) -> Value {
        json!({
            "session_id": self.session_id,
            "mode": self.mode,
            "max_context_items": self.max_context_items,
            "max_steps": self.max_steps,
            "step_count": self.turns.len(),
            "can_continue": self.can_continue(),
            "messages": self.messages,
            "turns": self.turns,
            "registered_extensions": self.extension_manifests,
            "action_ledger": self.action_ledger,
            "last_context_packet": self.last_context_packet,
            "last_decision": self.last_decision,
            "runtime_attestation": self.runtime_attestation_value(),
        })
    }

    fn runtime_attestation_value(&self) -> Value {
        json!({
            "runtime": "agent_kernel_rust_wasm",
            "core_version": CORE_VERSION,
            "wasm": true,
            "code_execution": "disabled",
            "storage_authority": "browser_adapter",
            "extension_execution": "adapter_with_user_approval",
            "registered_extension_count": self.extension_manifests.len(),
            "action_ledger_count": self.action_ledger.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packs_research_evidence_into_prompt() {
        let mut core = AgentLiteCore::new("s".to_string(), "think".to_string(), 2);
        let rows = json!([
            {
                "title": "Paper A",
                "paper_id": "1001.5047v1",
                "abstract": "  alpha   beta   ",
                "primary_category": "cs.FL",
                "year": 2010
            }
        ]);
        let packet = core.start_turn(
            "write it".to_string(),
            rows.to_string(),
            "Rust".to_string(),
            320,
        );
        assert!(packet.contains("Paper A"));
        assert!(packet.contains("https://arxiv.org/pdf/1001.5047"));
        assert!(packet.contains("<AK_THINK> <AK_RESPOND> <AK_CONTEXT> <AK_ANSWER>"));
        assert!(packet.contains("<AK_EVIDENCE> <AK_EVIDENCE_ID> P1"));
        assert!(packet.contains("<AK_TITLE> Paper A"));
        assert!(packet.contains("<AK_LOOP> <AK_STATE>"));
        assert!(!packet.contains("<AK_USE_CONTEXT>"));
        assert!(packet.contains("mode=think"));
        assert!(packet.contains("target_language=Rust"));
    }

    #[test]
    fn records_turn_completion() {
        let mut core = AgentLiteCore::new("s".to_string(), "chat".to_string(), 2);
        core.start_turn(
            "hello".to_string(),
            "[]".to_string(),
            "Auto".to_string(),
            128,
        );
        let done = core.finish_turn("Assistant: hi".to_string());
        assert!(done.contains("\"completed_steps\":1"));
        assert!(core.snapshot_json().contains("\"assistant\""));
    }

    #[test]
    fn plans_lite_chat_before_retrieval() {
        let core = AgentLiteCore::new("s".to_string(), "chat".to_string(), 2);
        let greeting = serde_json::from_str::<Value>(&core.plan_lite_turn(
            "hello".to_string(),
            "[]".to_string(),
            "{}".to_string(),
        ))
        .unwrap();
        assert_eq!(
            greeting.get("action").and_then(Value::as_str),
            Some("respond")
        );

        let research = serde_json::from_str::<Value>(&core.plan_lite_turn(
            "compare papers on sparse attention".to_string(),
            "[]".to_string(),
            "{}".to_string(),
        ))
        .unwrap();
        assert_eq!(
            research.get("action").and_then(Value::as_str),
            Some("gather_context")
        );

        let substantive_topic = ["explain", "topic", "with", "several", "terms"].join(" ");
        let topic = serde_json::from_str::<Value>(&core.plan_lite_turn(
            substantive_topic,
            "[]".to_string(),
            "{}".to_string(),
        ))
        .unwrap();
        assert_eq!(
            topic.get("action").and_then(Value::as_str),
            Some("gather_context")
        );
    }

    #[test]
    fn strips_only_arxiv_version_suffix() {
        assert_eq!(strip_arxiv_version("1001.5047v2"), "1001.5047");
        assert_eq!(
            strip_arxiv_version("solv-int/9901001v1"),
            "solv-int/9901001"
        );
        assert_eq!(strip_arxiv_version("solv-int/9901001"), "solv-int/9901001");
    }

    #[test]
    fn records_extension_action_with_receipt() {
        let mut core = AgentLiteCore::new("s".to_string(), "chat".to_string(), 2);
        let registered = core.register_extension_manifest(
            json!({
                "id": "github",
                "name": "GitHub",
                "source": "official",
                "default_enabled": true,
                "enabled": true,
                "capabilities": [{"id": "github.read_repo", "description": "read repo"}]
            })
            .to_string(),
        );
        assert!(registered.contains("\"status\":\"installed\""));
        assert!(registered.contains("\"enabled\":false"));
        let disabled_proposal = core.propose_extension_action(
            "github".to_string(),
            "github.read_repo".to_string(),
            json!({"repo": "owner/name"}).to_string(),
        );
        assert!(disabled_proposal.contains("extension is not enabled"));
        let enabled = core.set_extension_enabled("github".to_string(), true);
        assert!(enabled.contains("\"status\":\"enabled\""));
        let proposal = core.propose_extension_action(
            "github".to_string(),
            "github.read_repo".to_string(),
            json!({"repo": "owner/name"}).to_string(),
        );
        assert!(proposal.contains("pending_user_approval"));
        let action_id = serde_json::from_str::<Value>(&proposal)
            .unwrap()
            .get("action_id")
            .unwrap()
            .as_str()
            .unwrap()
            .to_string();
        let receipt = core.record_extension_result(
            action_id.clone(),
            json!({
                "action_id": action_id,
                "status": "approved_executed",
                "output": {"ok": true}
            })
            .to_string(),
        );
        assert!(receipt.contains("approved_executed"));
        assert!(core.snapshot_json().contains("action_ledger"));
        assert!(core.list_extension_manifests().contains("enabled_extension_count"));
        let uninstalled = core.uninstall_extension("github".to_string());
        assert!(uninstalled.contains("\"status\":\"uninstalled\""));
    }
}
