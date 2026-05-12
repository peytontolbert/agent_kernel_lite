use serde_json::{json, Value};

use crate::actions::compact_whitespace;
use crate::schemas::{ContextPacketLite, RetrievedSpanLite};

const MAX_SPAN_TEXT_CHARS: usize = 1400;
const MAX_CONTEXT_CHUNK_CHARS: usize = 700;
const THINK_CONTEXT_CHUNK_CHARS: usize = 1100;
const DEEP_CONTEXT_CHUNK_CHARS: usize = 1800;

struct ModeProfile {
    normalized: &'static str,
    label: &'static str,
    retrieval_strategy: &'static str,
    answer_policy: &'static str,
    chunk_chars: usize,
}

fn mode_profile(mode: &str) -> ModeProfile {
    match mode.trim().to_ascii_lowercase().as_str() {
        "think" => ModeProfile {
            normalized: "think",
            label: "Think",
            retrieval_strategy: "semantic_synthesis",
            answer_policy: "Synthesize across the selected evidence before answering. Prefer a reasoned answer over a terse reply, and call out uncertainty.",
            chunk_chars: THINK_CONTEXT_CHUNK_CHARS,
        },
        "deep_research" | "deep-research" | "deep research" | "research" => ModeProfile {
            normalized: "deep_research",
            label: "Deep Research",
            retrieval_strategy: "evidence_by_evidence_review",
            answer_policy: "Review each retrieved evidence item, reconcile conflicts, separate what is directly supported from inference, and produce the highest-quality grounded answer.",
            chunk_chars: DEEP_CONTEXT_CHUNK_CHARS,
        },
        _ => ModeProfile {
            normalized: "chat",
            label: "Chat",
            retrieval_strategy: "fast_grounded_answer",
            answer_policy: "Answer the user directly and naturally. Use retrieved evidence as support, cite bracket numbers only when useful, and say when the evidence is insufficient.",
            chunk_chars: MAX_CONTEXT_CHUNK_CHARS,
        },
    }
}

pub fn rank_evidence(query: &str, candidates_json: &str, limit: usize) -> Vec<RetrievedSpanLite> {
    let Ok(value) = serde_json::from_str::<Value>(candidates_json) else {
        return Vec::new();
    };
    let tokens = query_tokens(query);
    let phrase = compact_whitespace(query).to_ascii_lowercase();
    let mut spans = candidate_rows(&value)
        .into_iter()
        .enumerate()
        .map(|(index, row)| span_from_row(index + 1, &row, &tokens, &phrase))
        .filter(|span| span.score > 0.0 || !span.text.is_empty())
        .collect::<Vec<_>>();
    spans.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.span_id.cmp(&b.span_id))
    });
    spans.truncate(limit.max(1));
    spans
}

pub fn compile_context_packet(
    request_id: &str,
    task_json: &str,
    query: &str,
    evidence_json: &str,
    history_json: &str,
    limit: usize,
    mode: &str,
) -> ContextPacketLite {
    let task_value = parse_json_or_string(task_json, "prompt");
    let history_value = parse_json_or_string(history_json, "history");
    let profile = mode_profile(mode);
    let spans = parse_spans_or_rank(query, evidence_json, limit);
    let selected_chunks = spans
        .iter()
        .take(limit.max(1))
        .map(|span| {
            json!({
                "span_id": span.span_id,
                "source_id": span.source_id,
                "span_type": span.span_type,
                "score": span.score,
                "title": span.metadata.get("title").cloned().unwrap_or(Value::Null),
                "text": truncate_chars(&span.text, profile.chunk_chars),
            })
        })
        .collect::<Vec<_>>();
    let top_score = spans.first().map(|span| span.score).unwrap_or(0.0);
    let trust_retrieval = top_score >= 5.0
        || spans.iter().any(|span| {
            span.metadata
                .get("source")
                .and_then(Value::as_str)
                .unwrap_or("")
                .contains("selected")
        });
    let task_summary = task_summary(&task_value, query);
    let verifier_contract = verifier_contract(&task_value);
    ContextPacketLite {
        request_id: nonempty(request_id, "browser-turn"),
        created_at: "browser-runtime".to_string(),
        task: task_summary,
        control: json!({
            "mode": profile.normalized,
            "mode_profile": {
                "label": profile.label,
                "retrieval_strategy": profile.retrieval_strategy,
                "answer_policy": profile.answer_policy,
                "chunk_chars": profile.chunk_chars,
            },
            "max_context_items": limit.max(1),
            "path_confidence": confidence_from_score(top_score),
            "trust_retrieval": trust_retrieval,
            "retrieval_guidance": {
                "strategy": if trust_retrieval { profile.retrieval_strategy } else { "cautious_synthesis" },
                "selected_span_count": spans.len(),
                "top_score": top_score,
            },
            "selected_context_chunks": selected_chunks,
            "history_summary": compact_history(&history_value),
        }),
        tolbert: json!({
            "route_mode": "rust_wasm_lite",
            "tree_version": "agentkernel_lite_context_v1",
            "path_confidence": confidence_from_score(top_score),
            "note": "Browser Rust context compiler; TOLBERT model inference remains an adapter surface.",
        }),
        retrieval: json!({
            "branch_scoped": spans,
            "global": [],
        }),
        verifier_contract,
    }
}

pub fn prompt_from_context_packet(
    mode: &str,
    language: &str,
    user_text: &str,
    packet: &ContextPacketLite,
    step_index: usize,
    max_steps: usize,
) -> String {
    let profile = mode_profile(mode);
    let spans = packet
        .retrieval
        .get("branch_scoped")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let ak_context = if spans.is_empty() {
        "No research context was retrieved.".to_string()
    } else {
        spans
            .iter()
            .enumerate()
            .map(|(index, span)| {
                let title = span
                    .get("metadata")
                    .and_then(|meta| meta.get("title"))
                    .and_then(Value::as_str)
                    .unwrap_or("Untitled evidence");
                let source = span.get("source_id").and_then(Value::as_str).unwrap_or("");
                let text = span.get("text").and_then(Value::as_str).unwrap_or("");
                let metadata = span.get("metadata").cloned().unwrap_or(Value::Null);
                let category = metadata
                    .get("primary_category")
                    .or_else(|| metadata.get("category"))
                    .or_else(|| metadata.get("categories"))
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let year = metadata
                    .get("year")
                    .map(|value| {
                        value
                            .as_str()
                            .map(ToString::to_string)
                            .unwrap_or_else(|| value.to_string())
                    })
                    .unwrap_or_default();
                format!(
                    "<AK_EVIDENCE> <AK_EVIDENCE_ID> P{}\n<AK_TITLE> {}\n<AK_PAPER_ID> {}\n<AK_CATEGORY> {}\n<AK_YEAR> {}\n<AK_ABSTRACT> {}",
                    index + 1,
                    title,
                    source,
                    category,
                    year,
                    compact_whitespace(text)
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    let answer_scaffold = spans
        .first()
        .and_then(|span| span.get("text").and_then(Value::as_str))
        .map(|text| evidence_scaffold(text))
        .filter(|text| !text.is_empty())
        .unwrap_or_else(|| {
            "No answer scaffold is available; answer cautiously from the user request.".to_string()
        });
    let mode_instruction = match profile.normalized {
        "think" => "Mode: Think. Build a semantic synthesis from the retrieved evidence before answering. Keep the final response readable, but preserve important nuance and uncertainty.".to_string(),
        "deep_research" => "Mode: Deep Research. Inspect every evidence item, cite evidence numbers when making supported claims, identify conflicts or gaps, and then give a careful final synthesis. This is the slow, highest-quality mode.".to_string(),
        _ => "Mode: Chat. Reply as a helpful assistant first. Use retrieved evidence as support, cite evidence numbers only when they improve clarity, and do not turn the answer into a paper list.".to_string(),
    };
    let mode_token = match profile.normalized {
        "think" => "<AK_THINK>",
        "deep_research" => "<AK_DEEP_RESEARCH>",
        _ => "<AK_CHAT>",
    };
    if spans.is_empty() {
        return [
            format!("{mode_token} <AK_RESPOND>"),
            format!("<AK_LOOP> <AK_STATE> mode={} selected_context=0 retrieval=none", profile.normalized),
            "Return exactly this decision format: Action: respond, then Content: your direct answer.".to_string(),
            "You cannot execute, test, install, browse, or modify files from this environment.".to_string(),
            format!(
                "Agent loop: step {step_index} of {max_steps}; mode={}; code_execution=disabled; target_language={language}.",
                profile.normalized
            ),
            mode_instruction,
            format!("<AK_USER> {}", compact_whitespace(user_text)),
            "Return a structured decision with action=respond.".to_string(),
        ]
        .join("\n");
    }
    [
        format!("{mode_token} <AK_RESPOND> <AK_CONTEXT> <AK_ANSWER>"),
        format!("<AK_LOOP> <AK_STATE> mode={} selected_context={} retrieval=ranked", profile.normalized, if spans.iter().any(|span| {
            span.get("metadata")
                .and_then(|meta| meta.get("source"))
                .and_then(Value::as_str)
                .unwrap_or("")
                .contains("selected")
        }) { 1 } else { 0 }),
        "Return exactly this decision format: Action: respond, then Content: your direct answer.".to_string(),
        "You are Agent Kernel Lite running entirely in this browser.".to_string(),
        "You cannot execute, test, install, browse, or modify files from this environment."
            .to_string(),
        "Use retrieved evidence when it is relevant, and say when it is not enough. Do not invent citations, paper titles, paper ids, or source claims.".to_string(),
        "Answer the user's question directly. When using evidence, cite the evidence id such as [1] or [P1]; the interface renders the exact paper title and PDF link from that id.".to_string(),
        mode_instruction,
        format!(
            "Agent loop: step {step_index} of {max_steps}; mode={}; code_execution=disabled; target_language={language}.",
            profile.normalized
        ),
        String::new(),
        "Context packet:".to_string(),
        serde_json::to_string(packet).unwrap_or_else(|_| "{}".to_string()),
        String::new(),
        "<AK_CONTEXT> Retrieved evidence:".to_string(),
        ak_context,
        String::new(),
        "<AK_ANSWER> Answer scaffold:".to_string(),
        answer_scaffold,
        String::new(),
        format!("<AK_USER> {}", compact_whitespace(user_text)),
        "Answer directly, cite evidence like [1] only for supported claims, and keep the response conversational.".to_string(),
    ]
    .join("\n")
}

fn parse_spans_or_rank(query: &str, evidence_json: &str, limit: usize) -> Vec<RetrievedSpanLite> {
    let Ok(value) = serde_json::from_str::<Value>(evidence_json) else {
        return Vec::new();
    };
    let rows = candidate_rows(&value);
    if rows.iter().any(|row| row.get("span_id").is_some()) {
        let mut spans = rows
            .into_iter()
            .enumerate()
            .map(|(index, row)| span_from_pre_ranked(index + 1, &row))
            .collect::<Vec<_>>();
        spans.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        spans.truncate(limit.max(1));
        spans
    } else {
        rank_evidence(query, evidence_json, limit)
    }
}

fn span_from_pre_ranked(index: usize, row: &Value) -> RetrievedSpanLite {
    let source_id = first_string(row, &["source_id", "paper_id", "canonical_paper_id", "id"]);
    RetrievedSpanLite {
        span_id: nonempty(
            first_string(row, &["span_id", "id"]),
            &format!("span-{index}"),
        ),
        text: truncate_chars(
            &nonempty(
                first_string(
                    row,
                    &["text", "snippet", "context_text", "abstract", "summary"],
                ),
                "",
            ),
            MAX_SPAN_TEXT_CHARS,
        ),
        source_id,
        span_type: nonempty(
            first_string(row, &["span_type", "kind", "source"]),
            "evidence",
        ),
        score: first_number(row, &["score", "retrieval_score", "semantic_score"]),
        node_path: node_path(row),
        metadata: metadata_from_row(row),
    }
}

fn span_from_row(index: usize, row: &Value, tokens: &[String], phrase: &str) -> RetrievedSpanLite {
    let title = first_string(row, &["title", "name"]);
    let source_id = nonempty(
        first_string(
            row,
            &[
                "source_id",
                "paper_id",
                "canonical_paper_id",
                "arxiv_id",
                "id",
            ],
        ),
        &format!("candidate-{index}"),
    );
    let source = first_string(row, &["source", "dataset", "pack", "memory_source"]);
    let text = nonempty(
        first_string(
            row,
            &[
                "context_text",
                "snippet",
                "abstract",
                "summary",
                "text",
                "full_text",
                "content",
            ],
        ),
        "",
    );
    let mut score = first_number(
        row,
        &[
            "retrieval_score",
            "lexical_score",
            "score",
            "rank_score",
            "memory_score",
        ],
    );
    score += first_number(row, &["semantic_score", "embedding_score"]) * 16.0;
    score += lexical_score(&title, &text, &source_id, &source, tokens, phrase);
    if source.contains("selected") || source.contains("memory") {
        score += 3.0;
    }
    RetrievedSpanLite {
        span_id: nonempty(
            first_string(row, &["span_id"]),
            &format!("span-{index}-{}", sanitize_id(&source_id)),
        ),
        text: truncate_chars(&text, MAX_SPAN_TEXT_CHARS),
        source_id,
        span_type: nonempty(first_string(row, &["span_type", "kind"]), "evidence"),
        score,
        node_path: node_path(row),
        metadata: metadata_from_row(row),
    }
}

fn lexical_score(
    title: &str,
    text: &str,
    source_id: &str,
    source: &str,
    tokens: &[String],
    phrase: &str,
) -> f64 {
    let title_l = title.to_ascii_lowercase();
    let text_l = text.to_ascii_lowercase();
    let id_l = source_id.to_ascii_lowercase();
    let source_l = source.to_ascii_lowercase();
    let mut score = 0.0;
    if !phrase.is_empty() && title_l.contains(phrase) {
        score += 18.0;
    }
    if !phrase.is_empty() && text_l.contains(phrase) {
        score += 5.0;
    }
    for token in tokens {
        if title_l.contains(token) {
            score += 5.0;
        }
        if id_l.contains(token) {
            score += 3.0;
        }
        if source_l.contains(token) {
            score += 2.0;
        }
        if text_l.contains(token) {
            score += 1.0;
        }
    }
    score
}

fn candidate_rows(value: &Value) -> Vec<Value> {
    if let Some(rows) = value.as_array() {
        return rows.clone();
    }
    for key in ["rows", "evidence", "spans", "candidates", "documents"] {
        if let Some(rows) = value.get(key).and_then(Value::as_array) {
            return rows.clone();
        }
    }
    Vec::new()
}

fn parse_json_or_string(raw: &str, key: &str) -> Value {
    if raw.trim().is_empty() {
        return Value::Null;
    }
    serde_json::from_str::<Value>(raw).unwrap_or_else(|_| json!({ key: raw }))
}

fn task_summary(task: &Value, query: &str) -> Value {
    json!({
        "task_id": first_string(task, &["task_id", "id"]),
        "prompt": truncate_chars(&nonempty(first_string(task, &["prompt", "user_text", "task"]), query), 900),
        "workspace_subdir": first_string(task, &["workspace_subdir", "workspace"]),
        "benchmark_family": task
            .get("metadata")
            .and_then(|metadata| metadata.get("benchmark_family"))
            .and_then(Value::as_str)
            .unwrap_or("agentkernel_lite_browser"),
    })
}

fn verifier_contract(task: &Value) -> Value {
    json!({
        "success_command": first_string(task, &["success_command"]),
        "expected_files": task.get("expected_files").cloned().unwrap_or_else(|| json!([])),
        "expected_output_substrings": task.get("expected_output_substrings").cloned().unwrap_or_else(|| json!([])),
        "forbidden_files": task.get("forbidden_files").cloned().unwrap_or_else(|| json!([])),
        "forbidden_output_substrings": task.get("forbidden_output_substrings").cloned().unwrap_or_else(|| json!([])),
        "max_steps": task.get("max_steps").and_then(Value::as_u64).unwrap_or(8),
    })
}

fn compact_history(history: &Value) -> Value {
    let rows = candidate_rows(history);
    Value::Array(
        rows.into_iter()
            .rev()
            .take(3)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .map(|row| {
                json!({
                    "role": first_string(&row, &["role", "action"]),
                    "text": truncate_chars(&first_string(&row, &["text", "content", "user_text"]), 260),
                })
            })
            .collect(),
    )
}

fn evidence_scaffold(text: &str) -> String {
    let compact = compact_whitespace(text);
    if compact.is_empty() {
        return String::new();
    }
    let mut sentences = Vec::new();
    let mut current = String::new();
    for ch in compact.chars() {
        current.push(ch);
        if matches!(ch, '.' | '!' | '?') {
            let sentence = current.trim();
            if sentence.len() >= 40 {
                sentences.push(sentence.to_string());
            }
            current.clear();
            if sentences.len() >= 2 {
                break;
            }
        }
    }
    if sentences.is_empty() && !current.trim().is_empty() {
        sentences.push(current.trim().to_string());
    }
    truncate_chars(&sentences.join(" "), 420)
}

fn metadata_from_row(row: &Value) -> Value {
    json!({
        "title": first_string(row, &["title", "name"]),
        "paper_id": first_string(row, &["paper_id", "canonical_paper_id", "arxiv_id", "id"]),
        "primary_category": first_string(row, &["primary_category", "category", "categories"]),
        "year": first_string(row, &["year", "published_year"]),
        "source": first_string(row, &["source", "dataset", "pack", "memory_source"]),
        "pdf_url": first_string(row, &["pdf_url", "url"]),
    })
}

fn node_path(row: &Value) -> Vec<i64> {
    row.get("node_path")
        .and_then(Value::as_array)
        .map(|items| items.iter().filter_map(Value::as_i64).collect())
        .unwrap_or_default()
}

fn query_tokens(query: &str) -> Vec<String> {
    let mut out = Vec::new();
    for token in query
        .to_ascii_lowercase()
        .split(|ch: char| !ch.is_ascii_alphanumeric() && !matches!(ch, '_' | '+' | '-' | '.'))
    {
        if token.len() > 1 && !out.iter().any(|existing| existing == token) {
            out.push(token.to_string());
        }
    }
    out
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
        if value.is_number() || value.is_boolean() {
            return value.to_string();
        }
    }
    String::new()
}

fn first_number(row: &Value, keys: &[&str]) -> f64 {
    for key in keys {
        let Some(value) = row.get(*key) else {
            continue;
        };
        if let Some(number) = value.as_f64() {
            return number;
        }
        if let Some(text) = value.as_str() {
            if let Ok(number) = text.parse::<f64>() {
                return number;
            }
        }
    }
    0.0
}

fn confidence_from_score(score: f64) -> f64 {
    (score / 18.0).clamp(0.0, 0.98)
}

fn sanitize_id(value: &str) -> String {
    let mut out = String::new();
    for ch in value.chars().take(48) {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
        } else if matches!(ch, '.' | '-' | '_' | '/') {
            out.push('_');
        }
    }
    nonempty(out, "candidate")
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranks_title_and_context_matches() {
        let rows = r#"[
            {"title":"Unrelated", "abstract":"nothing here", "paper_id":"a"},
            {"title":"Flash attention kernel", "abstract":"tiling softmax memory traffic", "paper_id":"b"}
        ]"#;
        let ranked = rank_evidence("flash attention tiling", rows, 2);
        assert_eq!(ranked[0].source_id, "b");
        assert!(ranked[0].score > ranked[1].score);
    }

    #[test]
    fn compiles_python_shaped_context_packet() {
        let packet = compile_context_packet(
            "req-1",
            r#"{"task_id":"t","prompt":"fix attention","success_command":"pytest"}"#,
            "fix attention",
            r#"[{"title":"Attention fix","abstract":"mask orientation","paper_id":"1001.0001"}]"#,
            "[]",
            3,
            "think",
        );
        assert!(packet.control.get("retrieval_guidance").is_some());
        assert_eq!(packet.control["mode"], "think");
        assert!(packet.retrieval.get("branch_scoped").is_some());
        assert_eq!(packet.verifier_contract["success_command"], "pytest");
    }
}
