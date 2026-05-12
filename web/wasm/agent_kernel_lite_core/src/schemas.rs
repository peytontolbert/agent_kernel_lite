use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::actions::{compact_whitespace, ActionKind};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TaskSpecLite {
    pub task_id: String,
    pub prompt: String,
    pub workspace_subdir: String,
    #[serde(default)]
    pub setup_commands: Vec<String>,
    #[serde(default)]
    pub success_command: String,
    #[serde(default)]
    pub suggested_commands: Vec<String>,
    #[serde(default)]
    pub expected_files: Vec<String>,
    #[serde(default)]
    pub expected_output_substrings: Vec<String>,
    #[serde(default)]
    pub forbidden_files: Vec<String>,
    #[serde(default)]
    pub forbidden_output_substrings: Vec<String>,
    #[serde(default)]
    pub expected_file_contents: Value,
    #[serde(default = "default_max_steps")]
    pub max_steps: usize,
    #[serde(default)]
    pub metadata: Value,
}

impl TaskSpecLite {
    pub fn normalize(&mut self) {
        self.task_id = compact_whitespace(&self.task_id);
        self.prompt = self.prompt.trim().to_string();
        self.workspace_subdir = compact_whitespace(&self.workspace_subdir);
        self.setup_commands = normalize_string_vec(&self.setup_commands);
        self.success_command = self.success_command.trim().to_string();
        self.suggested_commands = normalize_string_vec(&self.suggested_commands);
        self.expected_files = normalize_string_vec(&self.expected_files);
        self.expected_output_substrings = normalize_string_vec(&self.expected_output_substrings);
        self.forbidden_files = normalize_string_vec(&self.forbidden_files);
        self.forbidden_output_substrings = normalize_string_vec(&self.forbidden_output_substrings);
        self.max_steps = self.max_steps.max(1);
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.task_id.is_empty() {
            return Err("task_id must not be empty".to_string());
        }
        if self.prompt.trim().is_empty() {
            return Err("prompt must not be empty".to_string());
        }
        if self.workspace_subdir.is_empty() {
            return Err("workspace_subdir must not be empty".to_string());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActionDecisionLite {
    #[serde(default)]
    pub thought: String,
    pub action: ActionKind,
    #[serde(default)]
    pub content: String,
    #[serde(default)]
    pub done: bool,
    #[serde(default)]
    pub selected_retrieval_span_id: Option<String>,
    #[serde(default)]
    pub retrieval_influenced: bool,
    #[serde(default = "default_decision_source")]
    pub decision_source: String,
    #[serde(default)]
    pub proposal_metadata: Value,
}

impl ActionDecisionLite {
    pub fn respond(content: String) -> ActionDecisionLite {
        ActionDecisionLite {
            thought: String::new(),
            action: ActionKind::Respond,
            content,
            done: false,
            selected_retrieval_span_id: None,
            retrieval_influenced: false,
            decision_source: default_decision_source(),
            proposal_metadata: Value::Null,
        }
    }

    pub fn normalize(&mut self) {
        self.thought = self.thought.trim().to_string();
        self.content = self.content.trim().to_string();
        self.decision_source = compact_whitespace(&self.decision_source);
        if self.decision_source.is_empty() {
            self.decision_source = default_decision_source();
        }
        self.selected_retrieval_span_id = self
            .selected_retrieval_span_id
            .as_ref()
            .map(|value| compact_whitespace(value))
            .filter(|value| !value.is_empty());
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MessageRecord {
    pub role: String,
    pub text: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvidenceRecord {
    pub index: usize,
    pub title: String,
    pub paper_id: String,
    pub primary_category: String,
    pub year: String,
    pub source: String,
    pub snippet: String,
    pub pdf_url: String,
    pub score: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RetrievedSpanLite {
    pub span_id: String,
    pub text: String,
    pub source_id: String,
    pub span_type: String,
    pub score: f64,
    #[serde(default)]
    pub node_path: Vec<i64>,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContextPacketLite {
    pub request_id: String,
    pub created_at: String,
    pub task: Value,
    pub control: Value,
    pub tolbert: Value,
    pub retrieval: Value,
    pub verifier_contract: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecisionPacketLite {
    pub parse_status: String,
    pub raw_model_text: String,
    pub decision: ActionDecisionLite,
    #[serde(default)]
    pub action_contract: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TurnRecord {
    pub step_index: usize,
    pub mode: String,
    pub language: String,
    pub user_text: String,
    pub evidence_count: usize,
    pub max_new_tokens: u32,
    pub completed: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StepRecordLite {
    pub index: usize,
    pub action: ActionKind,
    pub content: String,
    #[serde(default)]
    pub verification: Value,
    #[serde(default)]
    pub runtime_attestation: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EpisodeRecordLite {
    pub task_id: String,
    pub prompt: String,
    pub workspace: String,
    pub success: bool,
    #[serde(default)]
    pub task_metadata: Value,
    #[serde(default)]
    pub task_contract: Value,
    #[serde(default)]
    pub termination_reason: String,
    #[serde(default)]
    pub steps: Vec<StepRecordLite>,
    #[serde(default)]
    pub messages: Vec<MessageRecord>,
    #[serde(default)]
    pub action_ledger: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RuntimeConfigLite {
    #[serde(default = "default_model_id")]
    pub model_id: String,
    #[serde(default = "default_storage_mode")]
    pub storage_mode: String,
    #[serde(default)]
    pub pack_id: String,
    #[serde(default)]
    pub extension_policy: String,
}

fn normalize_string_vec(values: &[String]) -> Vec<String> {
    values
        .iter()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .collect()
}

fn default_max_steps() -> usize {
    5
}

fn default_decision_source() -> String {
    "model".to_string()
}

fn default_model_id() -> String {
    "agentkernel-lite-encdec".to_string()
}

fn default_storage_mode() -> String {
    "browser".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_task_spec_like_python() {
        let mut spec: TaskSpecLite = serde_json::from_str(
            r#"{
                "task_id": " task-1 ",
                "prompt": " hello ",
                "workspace_subdir": " work ",
                "setup_commands": [" ", "echo hi"],
                "max_steps": 0
            }"#,
        )
        .expect("task spec parses");
        spec.normalize();
        assert_eq!(spec.task_id, "task-1");
        assert_eq!(spec.setup_commands, vec!["echo hi"]);
        assert_eq!(spec.max_steps, 1);
        spec.validate().expect("normalized spec is valid");
    }

    #[test]
    fn episode_record_roundtrips() {
        let episode = EpisodeRecordLite {
            task_id: "browser-chat".to_string(),
            prompt: "help".to_string(),
            workspace: "browser".to_string(),
            success: true,
            task_metadata: Value::Null,
            task_contract: Value::Null,
            termination_reason: "done".to_string(),
            steps: vec![StepRecordLite {
                index: 1,
                action: ActionKind::Respond,
                content: "ok".to_string(),
                verification: Value::Null,
                runtime_attestation: Value::Null,
            }],
            messages: vec![MessageRecord {
                role: "assistant".to_string(),
                text: "ok".to_string(),
            }],
            action_ledger: Value::Null,
        };
        let encoded = serde_json::to_string(&episode).expect("serializes");
        let decoded: EpisodeRecordLite = serde_json::from_str(&encoded).expect("deserializes");
        assert_eq!(decoded.steps.len(), 1);
    }
}
