use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionKind {
    Respond,
    GatherContext,
    CodeExecute,
    ProposeCode,
    ProposeArtifactWrite,
    Retrieve,
    SaveMemory,
    ExtensionRequest,
    AskUser,
    Done,
}

impl ActionKind {
    pub fn normalized(value: &str) -> ActionKind {
        match value.trim().to_ascii_lowercase().as_str() {
            "gather_context" | "gather-context" | "retrieve" | "retrieval" => {
                ActionKind::GatherContext
            }
            "code_execute" | "execute_code" | "shell" | "command" => ActionKind::CodeExecute,
            "propose_code" | "code" => ActionKind::ProposeCode,
            "propose_artifact_write" | "artifact_write" | "write_artifact" => {
                ActionKind::ProposeArtifactWrite
            }
            "save_memory" | "memory" => ActionKind::SaveMemory,
            "extension_request" | "extension" => ActionKind::ExtensionRequest,
            "ask_user" | "question" => ActionKind::AskUser,
            "done" | "finish" => ActionKind::Done,
            _ => ActionKind::Respond,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            ActionKind::Respond => "respond",
            ActionKind::GatherContext => "gather_context",
            ActionKind::CodeExecute => "code_execute",
            ActionKind::ProposeCode => "propose_code",
            ActionKind::ProposeArtifactWrite => "propose_artifact_write",
            ActionKind::Retrieve => "retrieve",
            ActionKind::SaveMemory => "save_memory",
            ActionKind::ExtensionRequest => "extension_request",
            ActionKind::AskUser => "ask_user",
            ActionKind::Done => "done",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActionLedgerRecord {
    pub action_id: String,
    pub kind: ActionKind,
    pub requested_capability: String,
    pub input: Value,
    pub requires_user_approval: bool,
    pub policy_state: String,
    pub created_turn: usize,
    pub model_trace_id: String,
    pub receipt: Option<Value>,
}

impl ActionLedgerRecord {
    pub fn pending_extension(
        action_id: String,
        requested_capability: String,
        input: Value,
        created_turn: usize,
        model_trace_id: String,
    ) -> ActionLedgerRecord {
        ActionLedgerRecord {
            action_id,
            kind: ActionKind::ExtensionRequest,
            requested_capability,
            input,
            requires_user_approval: true,
            policy_state: "pending".to_string(),
            created_turn,
            model_trace_id,
            receipt: None,
        }
    }
}

pub fn compact_whitespace(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_action_aliases() {
        assert_eq!(
            ActionKind::normalized("retrieve"),
            ActionKind::GatherContext
        );
        assert_eq!(
            ActionKind::normalized("gather-context"),
            ActionKind::GatherContext
        );
        assert_eq!(ActionKind::normalized("code"), ActionKind::ProposeCode);
        assert_eq!(
            ActionKind::normalized("code_execute"),
            ActionKind::CodeExecute
        );
        assert_eq!(
            ActionKind::normalized("extension"),
            ActionKind::ExtensionRequest
        );
        assert_eq!(ActionKind::normalized("unknown"), ActionKind::Respond);
    }
}
