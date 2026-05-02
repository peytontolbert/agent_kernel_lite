use serde_json::{json, Value};

use crate::actions::{compact_whitespace, ActionKind};
use crate::schemas::{ActionDecisionLite, DecisionPacketLite};

pub fn parse_model_decision(
    model_text: &str,
    mode: &str,
    options_json: &str,
) -> DecisionPacketLite {
    let options = serde_json::from_str::<Value>(options_json).unwrap_or(Value::Null);
    let code_execution_enabled = options
        .get("code_execution_enabled")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let raw = trim_generated_text(model_text);
    let (parse_status, decision) = if let Some(value) = parse_json_decision(&raw) {
        ("json".to_string(), decision_from_value(&value, &raw))
    } else if let Some(value) = parse_line_decision(&raw) {
        (
            "line_protocol".to_string(),
            decision_from_value(&value, &raw),
        )
    } else {
        ("plain_text".to_string(), fallback_decision(&raw, mode))
    };
    let browser_safe_action =
        if decision.action == ActionKind::CodeExecute && !code_execution_enabled {
            ActionKind::ProposeCode.as_str()
        } else {
            decision.action.as_str()
        };
    DecisionPacketLite {
        parse_status,
        raw_model_text: raw,
        action_contract: json!({
            "code_execution_enabled": code_execution_enabled,
            "browser_safe_action": browser_safe_action,
            "structured_actions": [
                "respond",
                "gather_context",
                "extension_request",
                "ask_user",
                "done"
            ],
            "lite_loop": {
                "commands": ["respond", "gather_context"],
                "gather_context": "request ranked paper/context retrieval before the final response",
                "respond": "answer the user directly; use retrieved context only when relevant"
            },
        }),
        decision,
    }
}

pub fn decision_packet_json(model_text: &str, mode: &str, options_json: &str) -> String {
    serde_json::to_string(&parse_model_decision(model_text, mode, options_json)).unwrap_or_else(
        |error| {
            json!({
                "parse_status": "error",
                "error": format!("decision serialization failed: {error}"),
                "raw_model_text": model_text,
            })
            .to_string()
        },
    )
}

fn parse_json_decision(raw: &str) -> Option<Value> {
    if let Ok(value) = serde_json::from_str::<Value>(raw) {
        if value.is_object() {
            return Some(value);
        }
    }
    let fenced = raw
        .trim()
        .strip_prefix("```json")
        .or_else(|| raw.trim().strip_prefix("```"))
        .and_then(|text| text.rsplit_once("```").map(|(body, _)| body.trim()));
    if let Some(body) = fenced {
        if let Ok(value) = serde_json::from_str::<Value>(body) {
            if value.is_object() {
                return Some(value);
            }
        }
    }
    extract_first_json_object(raw).and_then(|body| serde_json::from_str::<Value>(&body).ok())
}

fn parse_line_decision(raw: &str) -> Option<Value> {
    let mut action = String::new();
    let mut thought = String::new();
    let mut content = Vec::new();
    let mut in_content = false;
    for line in raw.lines() {
        let trimmed = line.trim();
        let lower = trimmed.to_ascii_lowercase();
        if let Some(rest) = lower.strip_prefix("action:") {
            action = rest.trim().to_string();
            in_content = false;
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("Action:") {
            action = rest.trim().to_string();
            in_content = false;
            continue;
        }
        if let Some(rest) = trimmed
            .strip_prefix("thought:")
            .or_else(|| trimmed.strip_prefix("Thought:"))
        {
            thought = rest.trim().to_string();
            in_content = false;
            continue;
        }
        if let Some(rest) = trimmed
            .strip_prefix("content:")
            .or_else(|| trimmed.strip_prefix("Content:"))
        {
            content.push(rest.trim().to_string());
            in_content = true;
            continue;
        }
        if in_content {
            content.push(line.to_string());
        }
    }
    if action.is_empty() {
        return None;
    }
    Some(json!({
        "thought": thought,
        "action": action,
        "content": content.join("\n"),
    }))
}

fn decision_from_value(value: &Value, raw: &str) -> ActionDecisionLite {
    let mut decision = ActionDecisionLite {
        thought: first_string(value, &["thought", "reasoning", "rationale"]),
        action: ActionKind::normalized(&first_string(value, &["action", "type"])),
        content: nonempty(
            first_string(value, &["content", "answer", "command", "code"]),
            raw,
        ),
        done: value.get("done").and_then(Value::as_bool).unwrap_or(false),
        selected_retrieval_span_id: optional_string(
            value,
            &["selected_retrieval_span_id", "span_id"],
        ),
        retrieval_influenced: value
            .get("retrieval_influenced")
            .and_then(Value::as_bool)
            .unwrap_or_else(|| {
                optional_string(value, &["selected_retrieval_span_id", "span_id"]).is_some()
            }),
        decision_source: nonempty(first_string(value, &["decision_source", "source"]), "model"),
        proposal_metadata: value
            .get("proposal_metadata")
            .cloned()
            .unwrap_or_else(|| json!({})),
    };
    if decision.action == ActionKind::Done {
        decision.done = true;
    }
    decision.normalize();
    decision
}

fn fallback_decision(raw: &str, mode: &str) -> ActionDecisionLite {
    let action = if mode.trim().eq_ignore_ascii_case("code") && raw.contains("```") {
        ActionKind::ProposeCode
    } else {
        ActionKind::Respond
    };
    let mut decision = ActionDecisionLite {
        thought: String::new(),
        action,
        content: raw.to_string(),
        done: false,
        selected_retrieval_span_id: None,
        retrieval_influenced: false,
        decision_source: "model_fallback".to_string(),
        proposal_metadata: json!({}),
    };
    decision.normalize();
    decision
}

fn extract_first_json_object(raw: &str) -> Option<String> {
    let mut start = None;
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;
    for (index, ch) in raw.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        if ch == '"' {
            in_string = true;
            continue;
        }
        if ch == '{' {
            if depth == 0 {
                start = Some(index);
            }
            depth += 1;
        } else if ch == '}' && depth > 0 {
            depth -= 1;
            if depth == 0 {
                if let Some(start_index) = start {
                    return Some(raw[start_index..=index].to_string());
                }
            }
        }
    }
    None
}

fn trim_generated_text(text: &str) -> String {
    let mut out = text.trim().to_string();
    if let Some(index) = out.rfind("Assistant:") {
        out = out[index + "Assistant:".len()..].trim().to_string();
    }
    out
}

fn first_string(value: &Value, keys: &[&str]) -> String {
    for key in keys {
        let Some(item) = value.get(*key) else {
            continue;
        };
        if let Some(text) = item.as_str() {
            let normalized = compact_whitespace(text);
            if !normalized.is_empty() {
                return normalized;
            }
        }
        if item.is_number() || item.is_boolean() {
            return item.to_string();
        }
    }
    String::new()
}

fn optional_string(value: &Value, keys: &[&str]) -> Option<String> {
    let out = first_string(value, keys);
    if out.is_empty() {
        None
    } else {
        Some(out)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_json_decision_with_code_execute() {
        let packet = parse_model_decision(
            r#"{"thought":"patch","action":"code_execute","content":"python test.py","selected_retrieval_span_id":"s1"}"#,
            "code",
            "{}",
        );
        assert_eq!(packet.decision.action, ActionKind::CodeExecute);
        assert_eq!(
            packet.action_contract["browser_safe_action"],
            "propose_code"
        );
        assert!(packet.decision.retrieval_influenced);
    }

    #[test]
    fn plain_code_mode_falls_back_to_propose_code() {
        let packet = parse_model_decision("```python\nprint(1)\n```", "code", "{}");
        assert_eq!(packet.decision.action, ActionKind::ProposeCode);
    }
}
