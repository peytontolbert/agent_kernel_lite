use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::actions::compact_whitespace;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalPolicy {
    AlwaysAsk,
    RememberForSession,
    Disabled,
    TrustedLocal,
}

impl Default for ApprovalPolicy {
    fn default() -> Self {
        ApprovalPolicy::AlwaysAsk
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExtensionCapability {
    pub id: String,
    pub description: String,
    #[serde(default)]
    pub scopes: Vec<String>,
}

impl ExtensionCapability {
    pub fn normalize(&mut self) {
        self.id = compact_whitespace(&self.id);
        self.description = compact_whitespace(&self.description);
        self.scopes = self
            .scopes
            .iter()
            .map(|scope| compact_whitespace(scope))
            .filter(|scope| !scope.is_empty())
            .collect();
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExtensionManifest {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub approval_policy: ApprovalPolicy,
    #[serde(default)]
    pub capabilities: Vec<ExtensionCapability>,
    #[serde(default)]
    pub metadata: Value,
}

impl ExtensionManifest {
    pub fn normalize(&mut self) {
        self.id = compact_whitespace(&self.id);
        self.name = compact_whitespace(&self.name);
        self.version = compact_whitespace(&self.version);
        for capability in &mut self.capabilities {
            capability.normalize();
        }
        self.capabilities
            .retain(|capability| !capability.id.is_empty());
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.id.is_empty() {
            return Err("extension manifest id must not be empty".to_string());
        }
        if self.name.is_empty() {
            return Err("extension manifest name must not be empty".to_string());
        }
        if self.capabilities.is_empty() {
            return Err("extension manifest must declare at least one capability".to_string());
        }
        Ok(())
    }

    pub fn supports_capability(&self, capability_id: &str) -> bool {
        let normalized = compact_whitespace(capability_id);
        self.capabilities
            .iter()
            .any(|capability| capability.id == normalized)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExtensionActionReceipt {
    pub action_id: String,
    pub status: String,
    #[serde(default)]
    pub output: Value,
    #[serde(default)]
    pub artifact_refs: Vec<String>,
    #[serde(default)]
    pub error: String,
}

impl ExtensionActionReceipt {
    pub fn normalize(&mut self) {
        self.action_id = compact_whitespace(&self.action_id);
        self.status = compact_whitespace(&self.status);
        self.artifact_refs = self
            .artifact_refs
            .iter()
            .map(|value| compact_whitespace(value))
            .filter(|value| !value.is_empty())
            .collect();
        self.error = compact_whitespace(&self.error);
    }
}

pub fn parse_manifest(raw_json: &str) -> Result<ExtensionManifest, String> {
    let mut manifest: ExtensionManifest = serde_json::from_str(raw_json)
        .map_err(|exc| format!("invalid extension manifest JSON: {exc}"))?;
    manifest.normalize();
    manifest.validate()?;
    Ok(manifest)
}

pub fn parse_receipt(raw_json: &str) -> Result<ExtensionActionReceipt, String> {
    let mut receipt: ExtensionActionReceipt = serde_json::from_str(raw_json)
        .map_err(|exc| format!("invalid extension receipt JSON: {exc}"))?;
    receipt.normalize();
    if receipt.action_id.is_empty() {
        return Err("extension receipt action_id must not be empty".to_string());
    }
    if receipt.status.is_empty() {
        return Err("extension receipt status must not be empty".to_string());
    }
    Ok(receipt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_extension_manifest() {
        let raw = r#"{
            "id": " github ",
            "name": " GitHub ",
            "capabilities": [{"id": " github.read_repo ", "description": " Read repository "}]
        }"#;
        let manifest = parse_manifest(raw).expect("manifest should parse");
        assert_eq!(manifest.id, "github");
        assert!(manifest.supports_capability("github.read_repo"));
        assert_eq!(manifest.approval_policy, ApprovalPolicy::AlwaysAsk);
    }
}
