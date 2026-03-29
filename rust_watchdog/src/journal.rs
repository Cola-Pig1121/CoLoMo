use chrono::Utc;
use serde::Serialize;
use serde_json::Value;
use std::fs::OpenOptions;
use std::io::Write;

#[derive(Serialize)]
pub struct JournalEntry<T: serde::Serialize> {
    pub ts: String,
    pub action: String,
    pub details: T,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inverse: Option<Value>,
}

pub fn append<T: serde::Serialize>(journal_path: &str, action: &str, details: &T, inverse: Option<Value>) -> anyhow::Result<()> {
    let entry = JournalEntry {
        ts: Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        action: action.to_string(),
        details,
        inverse,
    };
    let line = serde_json::to_string(&entry)?;
    let mut f = OpenOptions::new().create(true).append(true).open(journal_path)?;
    writeln!(f, "{}", line)?;
    Ok(())
}

pub fn snapshot_config(path: &std::path::Path) -> Value {
    match std::fs::read_to_string(path) {
        Ok(text) => serde_yaml::from_str(&text).unwrap_or(Value::Null),
        Err(_) => Value::Null,
    }
}

pub fn restore_config(path: &std::path::Path, old: &Value) -> anyhow::Result<()> {
    let text = serde_yaml::to_string(old).unwrap_or_default();
    std::fs::write(path, text).map_err(Into::into)
}

pub fn pop_entries(journal_path: &str, steps: usize) -> anyhow::Result<Vec<serde_json::Value>> {
    let content = std::fs::read_to_string(journal_path).unwrap_or_default();
    let all_lines: Vec<&str> = content.lines().collect();
    let keep = all_lines.len().saturating_sub(steps);
    let entries: Vec<serde_json::Value> = all_lines[keep..]
        .iter()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();
    let new_content: String = all_lines[..keep].join("\n");
    std::fs::write(journal_path, new_content)?;
    Ok(entries)
}
