use anyhow::{Context, Result};
use reqwest::blocking::Client;

pub fn call_qwen(prompt: &str) -> Result<Option<String>> {
    let base = std::env::var("IFLOW_API_BASE").ok();
    let key = std::env::var("IFLOW_API_KEY").ok();
    let model = std::env::var("IFLOW_MODEL").unwrap_or_else(|_| "qwen3-max-preview".to_string());
    if base.is_none() || key.is_none() {
        return Ok(None);
    }
    let url = format!(
        "{}/chat/completions",
        base.as_ref().unwrap().trim_end_matches('/')
    );

    let timeout_ms: u64 = std::env::var("IFLOW_TIMEOUT_MS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(15000);
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_millis(timeout_ms))
        .build()?;

    let body = serde_json::json!({
        "model": model,
        "messages": [
            {"role": "system", "content": "You summarize ML training error logs succinctly."},
            {"role": "user", "content": prompt}
        ]
    });

    let resp = client
        .post(&url)
        .bearer_auth(key.unwrap())
        .json(&body)
        .send()
        .with_context(|| "iflow request failed")?;

    if !resp.status().is_success() {
        return Ok(None);
    }
    let v: serde_json::Value = resp.json().with_context(|| "iflow parse json")?;
    // try common shapes
    if let Some(text) = v
        .pointer("/choices/0/message/content")
        .and_then(|x: &serde_json::Value| x.as_str())
    {
        return Ok(Some(text.to_string()));
    }
    if let Some(text) = v
        .get("output_text")
        .and_then(|x: &serde_json::Value| x.as_str())
    {
        return Ok(Some(text.to_string()));
    }
    Ok(None)
}
