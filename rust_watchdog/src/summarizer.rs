use regex::Regex;

#[derive(Debug, Clone)]
pub struct Summary {
    pub error_code: Option<String>,
    pub context: String,
    pub llm_note: Option<String>,
}

pub fn summarize(lines: &[String]) -> Summary {
    let text = lines.join("\n");
    let re_oom = Regex::new(r"(?i)CUDA\s+(out of memory|OOM)").unwrap();
    let error_code = if re_oom.is_match(&text) { Some("CUDA_OOM".into()) } else { None };

    // Optionally call LLM (if env present)
    let llm_note = match crate::llm::call_qwen(&format!(
        "Please summarize the following training error logs in 1-2 sentences and suggest one fix if obvious.\n\n{}",
        &text
    )) {
        Ok(Some(s)) => Some(s),
        _ => None,
    };

    Summary { error_code, context: lines.iter().rev().take(10).cloned().collect::<Vec<_>>().join("\n"), llm_note }
}
