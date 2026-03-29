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

    // Optionally call LLM (if env present) — TODO: make async once summarize is called from async context
    let llm_note: Option<String> = None;

    Summary { error_code, context: lines.iter().rev().take(10).cloned().collect::<Vec<_>>().join("\n"), llm_note }
}
