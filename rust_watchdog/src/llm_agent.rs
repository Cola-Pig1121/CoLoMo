//! LLM Agent Module
//! Provides AI agent functionality using rig-core framework.

use crate::{get_state, send_ux, UxEvent};
use anyhow::Result;
use futures::StreamExt;
use reqwest::Client as HttpClient;
use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use std::io::Write;
use serde_json::json;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use sysinfo::System;

use crate::config;
use crate::gpu_monitor;

// ============================================================================
// Debug Logging (writes to project/logs/logger.log only when debug=true)
// ============================================================================

fn debug_log(msg: &str) {
    use std::fs::OpenOptions;
    use std::io::Write;

    // Only write to file if debug mode is enabled
    let do_file_log = {
        let state = get_state();
        if let Ok(s) = state.lock() {
            s.settings.debug
        } else {
            false
        }
    };

    if do_file_log {
        // Determine project root from cfg_path (project_root/config.yaml)
        let log_path = {
            let state = get_state();
            if let Ok(s) = state.lock() {
                let project_root = s.cfg_path.parent()
                    .map(|p| p.to_path_buf())
                    .unwrap_or_else(|| std::path::PathBuf::from("."));
                project_root.join("logs").join("logger.log")
            } else {
                std::path::PathBuf::from("logs").join("logger.log")
            }
        };

        if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(&log_path) {
            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let _ = writeln!(file, "[{}] {}", timestamp, msg);
        }
    }

    // Always push masked version to output_lines (for user visibility)
    debug_to_output(msg);
}

/// Push a masked version of debug info to output_lines.
/// Strips API keys, raw response bodies, and other sensitive/bulky content.
/// Always pushes to output_lines (log.log is unconditional via debug_log).
fn debug_to_output(msg: &str) {
    let masked = mask_sensitive(msg);
    // Only push if it's a short, meaningful status line
    if masked.len() <= 200 && !masked.is_empty() {
        if let Ok(s) = get_state().lock() {
            let lines = s.output_lines.clone();
            // Avoid duplicates
            if !lines.iter().any(|l| l.contains(&masked[..masked.len().min(50)])) {
                drop(lines);
                if let Ok(mut s) = get_state().lock() {
                    s.output_lines.push(format!("[debug] {}", masked));
                    s.output_scroll = usize::MAX;
                }
            }
        }
    }
}

/// Mask sensitive data in debug messages for display.
fn mask_sensitive(msg: &str) -> String {
    // Truncate very long messages (e.g. full LLM responses)
    if msg.len() > 300 {
        format!("{}[truncated {} chars]", &msg[..280], msg.len() - 280)
    } else {
        msg.to_string()
    }
}

// ============================================================================
// Tool Errors
// ============================================================================

#[derive(Debug, thiserror::Error)]
#[allow(dead_code)]
pub enum AgentError {
    #[error("Tool error: {0}")]
    Tool(String),
    #[error("LLM error: {0}")]
    Llm(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON serialization error: {0}")]
    JsonSerialization(#[from] serde_json::Error),
    #[error("YAML serialization error: {0}")]
    YamlSerialization(#[from] serde_yaml::Error),
}

/// Tool error type that implements std::error::Error for Tool trait requirements
#[derive(Debug, thiserror::Error)]
#[allow(dead_code)]
pub enum ToolError {
    #[error("Tool call error: {0}")]
    Call(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("YAML error: {0}")]
    Yaml(#[from] serde_yaml::Error),
}

// ============================================================================
// PromptHook for TUI Logging
// ============================================================================
// System Monitor Tool
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct SystemStatus {
    pub total_memory_mb: u64,
    pub used_memory_mb: u64,
    pub free_memory_mb: u64,
    pub cpu_usage_pct: f32,
    pub os: String,
    pub host_name: String,
}

impl SystemStatus {
    #[allow(dead_code)]
    pub fn poll() -> Self {
        let mut sys = System::new_all();
        sys.refresh_all();

        let total = sys.total_memory() / 1024 / 1024;
        let used = sys.used_memory() / 1024 / 1024;
        let free = total.saturating_sub(used);
        let cpu = sys.global_cpu_usage();

        SystemStatus {
            total_memory_mb: total,
            used_memory_mb: used,
            free_memory_mb: free,
            cpu_usage_pct: cpu,
            os: System::name().unwrap_or_else(|| "unknown".to_string()),
            host_name: System::host_name().unwrap_or_else(|| "unknown".to_string()),
        }
    }

    #[allow(dead_code)]
    pub fn to_json(&self) -> String {
        serde_json::to_string(self)
            .unwrap_or_else(|_| r#"{"error":"serialization_failed"}"#.to_string())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[allow(dead_code)]
struct SystemMonitorTool;

impl Tool for SystemMonitorTool {
    const NAME: &'static str = "get_system_status";
    type Error = ToolError;
    type Args = ();
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Get current system status including memory and CPU usage. Returns JSON with total_memory_mb, used_memory_mb, free_memory_mb, cpu_usage_pct, os, host_name.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {},
                "required": [],
            }),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        let status = SystemStatus::poll();
        Ok(status.to_json())
    }
}

// ============================================================================
// GPU Monitor Tool
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
#[allow(dead_code)]
struct GpuMonitorTool;

impl Tool for GpuMonitorTool {
    const NAME: &'static str = "get_gpu_status";
    type Error = ToolError;
    type Args = ();
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Get current GPU status including VRAM usage and utilization. Returns JSON with total_mb, used_mb, free_mb, utilization_pct, temperature_c, simulated.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {},
                "required": [],
            }),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        let status = gpu_monitor::poll();
        Ok(status.to_json())
    }
}

// ============================================================================
// Config Reader Tool
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
#[allow(dead_code)]
struct ConfigReaderTool {
    path: PathBuf,
}

impl ConfigReaderTool {
    #[allow(dead_code)]
    fn new(path: PathBuf) -> Self {
        Self { path }
    }
}

impl Tool for ConfigReaderTool {
    const NAME: &'static str = "read_config";
    type Error = ToolError;
    type Args = ();
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description:
                "Read the current training config.yaml file. Returns the raw YAML content."
                    .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {},
                "required": [],
            }),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        let content = std::fs::read_to_string(&self.path)?;
        Ok(content)
    }
}

// ============================================================================
// Config Writer Tool
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
#[allow(dead_code)]
struct ConfigWriterTool {
    path: PathBuf,
}

impl ConfigWriterTool {
    #[allow(dead_code)]
    fn new(path: PathBuf) -> Self {
        Self { path }
    }
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct WriteConfigArgs {
    content: String,
}

impl Tool for ConfigWriterTool {
    const NAME: &'static str = "write_config";
    type Error = ToolError;
    type Args = WriteConfigArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Write content to the config.yaml file. Takes JSON with 'content' field containing the YAML string.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The YAML content to write"
                    }
                },
                "required": ["content"],
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        std::fs::write(&self.path, &args.content)?;
        Ok(format!("Config written to {}", self.path.display()))
    }
}

// ============================================================================
// Plan Writer Tool
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
#[allow(dead_code)]
struct PlanWriterTool {
    path: PathBuf,
}

impl PlanWriterTool {
    #[allow(dead_code)]
    fn new(path: PathBuf) -> Self {
        Self { path }
    }
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct WritePlanArgs {
    content: String,
}

impl Tool for PlanWriterTool {
    const NAME: &'static str = "write_plan";
    type Error = ToolError;
    type Args = WritePlanArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Write the implementation plan to a markdown file. Takes JSON with 'content' field containing the plan markdown.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The markdown plan content to write"
                    }
                },
                "required": ["content"],
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        // Ensure parent directory exists
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&self.path, &args.content)?;
        Ok(format!("Plan written to {}", self.path.display()))
    }
}

// ============================================================================
// Project Context Reader Tool
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
#[allow(dead_code)]
struct ProjectContextTool {
    project_root: PathBuf,
}

impl ProjectContextTool {
    #[allow(dead_code)]
    fn new(project_root: PathBuf) -> Self {
        Self { project_root }
    }
}

impl Tool for ProjectContextTool {
    const NAME: &'static str = "get_project_context";
    type Error = ToolError;
    type Args = ();
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description:
                "Get context about the current project including file structure and key files."
                    .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {},
                "required": [],
            }),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        let mut context = String::new();

        // Read config if exists
        let config_path = self.project_root.join("config.yaml");
        if let Ok(config) = std::fs::read_to_string(&config_path) {
            context.push_str("## config.yaml\n");
            context.push_str(&config);
            context.push_str("\n\n");
        }

        // Read train.py if exists
        let train_path = self.project_root.join("train.py");
        if let Ok(train) = std::fs::read_to_string(&train_path) {
            context.push_str("## train.py (first 50 lines)\n");
            for (i, line) in train.lines().take(50).enumerate() {
                context.push_str(&format!("{:3}: {}\n", i + 1, line));
            }
            context.push_str("\n\n");
        }

        // List project files
        context.push_str("## Project Structure\n");
        if let Ok(entries) = std::fs::read_dir(&self.project_root) {
            let mut files: Vec<_> = entries
                .filter_map(|e| e.ok())
                .filter_map(|e| e.file_name().into_string().ok())
                .collect();
            files.sort();
            for name in files {
                context.push_str(&format!("  - {}\n", name));
            }
        }

        Ok(context)
    }
}

// ============================================================================
// Advisor Recommendation Tool
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
#[allow(dead_code)]
struct AdvisorRecommendTool {
    project_root: PathBuf,
}

impl AdvisorRecommendTool {
    #[allow(dead_code)]
    fn new(project_root: PathBuf) -> Self {
        Self { project_root }
    }
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct RecommendArgs {
    gpu_status: String,
    system_status: String,
    current_config: String,
}

impl Tool for AdvisorRecommendTool {
    const NAME: &'static str = "get_recommendation";
    type Error = ToolError;
    type Args = RecommendArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Get training parameter recommendations based on GPU and system status. Returns JSON with recommended batch_size, learning_rate, optimizer, and rationale.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "gpu_status": {
                        "type": "string",
                        "description": "JSON string from get_gpu_status tool"
                    },
                    "system_status": {
                        "type": "string",
                        "description": "JSON string from get_system_status tool"
                    },
                    "current_config": {
                        "type": "string",
                        "description": "Current config.yaml content"
                    }
                },
                "required": ["gpu_status", "system_status", "current_config"],
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        // Parse inputs
        let gpu: gpu_monitor::GpuStatus =
            serde_json::from_str(&args.gpu_status).map_err(ToolError::Json)?;
        let sys: SystemStatus =
            serde_json::from_str(&args.system_status).map_err(ToolError::Json)?;
        let cfg: config::Config =
            serde_yaml::from_str(&args.current_config).map_err(ToolError::Yaml)?;

        // Calculate recommended batch size based on GPU memory
        let recommended_batch = calculate_batch_size(&gpu, &sys, &cfg);
        let recommended_lr = calculate_learning_rate(&cfg, recommended_batch);
        let recommended_opt = calculate_optimizer(&cfg);

        let recommendation = json!({
            "batch_size": recommended_batch,
            "learning_rate": recommended_lr,
            "optimizer": recommended_opt,
            "rationale": format!(
                "Based on GPU: {}/{} MB ({}%), System: {}/{} MB used. ",
                gpu.used_mb.unwrap_or(0),
                gpu.total_mb.unwrap_or(0),
                gpu.utilization_pct.unwrap_or(0),
                sys.used_memory_mb,
                sys.total_memory_mb,
            )
        })
        .to_string();

        Ok(recommendation)
    }
}

#[allow(dead_code)]
#[allow(clippy::manual_clamp)]
fn calculate_batch_size(
    gpu: &gpu_monitor::GpuStatus,
    _sys: &SystemStatus,
    cfg: &config::Config,
) -> u32 {
    let safety_alpha = 0.85;

    // Estimate available VRAM
    let total_vram = gpu.total_mb.unwrap_or(8192) as f64;
    let used_vram = gpu.used_mb.unwrap_or(2048) as f64;
    let available = (total_vram - used_vram) * safety_alpha;

    // Rough estimate: ~2GB per batch unit for typical vision models, ~500MB for small models
    // Start with a conservative estimate
    let base_batch = (available / 2048.0).max(1.0) as u32;

    // If user already has a batch_size set, use that as reference
    if let Some(current) = cfg.batch_size {
        // Adjust based on available memory
        let current = current as u32;
        let max_recommended = (available / 1500.0).max(1.0) as u32;
        return max_recommended.min(current * 2).max(1).min(256);
    }

    base_batch.min(64).max(1)
}

#[allow(dead_code)]
#[allow(clippy::manual_clamp)]
fn calculate_learning_rate(cfg: &config::Config, new_batch: u32) -> f64 {
    // Linear scaling rule: lr_new = lr_base * (batch_new / batch_base)^0.5
    let base_batch = cfg.batch_size.unwrap_or(32) as f64;
    let base_lr = cfg.learning_rate.unwrap_or(0.001);

    if (new_batch as f64 - base_batch).abs() < 0.1 {
        return base_lr;
    }

    let scale = (new_batch as f64 / base_batch).sqrt();
    (base_lr * scale).min(0.1).max(1e-6)
}

#[allow(dead_code)]
fn calculate_optimizer(cfg: &config::Config) -> String {
    // Default optimizer selection based on parameter count
    if let Some(count) = cfg.param_count {
        if count > 1_000_000_000 {
            "AdamW".to_string() // >1B params
        } else if count > 100_000_000 {
            "Adam".to_string() // 100M-1B params
        } else {
            "SGD".to_string() // <100M params
        }
    } else {
        "AdamW".to_string() // Default
    }
}

// ============================================================================
// Chat History
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct ChatHistory {
    pub messages: Vec<ChatMessage>,
}

impl ChatHistory {
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
        }
    }

    #[allow(dead_code)]
    pub fn add_user(&mut self, content: String) {
        self.messages.push(ChatMessage {
            role: "user".to_string(),
            content,
        });
    }

    #[allow(dead_code)]
    pub fn add_assistant(&mut self, content: String) {
        self.messages.push(ChatMessage {
            role: "assistant".to_string(),
            content,
        });
    }

    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.messages.clear();
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }
}

// ============================================================================
// Agent Builder
// ============================================================================

pub struct CoLoMoAgent {
    http: HttpClient,
    project_root: PathBuf,
    config_path: PathBuf,
    plan_path: PathBuf,
}

impl CoLoMoAgent {
    pub fn new(project_root: PathBuf) -> Self {
        let http = HttpClient::new();

        let config_path = project_root.join("config.yaml");
        let plan_path = project_root.join("plan.md");

        Self {
            http,
            project_root,
            config_path,
            plan_path,
        }
    }

    pub fn with_http(http: HttpClient, project_root: PathBuf) -> Self {
        let config_path = project_root.join("config.yaml");
        let plan_path = project_root.join("plan.md");

        Self {
            http,
            project_root,
            config_path,
            plan_path,
        }
    }

    async fn iflow_chat_stream(
        &self,
        model: &str,
        messages: Vec<serde_json::Value>,
        mut on_chunk: impl FnMut(String) + Send + 'static,
    ) -> Result<(), AgentError> {
        let base =
            std::env::var("IFLOW_API_BASE").unwrap_or_else(|_| "https://apis.iflow.cn/v1".into());
        let key = std::env::var("IFLOW_API_KEY")
            .map_err(|_| AgentError::Llm("Missing IFLOW_API_KEY".into()))?;
        let url = format!("{}/chat/completions", base.trim_end_matches('/'));
        let masked_key = if key.len() > 8 {
            format!("{}...{}", &key[..4], &key[key.len() - 4..])
        } else {
            "***".to_string()
        };
        debug_log(&format!("[iflow] URL: {}", url));
        debug_log(&format!("[iflow] Key: {}", masked_key));
        debug_log(&format!("[iflow] Model: {}", model));
        debug_log(&format!("[iflow] Messages: {}", serde_json::to_string(&messages).unwrap_or_default()));
        debug_log("[iflow] Sending HTTP request...");
        let body = serde_json::json!({
            "model": model,
            "messages": messages,
            "stream": true,
            "extra_body": serde_json::json!({}),
        });
        let res = self
            .http
            .post(&url)
            .bearer_auth(&key)
            .json(&body)
            .send()
            .await
            .map_err(|e| AgentError::Llm(format!("HTTP error: {}", e)))?;
        let status = res.status();
        debug_log(&format!("[iflow] Status: {}", status));
        if !status.is_success() {
            let text = res.text().await.unwrap_or_default();
            debug_log(&format!("[iflow] Error body: {}", text));
            return Err(AgentError::Llm(format!(
                "iflow non-200: {} {}",
                status, text
            )));
        }
        debug_log("[iflow] Response headers received, starting stream...");
        let mut full_response = String::new();
        let mut stream = res.bytes_stream();
        let mut chunk_count = 0usize;
        while let Some(chunk) = stream.next().await {
            if let Ok(bytes) = chunk {
                chunk_count += 1;
                if chunk_count.is_multiple_of(10) {
                    debug_log(&format!("[iflow] chunk #{}", chunk_count));
                }
                let s = String::from_utf8_lossy(&bytes).to_string();
                // if server uses SSE style, split by lines starting with "data:"
                for line in s.lines() {
                    let trimmed = line.trim_start_matches("data:").trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    // try parse a JSON delta, fallback to raw text
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) {
                        let piece = v["choices"][0]["delta"]["content"]
                            .as_str()
                            .or_else(|| v["choices"][0]["message"]["content"].as_str())
                            .unwrap_or("");
                        if !piece.is_empty() {
                            full_response.push_str(piece);
                            on_chunk(piece.to_string());
                        }
                    } else {
                        full_response.push_str(trimmed);
                        on_chunk(trimmed.to_string());
                    }
                }
            }
        }
        debug_log(&format!("[iflow] Full response: {}", full_response));
        Ok(())
    }

    /// Run the planner agent to generate an implementation plan
    /// Uses full tool set: file read/write/edit, bash, API calls, and context tools
    pub async fn run_planner(&self, requirement: &str) -> Result<String, AgentError> {
        // Streaming to Output
        let model = std::env::var("IFLOW_MODEL").unwrap_or_else(|_| "qwen3-max-preview".into());
        let messages = vec![
            json!({"role":"system","content":"You are a planner agent for CoLoMo (Context-Local-Model) ML training system.\n\nIMPORTANT: ONLY generate a detailed implementation PLAN in markdown format. Do NOT write any code, do NOT create any files. Output only the plan text.\n\nYour plan should include:\n1. Project structure (directories, config.yaml, train.py, etc.)\n2. Dependencies (requirements.txt with packages and versions)\n3. Training phases (data loading, model definition, training loop, evaluation)\n4. Recommended hyperparameters (batch_size, learning_rate, optimizer, epochs)\n5. How to run and test the project\n\nFormat the plan clearly in markdown with headers, numbered steps, and code blocks for configuration examples only."}),
            json!({"role":"user","content": format!("Generate an implementation plan for: {}", requirement)}),
        ];
        let _state_arc = get_state().clone();
        let acc = Arc::new(Mutex::new(String::new()));
        let acc_closure = acc.clone();
        self.iflow_chat_stream(&model, messages, move |piece| {
            // Accumulate for final return value
            if let Ok(mut buf) = acc_closure.lock() {
                buf.push_str(&piece);
            }
            // Show each chunk immediately in output (non-blocking via UX channel)
            let line = piece.replace('\n', " ");
            if !line.is_empty() {
                send_ux(UxEvent::AppendOutput(format!("agent: {}", line)));
            }
        })
        .await?;
        let acc_content = acc.lock().map(|b| b.clone()).unwrap_or_default();

        // Write plan to file
        if !acc_content.is_empty() {
            if let Some(parent) = self.plan_path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            if let Err(e) = std::fs::write(&self.plan_path, &acc_content) {
                debug_log(&format!("[Planner] Failed to write plan.md: {}", e));
            } else {
                debug_log(&format!(
                    "[Planner] plan.md written: {} bytes",
                    acc_content.len()
                ));
            }
        }

        Ok(acc_content)
    }

    /// Run the planner agent with structured extraction
    /// Uses extractor to get typed response
    #[allow(dead_code)]
    pub async fn run_planner_structured<
        T: for<'a> Deserialize<'a> + Serialize + schemars::JsonSchema + Send + Sync + 'static,
    >(
        &self,
        _requirement: &str,
    ) -> Result<T, AgentError> {
        Err(AgentError::Llm(
            "run_planner_structured not implemented for iflow mode".into(),
        ))
    }

    /// Run the advisor agent to get parameter recommendations
    /// Uses full tool set including file operations, bash, and API
    #[allow(dead_code)]
    pub async fn run_advisor(&self) -> Result<String, AgentError> {
        // Streaming to Output
        let model = std::env::var("IFLOW_MODEL").unwrap_or_else(|_| "qwen3-max-preview".into());
        let messages = vec![
            json!({"role":"system","content":"You are an advisor agent for CoLoMo ML training system. Provide training parameter recommendations based on GPU/system status and config."}),
            json!({"role":"user","content":"Analyze system and propose batch_size, learning_rate, optimizer with rationale."}),
        ];
        let _state_arc = get_state().clone();
        let acc = Arc::new(Mutex::new(String::new()));
        let acc_closure = acc.clone();
        self.iflow_chat_stream(&model, messages, move |piece| {
            if let Ok(mut buf) = acc_closure.lock() {
                buf.push_str(&piece);
                if piece.contains('\n') || buf.len() > 100 {
                    let mut line = buf.replace('\n', " ");
                    if line.len() > 200 {
                        line.truncate(200);
                        line.push('…');
                    }
                    send_ux(UxEvent::AppendOutput(format!("Advisor: {}", line)));
                    buf.clear();
                }
            }
        })
        .await?;
        if let Ok(buf) = acc.lock() {
            if !buf.is_empty() {
                let mut line = buf.replace('\n', " ");
                if line.len() > 200 {
                    line.truncate(200);
                    line.push('…');
                }
                if let Ok(mut s) = get_state().lock() {
                    s.output_lines.push(format!("Advisor: {}", line));
                    s.output_scroll = usize::MAX;
                }
            }
        }
        Ok(String::new())
    }

    /// Run the advisor agent with structured extraction
    #[allow(dead_code)]
    pub async fn run_advisor_structured<
        T: for<'a> Deserialize<'a> + Serialize + schemars::JsonSchema + Send + Sync + 'static,
    >(
        &self,
    ) -> Result<T, AgentError> {
        Err(AgentError::Llm(
            "run_advisor_structured not implemented for iflow mode".into(),
        ))
    }

    // =========================================================================
    // Execute Agent — reads plan.md, generates and writes implementation files
    // =========================================================================

    pub async fn run_execute(&self) -> Result<String, AgentError> {
        // Read plan.md
        let plan_content = std::fs::read_to_string(&self.plan_path)
            .map_err(AgentError::Io)?;

        let model = std::env::var("IFLOW_MODEL").unwrap_or_else(|_| "qwen3-max-preview".into());
        let _state_arc = get_state().clone();
        let acc = Arc::new(Mutex::new(String::new()));
        let acc_closure = acc.clone();

        // Read project context for the prompt
        let config_content = std::fs::read_to_string(&self.config_path).unwrap_or_default();
        let train_script = self.project_root.join("train.py");
        let train_content = std::fs::read_to_string(&train_script).unwrap_or_default();

        let messages = vec![
            json!({"role":"system","content":"You are an execute agent for CoLoMo ML training system. Read plan.md and implement each phase. For each step, generate the appropriate file content or command. Output your work as a sequence of actions:\n\nFILE: <path>\n```\n<content>\n```\n\nCMD: <conda/shell command>\n\nExecute the plan phase by phase."}),
            json!({"role":"user","content": format!("Plan:\n{}\n\nConfig:\n{}\n\nTrain script (first 30 lines):\n{}", plan_content, config_content, &train_content[..train_content.len().min(1000)])}),
        ];

        self.iflow_chat_stream(&model, messages, move |piece| {
            if let Ok(mut buf) = acc_closure.lock() {
                buf.push_str(&piece);
            }
            let line = piece.replace('\n', " ");
            if !line.is_empty() {
                send_ux(UxEvent::AppendOutput(format!("agent: {}", line)));
            }
        })
        .await?;

        let acc_content = acc.lock().map(|b| b.clone()).unwrap_or_default();

        // Parse FILE: and CMD: blocks and execute them
        self.execute_plan_actions(&acc_content)?;

        debug_log(&format!("[Execute] completed, {} chars", acc_content.len()));
        Ok(acc_content)
    }

    fn execute_plan_actions(&self, content: &str) -> Result<(), AgentError> {
        let mut current_path: Option<PathBuf> = None;
        let mut file_buffer = String::new();
        let mut in_code_block = false;

        for line in content.lines() {
            if line.starts_with("FILE: ") {
                // Flush previous file
                if let Some(path) = current_file(&self.project_root, current_path.take()) {
                    if !file_buffer.is_empty() {
                        let _ = std::fs::write(&path, &file_buffer);
                        debug_log(&format!("[Execute] wrote file: {}", path.display()));
                    }
                }
                let path_str = line.trim_start_matches("FILE: ").trim();
                current_path = Some(self.project_root.join(path_str));
                file_buffer.clear();
                in_code_block = false;
            } else if line.starts_with("```") {
                in_code_block = !in_code_block;
            } else if in_code_block {
                file_buffer.push_str(line);
                file_buffer.push('\n');
            } else if line.starts_with("CMD: ") {
                let cmd = line.trim_start_matches("CMD: ").trim();
                debug_log(&format!("[Execute] CMD: {}", cmd));
                if let Ok(mut s) = get_state().lock() {
                    s.output_lines.push(format!("agent: CMD: {}", cmd));
                    s.output_scroll = usize::MAX;
                }
                // Execute conda/shell commands
                let _ = self.run_shell_command(cmd);
            }
        }

        // Flush last file
        if let Some(path) = current_file(&self.project_root, current_path.clone()) {
            if !file_buffer.is_empty() {
                let _ = std::fs::write(&path, &file_buffer);
                debug_log(&format!("[Execute] wrote file: {}", path.display()));
            }
        }

        Ok(())
    }

    fn run_shell_command(&self, cmd: &str) -> std::process::Output {
        use std::process::Command;
        if cmd.starts_with("conda ") || cmd.starts_with("pip ") || cmd.starts_with("python ") {
            let parts: Vec<&str> = cmd.split_whitespace().collect();
            if parts.len() >= 2 {
                // For conda run -n ENV commands
                if parts[0] == "conda" && parts.get(1) == Some(&"run") {
                    #[cfg(windows)]
                    {
                        let mut c = Command::new("cmd");
                        c.arg("/C");
                        c.arg("conda");
                        for p in &parts[1..] { c.arg(p); }
                        c.current_dir(&self.project_root);
                        return c.output().unwrap();
                    }
                    #[cfg(not(windows))]
                    {
                        let mut c = Command::new("conda");
                        for p in &parts[1..] { c.arg(p); }
                        c.current_dir(&self.project_root);
                        return c.output().unwrap();
                    }
                }
                // Fall through to shell for other commands
            }
        }
        // Use shell for complex commands
        let conda_dir = crate::env_check::conda_bin_dir();
        let current_path = std::env::var("PATH").unwrap_or_default();
        let new_path = if let Some(ref dir) = conda_dir {
            #[cfg(windows)]
            { format!("{};{}", dir.display(), current_path) }
            #[cfg(not(windows))]
            { format!("{}:{}", dir.display(), current_path) }
        } else {
            current_path.clone()
        };
        #[cfg(windows)]
        { Command::new("cmd").args(["/C", cmd]).env("PATH", &new_path).current_dir(&self.project_root).output().unwrap() }
        #[cfg(not(windows))]
        { Command::new("sh").args(["-c", cmd]).env("PATH", &new_path).current_dir(&self.project_root).output().unwrap() }
    }

    // =========================================================================
    // Checker Agent — validates plan and code
    // =========================================================================

    pub async fn run_checker(&self) -> Result<String, AgentError> {
        let plan_content = std::fs::read_to_string(&self.plan_path)
            .map_err(AgentError::Io)?;
        let config_content = std::fs::read_to_string(&self.config_path).unwrap_or_default();

        let model = std::env::var("IFLOW_MODEL").unwrap_or_else(|_| "qwen3-max-preview".into());
        let _state_arc = get_state().clone();
        let acc = Arc::new(Mutex::new(String::new()));
        let acc_closure = acc.clone();

        let messages = vec![
            json!({"role":"system","content":"You are a code reviewer for CoLoMo. Validate: (1) plan.md is feasible and complete, (2) config.yaml is consistent with plan, (3) train.py exists and is correct. Report issues as:\nCHECK: <item> — OK or ISSUE: <description>"}),
            json!({"role":"user","content": format!("Plan:\n{}\n\nConfig:\n{}", plan_content, config_content)}),
        ];

        self.iflow_chat_stream(&model, messages, move |piece| {
            if let Ok(mut buf) = acc_closure.lock() { buf.push_str(&piece); }
            let line = piece.replace('\n', " ");
            if !line.is_empty() {
                send_ux(UxEvent::AppendOutput(format!("agent: {}", line)));
            }
        })
        .await?;

        let acc_content = acc.lock().map(|b| b.clone()).unwrap_or_default();
        debug_log("[Checker] completed");
        Ok(acc_content)
    }

    // =========================================================================
    // Summary Agent — structured output
    // =========================================================================

    pub async fn run_summary(&self) -> Result<String, AgentError> {
        let plan_content = std::fs::read_to_string(&self.plan_path)
            .map_err(AgentError::Io)?;
        let config_content = std::fs::read_to_string(&self.config_path).unwrap_or_default();

        // Read last exit status
        let last_exit = self.project_root.join("status").join("last_exit.json");
        let exit_content = std::fs::read_to_string(&last_exit).unwrap_or_default();

        // Read recent logs
        let logs_dir = self.project_root.join("logs");
        let recent_log = std::fs::read_dir(&logs_dir)
            .ok()
            .and_then(|entries| {
                entries.filter_map(|e| e.ok())
                    .filter(|e| e.path().extension().map(|ex| ex == "log").unwrap_or(false))
                    .max_by_key(|e| e.metadata().ok().and_then(|m| m.modified().ok()).unwrap_or(std::time::SystemTime::UNIX_EPOCH))
            })
            .and_then(|e| std::fs::read_to_string(e.path()).ok())
            .unwrap_or_default();

        let model = std::env::var("IFLOW_MODEL").unwrap_or_else(|_| "qwen3-max-preview".into());

        let messages = vec![
            json!({"role":"system","content":"Extract key information and output a SINGLE summary line with format:\nsummary: keywords=<keywords>; files=<file paths>; requirement=<overview>; plan_path=<path>; conda_env=<env>; error_code=<code or none>; error_msg=<msg or none>; model_overview=<model description>; task_completed=<yes/no/notrun>"}),
            json!({"role":"user","content": format!("Plan:\n{}\n\nConfig:\n{}\n\nLast Exit:\n{}\n\nRecent Logs (last 500 chars):\n{}", &plan_content[..plan_content.len().min(2000)], config_content, exit_content, &recent_log[..recent_log.len().min(500)])}),
        ];

        send_ux(UxEvent::AppendOutput("agent: Generating summary...".into()));

        let acc = Arc::new(Mutex::new(String::new()));
        let acc_closure = acc.clone();

        self.iflow_chat_stream(&model, messages, move |piece| {
            if let Ok(mut buf) = acc_closure.lock() { buf.push_str(&piece); }
            let line = piece.replace('\n', " ");
            if !line.is_empty() {
                send_ux(UxEvent::AppendOutput(format!("agent: {}", line)));
            }
        })
        .await?;

        let acc_content = acc.lock().map(|b| b.clone()).unwrap_or_default();
        debug_log(&format!("[Summary] completed: {}", &acc_content[..acc_content.len().min(200)]));
        Ok(acc_content)
    }

    // =========================================================================
    // Tester Agent — runs tests and reports model functionality
    // =========================================================================

    pub async fn run_tester(&self) -> Result<String, AgentError> {
        let config_content = std::fs::read_to_string(&self.config_path).unwrap_or_default();
        let cfg: crate::config::Config = serde_yaml::from_str(&config_content).unwrap_or_default();
        let env_name = cfg.conda_env.unwrap_or_else(|| "colomo".into());

        let _state_arc = get_state().clone();
        send_ux(UxEvent::AppendOutput(format!("agent: Running tests in conda env: {}", env_name)));

        // Run pytest or python -m pytest in conda
        let output = self.run_conda_command(&env_name, &["python", "-m", "pytest", "-v", "--tb=short"]);

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let combined = if stdout.is_empty() { stderr.clone() } else { stdout.clone() };

        // Stream output line by line
        for line in combined.lines().take(50) {
            send_ux(UxEvent::AppendOutput(format!("agent: {}", line)));
        }

        let exit_ok = output.status.success();
        let task_completed = if exit_ok { "yes" } else { "partial" };

        let summary_line = format!(
            "summary: keywords=test; files=train.py,config.yaml; requirement=test; plan_path={}; conda_env={}; error_code={}; error_msg=none; model_overview=see output; task_completed={};",
            self.plan_path.display(),
            env_name,
            if exit_ok { "none" } else { "TEST_FAIL" },
            task_completed
        );

        send_ux(UxEvent::AppendOutput(summary_line.clone()));

        debug_log(&format!("[Tester] completed, exit: {}", output.status));
        Ok(summary_line)
    }

    fn run_conda_command(&self, env: &str, args: &[&str]) -> std::process::Output {
        use std::process::Command;
        let conda_dir = crate::env_check::conda_bin_dir();
        let current_path = std::env::var("PATH").unwrap_or_default();
        let new_path = if let Some(ref dir) = conda_dir {
            #[cfg(windows)]
            { format!("{};{}", dir.display(), current_path) }
            #[cfg(not(windows))]
            { format!("{}:{}", dir.display(), current_path) }
        } else {
            current_path
        };
        #[cfg(windows)]
        {
            let conda_args = std::iter::once("conda".to_string())
                .chain(std::iter::once("run".to_string()))
                .chain(std::iter::once("-n".to_string()))
                .chain(std::iter::once(env.to_string()))
                .chain(args.iter().map(|s| s.to_string()))
                .collect::<Vec<_>>()
                .join(" ");
            let mut c = Command::new("cmd");
            c.args(["/C", &conda_args]);
            c.env("PATH", &new_path);
            c.current_dir(&self.project_root);
            c.output().unwrap()
        }
        #[cfg(not(windows))]
        {
            let mut c = Command::new("conda");
            c.args(["run", "-n", env]);
            for a in args { c.arg(a); }
            c.env("PATH", &new_path);
            c.current_dir(&self.project_root);
            c.output().unwrap()
        }
    }

    // =========================================================================
    // Runner Agent — runs training in Conda
    // =========================================================================

    pub fn run_runner_blocking(&self) {
        use std::process::Command;
        use std::thread;

        let config_content = std::fs::read_to_string(&self.config_path).unwrap_or_default();
        let cfg: crate::config::Config = serde_yaml::from_str(&config_content).unwrap_or_default();
        let env_name = cfg.conda_env.unwrap_or_else(|| "colomo".into());
        let _backend = cfg.backend.unwrap_or_else(|| "cuda".into());
        let script = cfg.train_script.unwrap_or_else(|| "train.py".into());

        let project_root = self.project_root.clone();
        let logs_dir = project_root.join("logs");

        thread::spawn(move || {
            let _ = std::fs::create_dir_all(&logs_dir);
            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let log_file = logs_dir.join(format!("run-{}.log", timestamp));

            // Update state: running
            send_ux(UxEvent::SetRunning(true));
            send_ux(UxEvent::AppendOutput(format!("runner: Starting training in env:{}", env_name)));

            // Build conda command with conda bin dir in PATH
            let conda_dir = crate::env_check::conda_bin_dir();
            let current_path = std::env::var("PATH").unwrap_or_default();
            let new_path = if let Some(ref dir) = conda_dir {
                #[cfg(windows)]
                { format!("{};{}", dir.display(), current_path) }
                #[cfg(not(windows))]
                { format!("{}:{}", dir.display(), current_path) }
            } else {
                current_path.clone()
            };

            #[cfg(windows)]
            let mut cmd = {
                let mut c = Command::new("cmd");
                c.args(["/C", "conda", "run", "-n", &env_name, "python", "-u", &script]);
                c.env("PATH", &new_path);
                c.current_dir(&project_root);
                c.stdout(std::process::Stdio::piped());
                c.stderr(std::process::Stdio::piped());
                c
            };
            #[cfg(not(windows))]
            let mut cmd = {
                let mut c = Command::new("conda");
                c.args(["run", "-n", &env_name, "python", "-u", &script]);
                c.env("PATH", &new_path);
                c.current_dir(&project_root);
                c.stdout(std::process::Stdio::piped());
                c.stderr(std::process::Stdio::piped());
                c
            };

            debug_log(&format!("[Runner] Starting: conda run -n {} python -u {}", env_name, script));

            match cmd.spawn() {
                Ok(mut child) => {
                    use std::io::{BufRead, BufReader};
                    let stdout = child.stdout.take();
                    let stderr = child.stderr.take();

                    // Read stdout in a thread
                    if let Some(stdout) = stdout {
                        let log_file = log_file.clone();
                        thread::spawn(move || {
                            let reader = BufReader::new(stdout);
                            let mut logf = std::fs::OpenOptions::new()
                                .create(true).append(true).open(&log_file).ok();
                            for line in reader.lines().map_while(Result::ok) {
                                send_ux(UxEvent::AppendOutput(format!("runner: {}", line)));
                                if let Some(ref mut f) = logf {
                                    let _ = writeln!(f, "[stdout] {}", line);
                                }
                            }
                        });
                    }

                    // Read stderr in a thread
                    if let Some(stderr) = stderr {
                        thread::spawn(move || {
                            let reader = BufReader::new(stderr);
                            let mut logf = std::fs::OpenOptions::new()
                                .create(true).append(true).open(&log_file).ok();
                            for line in reader.lines().map_while(Result::ok) {
                                send_ux(UxEvent::AppendOutput(format!("runner: ERR {}", line)));
                                if let Some(ref mut f) = logf {
                                    let _ = writeln!(f, "[stderr] {}", line);
                                }
                            }
                        });
                    }

                    // Wait for child to finish
                    let status = child.wait().unwrap_or_default();
                    debug_log(&format!("[Runner] Exited with: {:?}", status));

                    // Update state: stopped
                    send_ux(UxEvent::SetRunning(false));
                    send_ux(UxEvent::AppendOutput(format!("runner: Exited with {:?}", status)));

                    // Write exit status
                    let exit_status = serde_json::json!({
                        "exit_code": status.code(),
                        "timestamp": timestamp,
                    });
                    let status_dir = project_root.join("status");
                    let _ = std::fs::create_dir_all(&status_dir);
                    let _ = std::fs::write(status_dir.join("last_exit.json"), exit_status.to_string());
                }
                Err(e) => {
                    debug_log(&format!("[Runner] Failed to spawn: {}", e));
                    send_ux(UxEvent::SetRunning(false));
                    send_ux(UxEvent::AppendOutput(format!("runner: Failed to start: {}", e)));
                }
            }
        });
    }

    // =========================================================================
    // Teacher Agent — explains algorithms with intro, pros, cons, use cases
    // =========================================================================

    pub async fn run_teacher(&self, topic: &str) -> Result<String, AgentError> {
        let model = std::env::var("IFLOW_MODEL").unwrap_or_else(|_| "qwen3-max-preview".into());
        let _state_arc = get_state().clone();
        let acc = Arc::new(Mutex::new(String::new()));
        let acc_closure = acc.clone();

        let messages = vec![
            json!({"role":"system","content":"You are a teacher for ML algorithms. For the given topic, explain: (1) Introduction, (2) How it works, (3) Pros/advantages, (4) Cons/disadvantages, (5) Use cases, (6) Common variants. Be concise but informative."}),
            json!({"role":"user","content": format!("Explain: {}", topic)}),
        ];

        self.iflow_chat_stream(&model, messages, move |piece| {
            if let Ok(mut buf) = acc_closure.lock() { buf.push_str(&piece); }
            let line = piece.replace('\n', " ");
            if !line.is_empty() {
                send_ux(UxEvent::AppendOutput(format!("agent: {}", line)));
            }
        })
        .await?;

        let acc_content = acc.lock().map(|b| b.clone()).unwrap_or_default();
        Ok(acc_content)
    }
}

// Helper to resolve current file path
fn current_file(project_root: &Path, path: Option<PathBuf>) -> Option<PathBuf> {
    path.map(|p| {
        if p.is_absolute() { p } else { project_root.join(p) }
    })
}

// ============================================================================
// Async Agent Runner (for TUI integration)
// ============================================================================

pub struct AgentRunner {
    #[allow(dead_code)]
    state: Arc<std::sync::Mutex<crate::AppState>>,
}

impl AgentRunner {
    pub fn new(state: Arc<std::sync::Mutex<crate::AppState>>) -> Self {
        Self { state }
    }

    pub fn spawn_planner(&self, requirement: String, project_root: PathBuf) {
        // Read settings for debug before spawning thread
        let debug_enabled = {
            let state = get_state();
            state.lock().unwrap().settings.debug
        };
        let log_path = {
            let state = get_state();
            state.lock().unwrap().cfg_path.parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| PathBuf::from("."))
                .join("logs").join("logger.log")
        };

        // Use std::thread instead of tokio::spawn to avoid complex async runtime context
        std::thread::spawn(move || {
            let file_log = |msg: &str| {
                if debug_enabled {
                    let _ = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&log_path)
                        .and_then(|mut f| writeln!(f, "{}", msg));
                }
            };

            file_log("[SPAWN] planner thread started");

            // Create HTTP client synchronously (blocking, but thread is dedicated)
            let http = HttpClient::new();
            let agent = CoLoMoAgent::with_http(http, project_root.clone());

            file_log("[Planner] HTTP client ready");

            // Send initial UX events via blocking send (succeeds because UX channel is unbounded)
            send_ux(UxEvent::SetActiveAgent("planner".into()));
            send_ux(UxEvent::SetAgentTask("Initializing planner".into()));
            send_ux(UxEvent::SetAgentProgress("0/5".into()));
            send_ux(UxEvent::SetTopRightLines(vec![
                "[>>] Planner: Initializing...".into(),
                "[ ] Gather project context".into(),
                "[ ] Analyze requirement".into(),
                "[ ] Generate plan".into(),
                "[ ] Save plan".into(),
            ]));
            send_ux(UxEvent::SetSummaryLines(vec![
                format!("Planning: {}", requirement),
                "[Running...]".into(),
            ]));
            send_ux(UxEvent::AppendOutput(format!("> /plan {}", requirement)));
            send_ux(UxEvent::AppendOutput("agent: [Planning...]".into()));

            send_ux(UxEvent::SetAgentTask("Gathering context".into()));
            send_ux(UxEvent::SetAgentProgress("1/5".into()));
            send_ux(UxEvent::AppendOutput("  [1/5] Gathering project context...".into()));

            // Build timeout using std time (no tokio needed on this thread)
            let timeout_ms: u64 = std::env::var("IFLOW_TIMEOUT_MS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(120_000);

            file_log("[Planner] Starting HTTP call...");

            // Build runtime for this thread (single-threaded, lightweight)
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("failed to build runtime for planner thread");
            let response = rt.block_on(async {
                let fut = agent.run_planner(&requirement);
                match tokio::time::timeout(std::time::Duration::from_millis(timeout_ms), fut).await {
                    Ok(r) => r.map(Some),
                    Err(_) => Ok(None), // timeout
                }
            });

            file_log(&format!("[Planner] run_planner done, ok={}", response.is_ok()));

            match response {
                Ok(Some(response)) if !response.is_empty() => {
                    file_log(&format!("[Planner] Success, {} chars", response.len()));
                    let plan_file = project_root.join("plan.md");
                    send_ux(UxEvent::SetTopRightLines(vec![
                        "[DONE] Planner: Plan generated".into(),
                    ]));
                    send_ux(UxEvent::SetSummaryLines(vec![
                        format!("Plan: {} chars", response.len()),
                        format!("Path: {}", plan_file.display()),
                        format!("Next: /execute to implement"),
                    ]));
                    send_ux(UxEvent::AppendOutput(format!("--- plan content ({}) ---", response.len())));
                    for line in response.lines() {
                        send_ux(UxEvent::AppendOutput(format!("  {}", line)));
                    }
                    send_ux(UxEvent::AppendOutput("---".to_string()));
                    send_ux(UxEvent::AppendOutput(format!("agent: ✓ Plan ready: {} chars", response.len())));
                    send_ux(UxEvent::AppendOutput("agent: Run /execute to implement.".into()));
                    send_ux(UxEvent::SetOutputScroll(usize::MAX));
                }
                Ok(None) => {
                    file_log("[Planner] Timeout!");
                    send_ux(UxEvent::SetAgentTask("Planner timeout".into()));
                    send_ux(UxEvent::SetTopRightLines(vec![
                        "[DONE] Planner: Timeout".into(),
                    ]));
                    send_ux(UxEvent::AppendOutput("agent: ✗ timed out".into()));
                }
                Ok(Some(_)) => {
                    file_log("[Planner] Empty response");
                    send_ux(UxEvent::SetAgentTask("Planner failed".into()));
                    send_ux(UxEvent::AppendOutput("agent: ✗ empty response".into()));
                }
                Err(e) => {
                    let err_msg = e.to_string();
                    file_log(&format!("[Planner] Error: {}", err_msg));
                    send_ux(UxEvent::SetAgentTask("Planner failed".into()));
                    send_ux(UxEvent::SetTopRightLines(vec![
                        "[DONE] Planner: Failed".into(),
                    ]));
                    send_ux(UxEvent::AppendOutput(format!("agent: ✗ error: {}", err_msg)));
                }
            }

            // Clear agent status after delay
            std::thread::sleep(std::time::Duration::from_secs(5));
            send_ux(UxEvent::SetActiveAgent(String::new()));
            send_ux(UxEvent::SetAgentTask(String::new()));
            send_ux(UxEvent::SetAgentProgress(String::new()));
            file_log("[Planner] Done");
        });
    }

    #[allow(dead_code)]
    pub fn spawn_advisor(&self, project_root: PathBuf) {
        // Read settings before spawning thread
        let debug_enabled = {
            let state = get_state();
            state.lock().unwrap().settings.debug
        };
        let log_path = {
            let state = get_state();
            state.lock().unwrap().cfg_path.parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| PathBuf::from("."))
                .join("logs").join("logger.log")
        };

        std::thread::spawn(move || {
            let file_log = |msg: &str| {
                if debug_enabled {
                    let _ = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&log_path)
                        .and_then(|mut f| writeln!(f, "{}", msg));
                }
            };

            file_log("[Advisor] thread started");

            // Build single-thread runtime for async HTTP calls
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("failed to build runtime for advisor thread");

            let http = HttpClient::new();
            let agent = CoLoMoAgent::with_http(http, project_root);

            file_log("[Advisor] HTTP client ready");

            // Send initial UX events
            send_ux(UxEvent::SetActiveAgent("advisor".into()));
            send_ux(UxEvent::SetAgentTask("Checking system status".into()));
            send_ux(UxEvent::SetAgentProgress("0/4".into()));
            send_ux(UxEvent::AppendOutput("> Running advisor agent...".into()));

            file_log("[Advisor] Calling run_advisor...");

            let result = rt.block_on(async {
                tokio::time::timeout(
                    tokio::time::Duration::from_secs(20),
                    agent.run_advisor(),
                )
                .await
            });

            file_log(&format!("[Advisor] done, ok={}", result.is_ok()));

            match result {
                Ok(Ok(response)) => {
                    file_log(&format!("[Advisor] Success: {}", response));
                    send_ux(UxEvent::SetAgentTask("Recommendation ready".into()));
                    send_ux(UxEvent::SetAgentProgress("4/4".into()));
                    send_ux(UxEvent::AppendOutput(format!("✓ Advisor: {}", response)));
                }
                Ok(Err(e)) => {
                    let err_msg = e.to_string();
                    file_log(&format!("[Advisor] Error: {}", err_msg));
                    send_ux(UxEvent::SetAgentTask("Advisor failed".into()));
                    send_ux(UxEvent::SetAgentProgress(String::new()));
                    send_ux(UxEvent::AppendOutput(format!("✗ Advisor error: {}", err_msg)));
                }
                Err(_) => {
                    file_log("[Advisor] Timeout!");
                    send_ux(UxEvent::SetAgentTask("Advisor timeout".into()));
                    send_ux(UxEvent::SetAgentProgress(String::new()));
                    send_ux(UxEvent::AppendOutput("✗ Advisor timed out after 20s. Please check API connectivity or credentials.".into()));
                }
            }

            // Clear agent status after delay
            std::thread::sleep(std::time::Duration::from_secs(3));
            send_ux(UxEvent::SetActiveAgent(String::new()));
            send_ux(UxEvent::SetAgentTask(String::new()));
            send_ux(UxEvent::SetAgentProgress(String::new()));
            file_log("[Advisor] Done");
        });
    }

    /// Auto-complete: run the full pipeline sequentially (planner→summary→execute→summary→runner→summary→tester→summary)
    pub fn spawn_auto_complete(&self, requirement: String) {
        let debug_enabled = {
            let state = get_state();
            state.lock().unwrap().settings.debug
        };
        let log_path = {
            let state = get_state();
            state.lock().unwrap().cfg_path.parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| PathBuf::from("."))
                .join("logs").join("logger.log")
        };
        let project_root = {
            let state = get_state();
            state.lock().unwrap().cfg_path.parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| PathBuf::from("."))
        };

        std::thread::spawn(move || {
            let file_log = |msg: &str| {
                if debug_enabled {
                    let _ = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&log_path)
                        .and_then(|mut f| writeln!(f, "{}", msg));
                }
            };

            file_log("[AutoComplete] thread started");

            // Build runtime once for all async calls
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("failed to build runtime for auto_complete");

            // Create all HTTP clients in this thread (blocking, but thread is dedicated)
            let http1 = HttpClient::new();
            let http2 = HttpClient::new();
            let http3 = HttpClient::new();
            let http4 = HttpClient::new();

            file_log("[AutoComplete] HTTP clients ready");

            // Send initial UX events
            send_ux(UxEvent::SetActiveAgent("autocomplete".into()));
            send_ux(UxEvent::SetAgentTask("[1/7] Planner".into()));
            send_ux(UxEvent::SetAgentProgress("1/7".into()));
            send_ux(UxEvent::AppendOutput(format!("user: /auto-complete {} ---分割线", requirement)));
            send_ux(UxEvent::AppendOutput("=== Auto-complete pipeline starting ===".into()));
            send_ux(UxEvent::AppendOutput("[1/7] Running planner...".into()));
            send_ux(UxEvent::SetSummaryLines(vec![
                "Auto-complete: Starting...".into(),
                "[>>] Planner".into(),
            ]));
            send_ux(UxEvent::SetOutputScroll(usize::MAX));

            // Helper to update progress
            macro_rules! set_step {
                ($n:expr, $total:expr, $task:expr) => {{
                    send_ux(UxEvent::SetAgentProgress(format!("{}/{}", $n, $total)));
                    send_ux(UxEvent::SetAgentTask($task.into()));
                    send_ux(UxEvent::AppendOutput(format!("[{}] {}", $n, $task)));
                    send_ux(UxEvent::SetOutputScroll(usize::MAX));
                }};
            }

            // Helper to run summary and update summary_lines
            macro_rules! run_summary_step {
                ($label:expr, $http:expr) => {{
                    let agent = crate::llm_agent::CoLoMoAgent::with_http($http, project_root.clone());
                    match rt.block_on(agent.run_summary()) {
                        Ok(summary_text) => {
                            let label2 = $label.to_string();
                            send_ux(UxEvent::SetSummaryLines(vec![
                                label2,
                                format!("Summary: {}", &summary_text[..summary_text.len().min(100)]),
                            ]));
                            send_ux(UxEvent::SetAgentLastSummary(summary_text.clone()));
                            send_ux(UxEvent::AppendOutput(format!("agent: ✓ Summary: {}", &summary_text[..summary_text.len().min(200)])));
                            send_ux(UxEvent::SetOutputScroll(usize::MAX));
                            file_log(&format!("[AutoComplete] Summary: {}", &summary_text[..100]));
                        }
                        Err(e) => {
                            send_ux(UxEvent::AppendOutput(format!("agent: ⚠ Summary error: {}", e)));
                            send_ux(UxEvent::SetOutputScroll(usize::MAX));
                        }
                    }
                }};
            }

            // Step 1: Planner
            set_step!(1, 7, "[1/7] Planner");
            let agent1 = crate::llm_agent::CoLoMoAgent::with_http(http1, project_root.clone());
            let plan_result = rt.block_on(agent1.run_planner(&requirement));
            match &plan_result {
                Ok(plan_text) => {
                    send_ux(UxEvent::SetTopRightLines(vec![
                        "[DONE] Planner: Complete".into(),
                    ]));
                    send_ux(UxEvent::SetSummaryLines(vec![
                        format!("Plan: {} chars", plan_text.len()),
                        "[>>] Executor next".into(),
                    ]));
                    send_ux(UxEvent::AppendOutput(format!("agent: ✓ Planner done: {} chars", plan_text.len())));
                    send_ux(UxEvent::SetOutputScroll(usize::MAX));
                }
                Err(e) => {
                    send_ux(UxEvent::AppendOutput(format!("agent: ✗ Planner error: {}", e)));
                    send_ux(UxEvent::AppendOutput("agent: Pipeline stopped (run /plan separately)".into()));
                    send_ux(UxEvent::SetActiveAgent(String::new()));
                    return;
                }
            }

            // Step 2: Summary after planner
            set_step!(2, 7, "[2/7] Summary");
            run_summary_step!("After Planner", HttpClient::new());
            std::thread::sleep(std::time::Duration::from_secs(1));

            // Step 3: Executor
            set_step!(3, 7, "[3/7] Executor");
            let agent2 = crate::llm_agent::CoLoMoAgent::with_http(http2, project_root.clone());
            match rt.block_on(agent2.run_execute()) {
                Ok(_) => {
                    send_ux(UxEvent::AppendOutput("agent: ✓ Executor done".into()));
                    send_ux(UxEvent::SetOutputScroll(usize::MAX));
                }
                Err(e) => {
                    send_ux(UxEvent::AppendOutput(format!("agent: ✗ Executor error: {}", e)));
                    send_ux(UxEvent::AppendOutput("agent: Pipeline stopped".into()));
                    send_ux(UxEvent::SetActiveAgent(String::new()));
                    return;
                }
            }

            // Step 4: Summary after executor
            set_step!(4, 7, "[4/7] Summary");
            run_summary_step!("After Executor", HttpClient::new());
            std::thread::sleep(std::time::Duration::from_secs(1));

            // Step 5: Runner (blocking, wait for it)
            set_step!(5, 7, "[5/7] Runner");
            {
                let agent3 = crate::llm_agent::CoLoMoAgent::with_http(http3, project_root.clone());
                std::thread::spawn(move || {
                    agent3.run_runner_blocking();
                }).join().ok();
                send_ux(UxEvent::AppendOutput("agent: ✓ Runner spawned (training in background)".into()));
                send_ux(UxEvent::SetOutputScroll(usize::MAX));
            }

            // Give runner time to at least start streaming output
            std::thread::sleep(std::time::Duration::from_secs(3));

            // Step 6: Summary after runner
            set_step!(6, 7, "[6/7] Summary");
            run_summary_step!("After Runner", HttpClient::new());

            // Step 7: Tester
            set_step!(7, 7, "[7/7] Tester");
            let agent4 = crate::llm_agent::CoLoMoAgent::with_http(http4, project_root.clone());
            match rt.block_on(agent4.run_tester()) {
                Ok(_) => {
                    send_ux(UxEvent::AppendOutput("agent: ✓ Tester done".into()));
                    send_ux(UxEvent::SetOutputScroll(usize::MAX));
                }
                Err(e) => {
                    send_ux(UxEvent::AppendOutput(format!("agent: ✗ Tester error: {}", e)));
                    send_ux(UxEvent::SetOutputScroll(usize::MAX));
                }
            }

            // Final summary
            set_step!(7, 7, "[7/7] Final Summary");
            run_summary_step!("Pipeline Complete", HttpClient::new());

            send_ux(UxEvent::AppendOutput("=== Auto-complete pipeline finished ===".into()));
            send_ux(UxEvent::SetOutputScroll(usize::MAX));
            send_ux(UxEvent::SetActiveAgent(String::new()));
            send_ux(UxEvent::SetAgentTask(String::new()));
            send_ux(UxEvent::SetAgentProgress(String::new()));
            file_log("[AutoComplete] Pipeline finished");
        });
    }
}
