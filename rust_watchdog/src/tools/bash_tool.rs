//! Bash Tool
//! Tool for executing bash commands with timeout support.

#[allow(dead_code)]
use rig::completion::ToolDefinition;
#[allow(dead_code)]
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use tokio::process::Command;

// ============================================================================
// Tool Error
// ============================================================================

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
// Bash Tool
// ============================================================================

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct BashTool {
    pub working_dir: std::path::PathBuf,
}

impl BashTool {
    #[allow(dead_code)]
    pub fn new(working_dir: std::path::PathBuf) -> Self {
        Self { working_dir }
    }
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct BashArgs {
    pub command: String,
    pub timeout_secs: Option<u64>,
    pub environment: Option<HashMap<String, String>>,
}

#[derive(Debug, Serialize)]
#[allow(dead_code)]
pub struct BashOutput {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
    pub timed_out: bool,
}

impl Tool for BashTool {
    const NAME: &'static str = "bash";
    type Error = ToolError;
    type Args = BashArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Execute a bash command and return its stdout, stderr, and exit code.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute"
                    },
                    "timeout_secs": {
                        "type": "integer",
                        "description": "Optional: Maximum seconds to wait before killing the process"
                    },
                    "environment": {
                        "type": "object",
                        "description": "Optional: Additional environment variables to set"
                    }
                },
                "required": ["command"],
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        // On Windows, use cmd.exe instead of bash
        let mut cmd = if cfg!(target_os = "windows") {
            let mut c = Command::new("cmd");
            c.arg("/C").arg(&args.command);
            c
        } else {
            let mut c = Command::new("bash");
            c.arg("-c").arg(&args.command);
            c
        };

        // Set environment variables if provided
        if let Some(env) = args.environment {
            for (key, value) in env {
                cmd.env(&key, &value);
            }
        }

        let timeout = args.timeout_secs.map(tokio::time::Duration::from_secs);

        let output = if let Some(dur) = timeout {
            match tokio::time::timeout(dur, cmd.output()).await {
                Ok(Ok(output)) => output,
                Ok(Err(e)) => {
                    return Err(ToolError::Io(std::io::Error::other(
                        format!("failed to execute: {}", e),
                    )));
                }
                Err(_) => {
                    // Timeout - kill the process
                    let _ = Command::new("pkill")
                        .arg("-f")
                        .arg(&args.command)
                        .output()
                        .await;

                    return Ok(serde_json::to_string(&BashOutput {
                        stdout: String::new(),
                        stderr: format!("Command timed out after {} seconds", dur.as_secs()),
                        exit_code: 124,
                        timed_out: true,
                    })?);
                }
            }
        } else {
            cmd.output().await.map_err(|e| {
                ToolError::Io(std::io::Error::other(
                    format!("failed to execute: {}", e),
                ))
            })?
        };

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let exit_code = output.status.code().unwrap_or(-1);

        let result = BashOutput {
            stdout,
            stderr,
            exit_code,
            timed_out: false,
        };

        Ok(serde_json::to_string(&result)?)
    }
}
