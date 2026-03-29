//! API Call Tool
//! Tool for making HTTP API requests.

#[allow(dead_code)]
use rig::completion::ToolDefinition;
#[allow(dead_code)]
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;

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
// API Call Tool
// ============================================================================

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ApiCallTool;

impl ApiCallTool {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ApiCallTool {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct ApiCallArgs {
    pub url: String,
    pub method: Option<String>,
    pub headers: Option<HashMap<String, String>>,
    pub body: Option<String>,
    pub timeout_secs: Option<u64>,
}

#[derive(Debug, Serialize)]
#[allow(dead_code)]
pub struct ApiResponse {
    pub status: u16,
    pub headers: HashMap<String, String>,
    pub body: String,
    pub error: Option<String>,
}

impl Tool for ApiCallTool {
    const NAME: &'static str = "api_call";
    type Error = ToolError;
    type Args = ApiCallArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Make an HTTP API request. Supports GET, POST, PUT, DELETE, PATCH methods. Returns JSON with status, headers, and body.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to request"
                    },
                    "method": {
                        "type": "string",
                        "description": "HTTP method: GET, POST, PUT, DELETE, PATCH",
                        "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"]
                    },
                    "headers": {
                        "type": "object",
                        "description": "Optional: HTTP headers as key-value pairs"
                    },
                    "body": {
                        "type": "string",
                        "description": "Optional: Request body (JSON string)"
                    },
                    "timeout_secs": {
                        "type": "integer",
                        "description": "Optional: Request timeout in seconds"
                    }
                },
                "required": ["url"],
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let client = reqwest::Client::new();

        let method = args.method.as_deref().unwrap_or("GET");
        let timeout = args.timeout_secs.map(std::time::Duration::from_secs);

        let mut request_builder = match method.to_uppercase().as_str() {
            "GET" => client.get(&args.url),
            "POST" => client.post(&args.url),
            "PUT" => client.put(&args.url),
            "DELETE" => client.delete(&args.url),
            "PATCH" => client.patch(&args.url),
            _ => {
                return Err(ToolError::Call(format!("Unsupported HTTP method: {}", method)));
            }
        };

        // Add headers
        if let Some(headers) = args.headers {
            for (key, value) in headers {
                request_builder = request_builder.header(&key, &value);
            }
        }

        // Add body
        if let Some(body) = args.body {
            request_builder = request_builder.body(body);
        }

        // Set timeout
        if let Some(timeout) = timeout {
            request_builder = request_builder.timeout(timeout);
        }

        // Execute request
        let response = request_builder.send().await.map_err(|e| {
            ToolError::Call(format!("API request failed: {}", e))
        })?;

        let status = response.status().as_u16();
        let mut resp_headers: HashMap<String, String> = HashMap::new();
        for (key, value) in response.headers() {
            if let Ok(v) = value.to_str() {
                resp_headers.insert(key.to_string(), v.to_string());
            }
        }

        let body = response.text().await.map_err(|e| {
            ToolError::Call(format!("Failed to read response body: {}", e))
        })?;

        let result = ApiResponse {
            status,
            headers: resp_headers,
            body,
            error: None,
        };

        Ok(serde_json::to_string(&result)?)
    }
}
