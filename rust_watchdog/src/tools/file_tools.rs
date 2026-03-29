//! File Tools
//! Tools for reading, editing, writing files and listing directories.

#[allow(dead_code)]
use rig::completion::ToolDefinition;
#[allow(dead_code)]
use rig::tool::Tool;
use serde::Deserialize;
use serde_json::json;
use std::io::Write;
use std::path::PathBuf;

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
// File Read Tool
// ============================================================================

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct FileReadTool {
    pub working_dir: PathBuf,
}

impl FileReadTool {
    #[allow(dead_code)]
    pub fn new(working_dir: PathBuf) -> Self {
        Self { working_dir }
    }
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct FileReadArgs {
    pub path: String,
    pub start_line: Option<u32>,
    pub max_lines: Option<u32>,
}

impl Tool for FileReadTool {
    const NAME: &'static str = "read_file";
    type Error = ToolError;
    type Args = FileReadArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Read the contents of a file. Supports partial reading with start_line and max_lines.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative or absolute path to the file"
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Optional: Line number to start reading from (1-based)"
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": "Optional: Maximum number of lines to read"
                    }
                },
                "required": ["path"],
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let path = PathBuf::from(&args.path);
        let full_path = if path.is_absolute() {
            path
        } else {
            self.working_dir.join(path)
        };

        let content = std::fs::read_to_string(&full_path)?;
        let lines: Vec<&str> = content.lines().collect();

        let start = args.start_line.unwrap_or(1).saturating_sub(1) as usize;
        let max_lines = args.max_lines.unwrap_or(lines.len() as u32) as usize;

        let selected_lines: String = lines
            .get(start..start.saturating_add(max_lines).min(lines.len()))
            .unwrap_or(&[])
            .iter()
            .enumerate()
            .map(|(i, l)| format!("{:3}: {}", start + i + 1, l))
            .collect::<Vec<_>>()
            .join("\n");

        Ok(selected_lines)
    }
}

// ============================================================================
// File Write Tool
// ============================================================================

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct FileWriteTool {
    pub working_dir: PathBuf,
}

impl FileWriteTool {
    #[allow(dead_code)]
    pub fn new(working_dir: PathBuf) -> Self {
        Self { working_dir }
    }
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct FileWriteArgs {
    pub path: String,
    pub content: String,
    pub append: Option<bool>,
}

impl Tool for FileWriteTool {
    const NAME: &'static str = "write_file";
    type Error = ToolError;
    type Args = FileWriteArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Write content to a file. Can create new files or overwrite existing ones. Use append=true to add to existing files.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative or absolute path to the file"
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file"
                    },
                    "append": {
                        "type": "boolean",
                        "description": "Optional: If true, append to existing file instead of overwriting"
                    }
                },
                "required": ["path", "content"],
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let path = PathBuf::from(&args.path);
        let full_path = if path.is_absolute() {
            path
        } else {
            self.working_dir.join(path)
        };

        // Ensure parent directory exists
        if let Some(parent) = full_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        if args.append.unwrap_or(false) {
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&full_path)?
                .write_all(args.content.as_bytes())?;
        } else {
            std::fs::write(&full_path, &args.content)?;
        }

        Ok(format!(
            "Written {} bytes to {}",
            args.content.len(),
            full_path.display()
        ))
    }
}

// ============================================================================
// File Edit Tool
// ============================================================================

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct FileEditTool {
    pub working_dir: PathBuf,
}

impl FileEditTool {
    #[allow(dead_code)]
    pub fn new(working_dir: PathBuf) -> Self {
        Self { working_dir }
    }
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct FileEditArgs {
    pub path: String,
    pub find: String,
    pub replace: String,
    pub replace_all: Option<bool>,
}

impl Tool for FileEditTool {
    const NAME: &'static str = "edit_file";
    type Error = ToolError;
    type Args = FileEditArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Edit a file by finding and replacing text. Supports single replacement or replace_all for multiple occurrences.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative or absolute path to the file"
                    },
                    "find": {
                        "type": "string",
                        "description": "The text to find"
                    },
                    "replace": {
                        "type": "string",
                        "description": "The replacement text"
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Optional: If true, replace all occurrences. If false, replace only the first."
                    }
                },
                "required": ["path", "find", "replace"],
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let path = PathBuf::from(&args.path);
        let full_path = if path.is_absolute() {
            path
        } else {
            self.working_dir.join(path)
        };

        let content = std::fs::read_to_string(&full_path)?;
        let count;

        if args.replace_all.unwrap_or(false) {
            count = content.matches(&args.find).count();
            let new_content = content.replace(&args.find, &args.replace);
            std::fs::write(&full_path, new_content)?;
        } else {
            count = if content.contains(&args.find) { 1 } else { 0 };
            if count > 0 {
                let new_content = content.replacen(&args.find, &args.replace, 1);
                std::fs::write(&full_path, new_content)?;
            }
        }

        Ok(format!(
            "Replaced {} occurrence(s) in {}",
            count,
            full_path.display()
        ))
    }
}

// ============================================================================
// Directory List Tool
// ============================================================================

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct DirectoryListTool {
    pub working_dir: PathBuf,
}

impl DirectoryListTool {
    #[allow(dead_code)]
    pub fn new(working_dir: PathBuf) -> Self {
        Self { working_dir }
    }
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct DirectoryListArgs {
    pub path: Option<String>,
    pub recursive: Option<bool>,
    pub include_hidden: Option<bool>,
}

impl Tool for DirectoryListTool {
    const NAME: &'static str = "list_directory";
    type Error = ToolError;
    type Args = DirectoryListArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "List contents of a directory. Can recurse into subdirectories and optionally include hidden files.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Optional: Relative or absolute path to directory. Defaults to working directory."
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Optional: If true, list subdirectories recursively"
                    },
                    "include_hidden": {
                        "type": "boolean",
                        "description": "Optional: If true, include hidden files (starting with .)"
                    }
                },
                "required": [],
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let target_path = if let Some(ref p) = args.path {
            let path = PathBuf::from(p);
            if path.is_absolute() {
                path
            } else {
                self.working_dir.join(path)
            }
        } else {
            self.working_dir.clone()
        };

        let mut entries: Vec<String> = Vec::new();

        fn walk_dir(
            dir: &PathBuf,
            entries: &mut Vec<String>,
            recursive: bool,
            include_hidden: bool,
            prefix: &str,
        ) {
            if let Ok(list) = std::fs::read_dir(dir) {
                let mut items: Vec<_> = list.filter_map(|e| e.ok()).collect();
                items.sort_by_key(|e| e.file_name());

                for (i, entry) in items.iter().enumerate() {
                    let name = entry.file_name().into_string().unwrap_or_default();

                    if !include_hidden && name.starts_with('.') {
                        continue;
                    }

                    let is_last = i == items.len() - 1;
                    let connector = if is_last { "└── " } else { "├── " };
                    let full_path = entry.path();

                    if entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
                        entries.push(format!("{}{}{}/", prefix, connector, name));
                        if recursive {
                            let new_prefix = format!("{}{}    ", prefix, connector);
                            walk_dir(&full_path, entries, recursive, include_hidden, &new_prefix);
                        }
                    } else {
                        let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                        entries.push(format!("{}{}{} ({} bytes)", prefix, connector, name, size));
                    }
                }
            }
        }

        entries.push(format!("{}/", target_path.display()));
        walk_dir(
            &target_path,
            &mut entries,
            args.recursive.unwrap_or(false),
            args.include_hidden.unwrap_or(false),
            "",
        );

        Ok(entries.join("\n"))
    }
}
