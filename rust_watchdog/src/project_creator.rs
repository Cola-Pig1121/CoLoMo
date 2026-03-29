//! Project Creator Module
//! Handles creation of new CoLoMo training projects.

use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};

/// Project paths created by the generator
#[derive(Debug)]
pub struct ProjectPaths {
    pub root: PathBuf,
    pub config: PathBuf,
    pub train_script: PathBuf,
    pub requirements: PathBuf,
    pub logs: PathBuf,
    pub plan: PathBuf,
}

/// Create a new project with the given name in the specified base path.
/// Creates directory structure, config.yaml, train.py, and planner output.
pub fn create_project(name: &str, base_path: &Path) -> Result<ProjectPaths> {
    // Validate project name (alphanumeric, hyphens, underscores only)
    if !name
        .chars()
        .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
    {
        anyhow::bail!(
            "Invalid project name '{}'. Use only letters, numbers, hyphens, and underscores.",
            name
        );
    }

    if name.is_empty() {
        anyhow::bail!("Project name cannot be empty.");
    }

    // Build project root path
    let project_root = base_path.join(name);

    // Check if directory already exists
    if project_root.exists() {
        anyhow::bail!(
            "Project directory '{}' already exists.",
            project_root.display()
        );
    }

    // Create directory structure
    fs::create_dir_all(project_root.join("logs"))
        .with_context(|| "Failed to create logs directory".to_string())?;
    fs::create_dir_all(project_root.join("saved"))
        .with_context(|| "Failed to create saved directory".to_string())?;

    // Generate config.yaml
    let config_path = project_root.join("config.yaml");
    generate_config_yaml(&config_path)?;

    // Generate train.py
    let train_path = project_root.join("train.py");
    generate_train_py(&train_path)?;

    // Generate requirements.txt
    let requirements_path = project_root.join("requirements.txt");
    fs::write(&requirements_path, "# Add your dependencies here\n")?;

    // Save planner output
    let plan_path = save_planner_output(name, &project_root)?;

    Ok(ProjectPaths {
        root: project_root.clone(),
        config: config_path,
        train_script: train_path,
        requirements: requirements_path,
        logs: project_root.join("logs"),
        plan: plan_path,
    })
}

/// Generate default config.yaml for the project
fn generate_config_yaml(path: &PathBuf) -> Result<()> {
    let content = r#"# CoLoMo Project Configuration
# Edit these values to configure your training run

conda_env: null           # Conda environment name (e.g., colomo)
backend: cuda            # Training backend: cuda | tilelang
model_template: null     # Model template (optional)
train_script: null       # Path to training script (e.g., train.py)
tile_script: null        # TileLang script (only for tilelang backend)
batch_size: null         # Batch size (e.g., 32, 64, 128)
learning_rate: null       # Learning rate (e.g., 0.001)
optimizer: null           # Optimizer: SGD | Adam | AdamW
dataset_path: null        # Path to dataset (e.g., ./data)
param_count: null         # Model parameter count (for advisor recommendations)
"#;
    fs::write(path, content).with_context(|| format!("Failed to write: {}", path.display()))?;
    Ok(())
}

/// Generate stub train.py for the project
fn generate_train_py(path: &PathBuf) -> Result<()> {
    let content = r#"#!/usr/bin/env python3
"""
CoLoMo Training Script
Edit this file to implement your training logic.
"""

import yaml
import os

# Load config.yaml
CONFIG_PATHS = [
    "config.yaml",
    os.path.join(os.path.dirname(__file__), "config.yaml"),
]
cfg = {}
for p in CONFIG_PATHS:
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        break

# Training parameters from config
batch_size = cfg.get("batch_size", 32)
learning_rate = cfg.get("learning_rate", 0.001)
optimizer = cfg.get("optimizer", "Adam")
backend = cfg.get("backend", "cuda")

print(f"Starting training with batch_size={batch_size}, lr={learning_rate}, optimizer={optimizer}")

# TODO: Implement your training logic here
for step in range(1, 101):
    print(f"step={step} loss={1000/(step+10):.4f}")

print("Training complete!")
"#;
    fs::write(path, content).with_context(|| format!("Failed to write: {}", path.display()))?;
    Ok(())
}

/// Save planner output template to projects/plan/<name>.md
fn save_planner_output(name: &str, project_path: &Path) -> Result<PathBuf> {
    // Create plan directory
    let plan_dir = project_path.parent().unwrap().join("plan");
    if !plan_dir.exists() {
        fs::create_dir_all(&plan_dir)?;
    }

    let plan_path = plan_dir.join(format!("{}.md", name));

    let content = format!(
        r#"# Project Plan: {name}

## Overview
Project created via CoLoMo `/new` command.

## Directory Structure
```
{name}/
├── config.yaml        # Training configuration
├── train.py          # Training script
├── requirements.txt  # Python dependencies
├── logs/            # Training logs
└── saved/          # Model checkpoints
```

## Configuration

### config.yaml
- backend: cuda
- conda_env: (set your conda environment)
- train_script: train.py
- batch_size: (set based on GPU memory)
- learning_rate: (set based on model)
- optimizer: (set based on parameter count)

## Training Plan

### Phase 1: Setup
- [ ] Configure conda environment
- [ ] Install dependencies
- [ ] Verify dataset path

### Phase 2: Initial Training
- [ ] Run initial training with small batch
- [ ] Monitor GPU memory usage
- [ ] Apply CoLoMo recommendations

### Phase 3: Optimization
- [ ] Adjust batch size based on advisor
- [ ] Tune learning rate
- [ ] Monitor for OOM events

## Notes

*Plan generated: {timestamp}*
"#,
        name = name,
        timestamp = chrono::Utc::now().format("%Y-%m-%d %H:%M UTC")
    );

    fs::write(&plan_path, content)
        .with_context(|| format!("Failed to write plan: {}", plan_path.display()))?;

    Ok(plan_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_create_project_creates_directories() {
        let temp_dir = env::temp_dir().join("colomo_test_project");
        let _ = fs::remove_dir_all(&temp_dir); // Clean up if exists

        let result = create_project("test_proj", &temp_dir);
        assert!(result.is_ok());

        let paths = result.unwrap();
        assert!(paths.root.exists());
        assert!(paths.logs.exists());
        assert!(paths.config.exists());
        assert!(paths.train_script.exists());

        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_invalid_project_name() {
        let temp_dir = env::temp_dir();
        let result = create_project("invalid/name", &temp_dir);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid project name"));
    }

    #[test]
    fn test_empty_project_name() {
        let temp_dir = env::temp_dir();
        let result = create_project("", &temp_dir);
        assert!(result.is_err());
    }

    #[test]
    fn test_generate_config_yaml() {
        let temp_file = env::temp_dir().join("test_config.yaml");
        let result = generate_config_yaml(&temp_file);
        assert!(result.is_ok());
        assert!(temp_file.exists());

        let content = fs::read_to_string(&temp_file).unwrap();
        assert!(content.contains("conda_env:"));
        assert!(content.contains("backend:"));
        assert!(content.contains("batch_size:"));

        let _ = fs::remove_file(&temp_file);
    }

    #[test]
    fn test_generate_train_py() {
        let temp_file = env::temp_dir().join("test_train.py");
        let result = generate_train_py(&temp_file);
        assert!(result.is_ok());
        assert!(temp_file.exists());

        let content = fs::read_to_string(&temp_file).unwrap();
        assert!(content.contains("CoLoMo Training Script"));
        assert!(content.contains("batch_size"));
        assert!(content.contains("learning_rate"));

        let _ = fs::remove_file(&temp_file);
    }
}
