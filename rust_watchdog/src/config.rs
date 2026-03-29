use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Config {
    pub conda_env: Option<String>,
    pub backend: Option<String>,            // "cuda" | "tilelang"
    pub model_template: Option<String>,
    pub train_script: Option<String>,       // python script path
    pub tile_script: Option<String>,        // tile script path
    pub batch_size: Option<u64>,
    pub learning_rate: Option<f64>,
    pub optimizer: Option<String>,
    pub dataset_path: Option<String>,
    pub param_count: Option<u64>,
}

pub fn load_config(path: &Path) -> Result<Config> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("read config: {}", path.display()))?;
    let cfg: Config = serde_yaml::from_str(&text)
        .with_context(|| format!("parse yaml: {}", path.display()))?;
    Ok(cfg)
}

pub fn save_config(path: &Path, cfg: &Config) -> Result<()> {
    let s = serde_yaml::to_string(cfg)?;
    fs::write(path, s).with_context(|| format!("write config: {}", path.display()))?;
    Ok(())
}
