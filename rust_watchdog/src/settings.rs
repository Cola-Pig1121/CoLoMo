use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct ScoreWeights {
    pub acc: f64,
    pub lat: f64,
    pub mem: f64,
    pub thr: f64,
    pub energy: f64,
}

impl Default for ScoreWeights {
    fn default() -> Self {
        Self { acc: 0.6, lat: 0.2, mem: 0.2, thr: 0.0, energy: 0.0 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct Settings {
    pub safety_alpha: f64,
    pub learning_mode: bool,
    pub debug: bool,
    pub weights: ScoreWeights,
}

impl Default for Settings {
    fn default() -> Self {
        Self { safety_alpha: 0.85, learning_mode: false, debug: false, weights: ScoreWeights::default() }
    }
}

#[allow(dead_code)]
pub fn load(path: &Path) -> Result<Settings> {
    match fs::read_to_string(path) {
        Ok(text) => Ok(serde_yaml::from_str(&text).with_context(|| format!("parse settings: {}", path.display()))?),
        Err(_) => Ok(Settings::default()),
    }
}

#[allow(dead_code)]
pub fn save(path: &Path, s: &Settings) -> Result<()> {
    let text = serde_yaml::to_string(s)?;
    fs::write(path, text).with_context(|| format!("write settings: {}", path.display()))
}
