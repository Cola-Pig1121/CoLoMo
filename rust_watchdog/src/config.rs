use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
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

// ------------------ TDD: add validator skeleton ------------------
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum ConfigValidationError {
    #[error("backend must be one of: cuda, tilelang")]
    InvalidBackend,
    #[error("train_script must be set when backend=cuda")]
    MissingTrainScript,
    #[error("tile_script must be set when backend=tilelang")]
    MissingTileScript,
    #[error("batch_size must be > 0 if set")]
    InvalidBatchSize,
    #[error("learning_rate must be > 0 if set")]
    InvalidLearningRate,
}

pub fn validate_config(cfg: &Config, project_root: &Path) -> std::result::Result<(), Vec<ConfigValidationError>> {
    let mut errs = Vec::new();

    // Backend
    if let Some(b) = &cfg.backend {
        if b != "cuda" && b != "tilelang" {
            errs.push(ConfigValidationError::InvalidBackend);
        }
    } else {
        // no backend provided is okay (defaults elsewhere), skip path checks
        return if errs.is_empty() { Ok(()) } else { Err(errs) };
    }

    match cfg.backend.as_deref() {
        Some("cuda") => {
            if cfg.train_script.as_deref().filter(|s| !s.trim().is_empty()).is_none() {
                errs.push(ConfigValidationError::MissingTrainScript);
            } else if let Some(rel) = &cfg.train_script {
                // train_script must exist relative to project_root
                let p = project_root.join(rel);
                if !p.exists() { errs.push(ConfigValidationError::MissingTrainScript); }
            }
        }
        Some("tilelang") => {
            if cfg.tile_script.as_deref().filter(|s| !s.trim().is_empty()).is_none() {
                errs.push(ConfigValidationError::MissingTileScript);
            } else if let Some(rel) = &cfg.tile_script {
                let p = project_root.join(rel);
                if !p.exists() { errs.push(ConfigValidationError::MissingTileScript); }
            }
        }
        _ => {}
    }

    if let Some(bs) = cfg.batch_size { if bs == 0 { errs.push(ConfigValidationError::InvalidBatchSize); } }
    if let Some(lr) = cfg.learning_rate { if lr <= 0.0 { errs.push(ConfigValidationError::InvalidLearningRate); } }

    if errs.is_empty() { Ok(()) } else { Err(errs) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn root() -> PathBuf { std::env::temp_dir() }

    #[test]
    fn invalid_backend_is_rejected() {
        let cfg = Config { backend: Some("bogus".into()), ..Default::default() };
        let err = validate_config(&cfg, &root()).unwrap_err();
        assert!(err.contains(&ConfigValidationError::InvalidBackend));
    }

    #[test]
    fn cuda_requires_train_script() {
        let cfg = Config { backend: Some("cuda".into()), ..Default::default() };
        let err = validate_config(&cfg, &root()).unwrap_err();
        assert!(err.contains(&ConfigValidationError::MissingTrainScript));
    }

    #[test]
    fn tilelang_requires_tile_script() {
        let cfg = Config { backend: Some("tilelang".into()), ..Default::default() };
        let err = validate_config(&cfg, &root()).unwrap_err();
        assert!(err.contains(&ConfigValidationError::MissingTileScript));
    }

    #[test]
    fn batch_size_and_lr_must_be_positive() {
        let cfg = Config { backend: Some("cuda".into()), batch_size: Some(0), learning_rate: Some(0.0), ..Default::default() };
        let err = validate_config(&cfg, &root()).unwrap_err();
        assert!(err.contains(&ConfigValidationError::InvalidBatchSize));
        assert!(err.contains(&ConfigValidationError::InvalidLearningRate));
    }

    #[test]
    fn valid_cuda_config_passes() {
        // Arrange
        let tmp = tempfile::tempdir().unwrap();
        let train = tmp.path().join("train.py");
        fs::write(&train, "print('ok')\n").unwrap();
        let cfg = Config { backend: Some("cuda".into()), train_script: Some(train.file_name().unwrap().to_string_lossy().to_string()), ..Default::default() };
        // Act + Assert
        assert!(validate_config(&cfg, tmp.path()).is_ok());
    }

    #[test]
    fn valid_tilelang_config_passes() {
        let tmp = tempfile::tempdir().unwrap();
        let tile = tmp.path().join("train.tile");
        fs::write(&tile, "tile script").unwrap();
        let cfg = Config { backend: Some("tilelang".into()), tile_script: Some(tile.file_name().unwrap().to_string_lossy().to_string()), ..Default::default() };
        assert!(validate_config(&cfg, tmp.path()).is_ok());
    }
}
