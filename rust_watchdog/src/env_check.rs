use anyhow::{Context, Result};
use std::process::Command;
use std::path::{Path, PathBuf};

use crate::config::Config;

/// Find conda.exe by searching PATH and common install locations.
/// Returns the full path to conda.exe if found, or None.
pub fn find_conda() -> Option<PathBuf> {
    // 1) Try PATH directly
    if let Ok(out) = Command::new("conda").arg("--version").output() {
        if out.status.success() {
            return Some(PathBuf::from("conda"));
        }
    }

    // 2) Try cmd /c conda (Windows PATH issue)
    #[cfg(windows)]
    if let Ok(out) = Command::new("cmd").args(["/C", "conda --version"]).output() {
        if out.status.success() {
            return Some(PathBuf::from("conda"));
        }
    }

    // 3) Search common Windows install locations
    #[cfg(windows)]
    {
        let home = std::env::var("USERPROFILE").ok();
        let possible_paths: Vec<PathBuf> = [
            home.as_ref().map(|h| PathBuf::from(h).join("anaconda3").join("Scripts").join("conda.exe")),
            home.as_ref().map(|h| PathBuf::from(h).join("miniconda3").join("Scripts").join("conda.exe")),
            Some(PathBuf::from("C:\\Users\\12599\\anaconda3\\Scripts\\conda.exe")),
            Some(PathBuf::from("C:\\ProgramData\\anaconda3\\Scripts\\conda.exe")),
            Some(PathBuf::from("C:\\ProgramData\\miniconda3\\Scripts\\conda.exe")),
        ].into_iter().flatten().collect();
        for p in possible_paths {
            if p.exists() {
                return Some(p);
            }
        }
    }

    // 4) Search common Unix install locations
    #[cfg(not(windows))]
    {
        let user = std::env::var("USER").unwrap_or_default();
        let possible = [
            PathBuf::from("/home").join(&user).join("anaconda3").join("bin").join("conda"),
            PathBuf::from("/home").join(&user).join("miniconda3").join("bin").join("conda"),
            PathBuf::from("/opt/anaconda3/bin/conda"),
            PathBuf::from("/usr/local/bin/conda"),
        ];
        for p in possible {
            if p.exists() {
                return Some(p);
            }
        }
    }

    None
}

/// Get the conda bin directory (for adding to PATH).
pub fn conda_bin_dir() -> Option<PathBuf> {
    find_conda().and_then(|p| p.parent().map(|pp| pp.to_path_buf()))
}

/// Run a command with conda bin dir added to PATH.
fn run_with_conda_path(cmd: &str, args: &[&str]) -> Result<std::process::Output> {
    let conda_dir = conda_bin_dir();
    let mut command = Command::new(cmd);
    command.args(args);
    if let Some(ref dir) = conda_dir {
        let current_path = std::env::var("PATH").unwrap_or_default();
        #[cfg(windows)]
        let new_path = format!("{};{}", dir.display(), current_path);
        #[cfg(not(windows))]
        let new_path = format!("{}:{}", dir.display(), current_path);
        command.env("PATH", new_path);
    }
    command.output().context("failed to run command")
}

#[allow(dead_code)]
pub fn ensure_preflight(cfg: &Config) -> Result<()> {
    // 1) Check conda presence
    let conda_path = find_conda();
    if conda_path.is_none() {
        eprintln!("[CoLoMo] 未检测到 Conda。请先安装 Anaconda 或 Miniconda。");
        eprintln!("[CoLoMo] 下载: https://docs.conda.io/en/latest/miniconda.html");
        anyhow::bail!("conda not found");
    }
    eprintln!("[CoLoMo] 使用 conda: {}", conda_path.unwrap().display());

    // 2) Ensure env exists
    let env_name = cfg.conda_env.clone().unwrap_or_else(|| "colomo".to_string());
    if !env_exists(&env_name)? {
        eprintln!("[CoLoMo] 未检测到 Conda 环境 '{}'. 是否创建并安装依赖? (y/N)", env_name);
        let mut line = String::new();
        std::io::stdin().read_line(&mut line).ok();
        if line.trim().eq_ignore_ascii_case("y") {
            // Create env
            run_conda_cmd(&["create", "-n", &env_name, "python=3.10", "-y"])
                .context("create env")?;
            // Install requirements if present
            let req = Path::new("projects/demo/requirements.txt");
            if req.exists() {
                run_conda_cmd(&["run", "-n", &env_name, "pip", "install", "-r", req.to_str().unwrap()])
                    .context("pip install -r requirements.txt")?;
            }
            eprintln!("[CoLoMo] 环境已创建并安装依赖: {}", env_name);
        } else {
            eprintln!("[CoLoMo] 跳过创建环境，后续训练可能失败（环境缺失）。");
        }
    }

    Ok(())
}

/// Check if conda is present.
#[allow(dead_code)]
pub fn is_conda_present() -> bool {
    find_conda().is_some()
}

/// Check if a conda environment exists.
pub fn env_exists(env: &str) -> Result<bool> {
    // Use cmd /C on Windows to avoid PATH issues
    #[cfg(windows)]
    {
        let out = run_with_conda_path("cmd", &["/C", "conda", "env", "list"])?;
        let s = String::from_utf8_lossy(&out.stdout);
        Ok(s.lines().any(|l| {
            l.split_whitespace().next().map(|name| name == env).unwrap_or(false)
        }))
    }
    #[cfg(not(windows))]
    {
        let out = run_with_conda_path("conda", &["env", "list"])?;
        let s = String::from_utf8_lossy(&out.stdout);
        Ok(s.lines().any(|l| {
            l.split_whitespace().next().map(|name| name == env).unwrap_or(false)
        }))
    }
}

/// Run a conda command with proper PATH.
pub fn run_conda_cmd(args: &[&str]) -> Result<std::process::Output> {
    let conda_dir = conda_bin_dir();
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
        // Build: cmd /C "conda args..."
        let conda_args = args.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(" ");
        let combined = format!("conda {}", conda_args);
        let mut cmd = Command::new("cmd");
        cmd.args(["/C", &combined]);
        cmd.env("PATH", &new_path);
        cmd.output().context("failed to run conda command")
    }
    #[cfg(not(windows))]
    {
        let mut cmd = Command::new("conda");
        cmd.args(args);
        cmd.env("PATH", &new_path);
        cmd.output().context("failed to run conda command")
    }
}

#[allow(dead_code)]
fn install_miniconda() -> Result<()> {
    #[cfg(windows)]
    {
        let ps1 = Path::new("scripts").join("install_miniconda.ps1");
        if ps1.exists() {
            Command::new("powershell")
                .args(["-ExecutionPolicy", "Bypass", "-File", ps1.to_str().unwrap()])
                .status()?;
            Ok(())
        } else {
            anyhow::bail!("scripts/install_miniconda.ps1 not found");
        }
    }
    #[cfg(not(windows))]
    {
        let sh = Path::new("scripts").join("install_miniconda.sh");
        if sh.exists() {
            Command::new("bash").arg(sh).status()?;
            Ok(())
        } else {
            anyhow::bail!("scripts/install_miniconda.sh not found");
        }
    }
}
