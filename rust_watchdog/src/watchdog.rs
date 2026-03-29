use std::{ffi::OsStr, fs::OpenOptions, io::{BufRead, BufReader, Write}, path::PathBuf, process::Stdio, time::Duration};
use anyhow::{Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct Config {
    pub conda_env: Option<String>,
    pub backend: Option<String>,        // "cuda" | "tilelang" | "jupyter"
    pub train_script: Option<String>,   // python script or ipynb path
    pub tile_script: Option<String>,    // tile script path
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct RecentLines {
    pub lines: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct RunStatus {
    pub running: bool,
    pub pid: Option<u32>,
    pub last_exit_code: Option<i32>,
    pub last_exit_time: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LastExit { code: Option<i32>, time: String }

#[allow(dead_code)]
pub struct Watchdog {}

fn status_dir() -> PathBuf { PathBuf::from("status") }
fn pid_path() -> PathBuf { status_dir().join("run.pid") }
fn last_exit_path() -> PathBuf { status_dir().join("last_exit.json") }

#[allow(dead_code)]
pub fn current_pid() -> Option<u32> {
    let p = pid_path();
    let s = std::fs::read_to_string(p).ok()?;
    s.trim().parse::<u32>().ok()
}

#[allow(dead_code)]
pub fn stop_last() -> Result<()> {
    if let Some(pid) = current_pid() {
        if cfg!(target_os = "windows") {
            let _ = Command::new("taskkill").args(["/PID", &pid.to_string(), "/T", "/F"]).status();
        } else {
            let _ = Command::new("kill").args(["-TERM", &pid.to_string()]).status();
            let _ = Command::new("kill").args(["-KILL", &pid.to_string()]).status();
        }
        // Best-effort cleanup
        let _ = std::fs::remove_file(pid_path());
    }
    Ok(())
}

#[allow(dead_code)]
pub fn get_status() -> RunStatus {
    let pid = current_pid();
    let running = pid.is_some();
    let (mut last_exit_code, mut last_exit_time) = (None, None);
    if let Ok(s) = std::fs::read_to_string(last_exit_path()) {
        if let Ok(v) = serde_json::from_str::<LastExit>(&s) {
            last_exit_code = v.code;
            last_exit_time = Some(v.time);
        }
    }
    RunStatus { running, pid, last_exit_code, last_exit_time }
}

impl Watchdog {
    #[allow(dead_code)]
    pub fn run(config: &Config, project_root: PathBuf, logs_dir: PathBuf) -> Result<()> {
        let env_name = config.conda_env.clone().unwrap_or_else(|| "colomo".to_string());
        let backend = config.backend.clone().unwrap_or_else(|| "cuda".to_string());
        let (bin, args): (&str, Vec<String>) = match backend.as_str() {
            "tilelang" => {
                let tile = config
                    .tile_script
                    .clone()
                    .unwrap_or_else(|| "projects/demo/train.tile".to_string());
                (
                    "conda",
                    vec![
                        "run".into(), "-n".into(), env_name.clone(),
                        "tilelang".into(), "run".into(), "--script".into(), tile,
                        "--config".into(), "projects/demo/config.yaml".into(),
                    ],
                )
            }
            "jupyter" => {
                let train = config
                    .train_script
                    .clone()
                    .unwrap_or_else(|| "projects/demo/train.ipynb".to_string());
                (
                    "conda",
                    vec![
                        "run".into(), "-n".into(), env_name.clone(),
                        "jupyter".into(), "nbconvert".into(),
                        "--to".into(), "notebook".into(),
                        "--execute".into(), "--inplace".into(),
                        train,
                    ],
                )
            }
            _ => {
                let train = config
                    .train_script
                    .clone()
                    .unwrap_or_else(|| "projects/demo/train.py".to_string());
                (
                    "conda",
                    vec![
                        "run".into(), "-n".into(), env_name.clone(),
                        "python".into(), "-u".into(), train,
                    ],
                )
            }
        };

        let run_id = Utc::now().format("%Y%m%dT%H%M%S").to_string();
        std::fs::create_dir_all(&logs_dir)
            .with_context(|| format!("failed to create logs dir: {}", logs_dir.display()))?;
        let mut log_path = logs_dir.clone();
        log_path.push(format!("run-{}.log", run_id));

        let mut cmd = Command::new(bin);
        cmd.args(&args)
            .current_dir(&project_root)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = cmd.spawn().with_context(|| "failed to spawn training process")?;

        // Write PID status file (in our own working dir)
        std::fs::create_dir_all(status_dir())?;
        std::fs::write(pid_path(), format!("{}", child.id()))?;

        let stdout = child.stdout.take().context("no stdout")?;
        let stderr = child.stderr.take().context("no stderr")?;

        let file = OpenOptions::new().create(true).append(true).open(&log_path)
            .with_context(|| format!("open log file {}", log_path.display()))?;
        let mut writer = std::io::BufWriter::new(&file);

        let stdout_reader = BufReader::new(stdout);
        let stderr_reader = BufReader::new(stderr);

        let mut out_lines = stdout_reader.lines();
        let mut err_lines = stderr_reader.lines();

        let exit_code: Option<i32>;
        loop {
            let mut progressed = false;
            if let Some(Ok(line)) = out_lines.next() { writeln!(writer, "[STDOUT] {}", line)?; progressed = true; }
            if let Some(Ok(line)) = err_lines.next() { writeln!(writer, "[STDERR] {}", line)?; progressed = true; }
            writer.flush()?;
            match child.try_wait()? {
                Some(status) => {
                    exit_code = status.code();
                    // Drain remaining buffered lines
                    for l in out_lines.map_while(Result::ok) { writeln!(writer, "[STDOUT] {}", l)?; }
                    for l in err_lines.map_while(Result::ok) { writeln!(writer, "[STDERR] {}", l)?; }
                    writer.flush()?;
                    break;
                }
                None => {
                    if !progressed { std::thread::sleep(Duration::from_millis(10)); }
                }
            }
        }
        // Record last exit and clean up PID file when process exits
        let last = LastExit { code: exit_code, time: Utc::now().to_rfc3339() };
        let _ = std::fs::write(last_exit_path(), serde_json::to_string(&last).unwrap_or_else(|_| "{}".to_string()));
        let _ = std::fs::remove_file(pid_path());
        Ok(())
    }

    #[allow(dead_code)]
    pub fn get_recent_lines(logs_dir: PathBuf, n: usize) -> Result<RecentLines> {
        let mut entries: Vec<PathBuf> = std::fs::read_dir(&logs_dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension() == Some(OsStr::new("log")))
            .collect();
        entries.sort();
        let Some(latest) = entries.last() else { return Ok(RecentLines{ lines: vec![] }); };
        let file = std::fs::File::open(latest)?;
        let reader = BufReader::new(file);
        let mut lines: Vec<String> = reader.lines().map_while(Result::ok).collect();
        let len = lines.len();
        if len > n { lines = lines.split_off(len - n); }
        Ok(RecentLines { lines })
    }
}
