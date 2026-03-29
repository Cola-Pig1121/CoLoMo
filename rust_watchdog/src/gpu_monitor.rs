use std::process::Command;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuStatus {
    pub total_mb: Option<u64>,
    pub used_mb: Option<u64>,
    pub free_mb: Option<u64>,
    pub utilization_pct: Option<u32>,
    pub temperature_c: Option<u32>,
    pub simulated: bool,
}

impl Default for GpuStatus {
    fn default() -> Self {
        Self {
            total_mb: Some(8192),
            used_mb: Some(2048),
            free_mb: Some(6144),
            utilization_pct: Some(25),
            temperature_c: None,
            simulated: true,
        }
    }
}

impl GpuStatus {
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| r#"{"error":"serialization_failed"}"#.to_string())
    }
}

pub fn poll() -> GpuStatus {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output();

    match output {
        Ok(o) if o.status.success() => {
            let s = String::from_utf8_lossy(&o.stdout);
            let parts: Vec<&str> = s.trim().split(',').map(|x| x.trim()).collect();
            let total = parts.first().and_then(|x| x.parse::<u64>().ok());
            let used = parts.get(1).and_then(|x| x.parse::<u64>().ok());
            let free = parts.get(2).and_then(|x| x.parse::<u64>().ok());
            let util = parts.get(3).and_then(|x| x.parse::<u32>().ok());
            let temp = parts.get(4).and_then(|x| x.parse::<u32>().ok());
            GpuStatus {
                total_mb: total,
                used_mb: used,
                free_mb: free,
                utilization_pct: util,
                temperature_c: temp,
                simulated: false,
            }
        }
        _ => GpuStatus::default(),
    }
}
