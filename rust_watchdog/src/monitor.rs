//! System Monitor Module
//! Real-time system monitoring with stable parameter adjustment.
//!
//! Key features:
//! - Polls GPU/CPU status at regular intervals
//! - Adjusts parameters conservatively (向下取, 预留空间)
//! - Maintains stability by only making changes when necessary
//! - Provides streaming updates via mpsc channel

use crate::gpu_monitor;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::interval;

/// Shared monitor state for synchronous read access
#[derive(Default)]
pub struct SharedMonitorState {
    pub snapshot: Option<SystemSnapshot>,
    pub recommendation: Option<StableRecommendation>,
}

/// Stability configuration for parameter adjustment
#[derive(Debug, Clone)]
pub struct StabilityConfig {
    /// Minimum free VRAM headroom (MB) - default 20%
    pub vram_headroom_pct: f64,
    /// Minimum free system memory headroom (MB)
    #[allow(dead_code)]
    pub memory_headroom_mb: u64,
    /// Only adjust if change exceeds this threshold (%)
    pub adjustment_threshold_pct: f64,
    /// Cooldown between adjustments (seconds)
    pub adjustment_cooldown_secs: u64,
    /// Maximum batch size allowed
    pub max_batch_size: u32,
    /// Minimum batch size allowed
    pub min_batch_size: u32,
}

impl Default for StabilityConfig {
    fn default() -> Self {
        Self {
            vram_headroom_pct: 0.20,      // Keep 20% VRAM free
            memory_headroom_mb: 2048,     // Keep 2GB RAM free
            adjustment_threshold_pct: 10.0, // Only adjust if >10% change
            adjustment_cooldown_secs: 30,  // 30 second cooldown
            max_batch_size: 256,
            min_batch_size: 1,
        }
    }
}

/// System status snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemSnapshot {
    pub gpu: gpu_monitor::GpuStatus,
    pub system_memory_mb: u64,
    pub used_memory_mb: u64,
    pub free_memory_mb: u64,
    pub cpu_usage_pct: f32,
    pub timestamp_ms: u64,
}

/// Parameter recommendation with stability metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StableRecommendation {
    pub recommended_batch: u32,
    pub recommended_lr: f64,
    pub recommended_accumulation: u32,
    pub current_batch: u32,
    pub change_percentage: f64,
    pub stability_score: f64,  // 0.0-1.0, higher = more stable
    pub reason: String,
    pub gpu_utilization_pct: Option<u32>,
    pub vram_headroom_mb: Option<u64>,
}

/// Monitor event for streaming updates
#[derive(Debug, Clone)]
pub enum MonitorEvent {
    /// New system snapshot available
    Snapshot(SystemSnapshot),
    /// Stable recommendation available
    Recommendation(StableRecommendation),
    /// Parameter adjustment applied
    Adjustment { parameter: String, old_value: String, new_value: String },
    /// Error occurred
    Error(String),
    /// Monitor stopped
    Stopped,
}

/// System monitor that runs in background
pub struct SystemMonitor {
    state: Arc<std::sync::Mutex<SystemMonitorState>>,
    stability: StabilityConfig,
    sender: mpsc::UnboundedSender<MonitorEvent>,
    shared_state: Option<Arc<std::sync::Mutex<SharedMonitorState>>>,
}

struct SystemMonitorState {
    last_adjustment_ms: u64,
    last_recommended_batch: u32,
    is_running: bool,
}

impl SystemMonitor {
    /// Create a new system monitor with streaming
    #[allow(dead_code)]
    pub fn new(stability: StabilityConfig) -> (Self, mpsc::UnboundedReceiver<MonitorEvent>) {
        let (sender, receiver) = mpsc::unbounded_channel();
        let monitor = Self {
            state: Arc::new(std::sync::Mutex::new(SystemMonitorState {
                last_adjustment_ms: 0,
                last_recommended_batch: 32,
                is_running: false,
            })),
            stability,
            sender,
            shared_state: None,
        };
        (monitor, receiver)
    }

    /// Create a new system monitor with a shared synchronous state for UI reading
    pub fn with_shared_state(
        stability: StabilityConfig,
        shared: Arc<std::sync::Mutex<SharedMonitorState>>,
    ) -> (Self, mpsc::UnboundedReceiver<MonitorEvent>) {
        let (sender, receiver) = mpsc::unbounded_channel();
        let monitor = Self {
            state: Arc::new(std::sync::Mutex::new(SystemMonitorState {
                last_adjustment_ms: 0,
                last_recommended_batch: 32,
                is_running: false,
            })),
            stability,
            sender,
            shared_state: Some(shared),
        };
        (monitor, receiver)
    }

    /// Start monitoring with a given poll interval
    pub fn start(self, poll_interval: Duration) {
        let state = self.state.clone();
        let stability = self.stability.clone();
        let sender = self.sender.clone();
        let shared_state = self.shared_state.clone();

        {
            let mut s = state.lock().unwrap();
            s.is_running = true;
        }

        tokio::spawn(async move {
            let mut ticker = interval(poll_interval);
            let mut last_snapshot: Option<SystemSnapshot> = None;

            // Wait for first tick before starting to poll
            ticker.tick().await;

            loop {
                // Check if still running
                {
                    let s = state.lock().unwrap();
                    if !s.is_running {
                        let _ = sender.send(MonitorEvent::Stopped);
                        break;
                    }
                }

                // Poll system status
                let snapshot = Self::poll_system();

                // Update shared state for synchronous UI access
                if let Some(ref shared) = shared_state {
                    if let Ok(mut guard) = shared.lock() {
                        guard.snapshot = Some(snapshot.clone());
                    }
                }

                // Send snapshot event
                let _ = sender.send(MonitorEvent::Snapshot(snapshot.clone()));

                // Calculate stable recommendation
                let recommendation = Self::calculate_stable_recommendation(
                    &snapshot,
                    &stability,
                    &state,
                    last_snapshot.as_ref(),
                );

                // Send recommendation if significant
                if let Some(reco) = recommendation {
                    // Update shared state with recommendation
                    if let Some(ref shared) = shared_state {
                        if let Ok(mut guard) = shared.lock() {
                            guard.recommendation = Some(reco.clone());
                        }
                    }
                    let _ = sender.send(MonitorEvent::Recommendation(reco));
                    last_snapshot = Some(snapshot);
                }

                // Wait for next tick
                ticker.tick().await;
            }
        });
    }

    /// Stop the monitor
    #[allow(dead_code)]
    pub fn stop(&self) {
        let mut s = self.state.lock().unwrap();
        s.is_running = false;
    }

    /// Poll current system status
    fn poll_system() -> SystemSnapshot {
        let gpu = gpu_monitor::poll();

        let mut sys = sysinfo::System::new_all();
        sys.refresh_all();

        let total_mem = sys.total_memory() / 1024 / 1024;
        let used_mem = sys.used_memory() / 1024 / 1024;
        let free_mem = total_mem.saturating_sub(used_mem);
        let cpu = sys.global_cpu_usage();

        SystemSnapshot {
            gpu,
            system_memory_mb: total_mem,
            used_memory_mb: used_mem,
            free_memory_mb: free_mem,
            cpu_usage_pct: cpu,
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        }
    }

    /// Calculate stable recommendation with conservative adjustments
    fn calculate_stable_recommendation(
        snapshot: &SystemSnapshot,
        stability: &StabilityConfig,
        state: &Arc<std::sync::Mutex<SystemMonitorState>>,
        _last_snapshot: Option<&SystemSnapshot>,
    ) -> Option<StableRecommendation> {
        let mut state_guard = state.lock().unwrap();
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        // Check cooldown
        let cooldown_ms = stability.adjustment_cooldown_secs * 1000;
        if now_ms.saturating_sub(state_guard.last_adjustment_ms) < cooldown_ms {
            return None;
        }

        // Calculate available VRAM with headroom
        let total_vram = snapshot.gpu.total_mb.unwrap_or(8192);
        let used_vram = snapshot.gpu.used_mb.unwrap_or(2048);
        let headroom_mb = (total_vram as f64 * stability.vram_headroom_pct) as u64;
        let available_vram = total_vram.saturating_sub(used_vram).saturating_sub(headroom_mb);

        // Estimate safe batch size (rough: ~2GB per batch unit for typical models)
        let bytes_per_batch = 2048u64; // MB per batch unit
        let raw_batch = (available_vram / bytes_per_batch) as u32;

        // Apply floor and bounds
        let safe_batch = raw_batch
            .max(stability.min_batch_size)
            .min(stability.max_batch_size);

        // Calculate change percentage
        let last_batch = state_guard.last_recommended_batch;
        let change_pct = if last_batch > 0 {
            ((safe_batch as f64 - last_batch as f64) / last_batch as f64 * 100.0).abs()
        } else {
            100.0
        };

        // Only recommend if change exceeds threshold
        if change_pct < stability.adjustment_threshold_pct {
            return None;
        }

        // Calculate stability score (0.0-1.0)
        // Higher score = more stable (less change recommended)
        let stability_score = if change_pct > 50.0 {
            0.3 // Large change, low stability
        } else if change_pct > 20.0 {
            0.6 // Medium change
        } else {
            0.8 // Small change, high stability
        };

        // Update state
        state_guard.last_adjustment_ms = now_ms;
        state_guard.last_recommended_batch = safe_batch;

        // Calculate recommended learning rate (linear scaling rule)
        let base_batch = 32u32;
        let base_lr = 0.001;
        let lr_scale = if safe_batch > base_batch {
            (safe_batch as f64 / base_batch as f64).sqrt()
        } else {
            1.0
        };
        let recommended_lr = (base_lr * lr_scale).clamp(1e-6, 0.1);

        // Calculate gradient accumulation if batch increased significantly
        let recommended_accumulation = if safe_batch > last_batch * 2 {
            (safe_batch / last_batch).max(1)
        } else {
            1
        };

        Some(StableRecommendation {
            recommended_batch: safe_batch,
            recommended_lr,
            recommended_accumulation,
            current_batch: last_batch,
            change_percentage: change_pct,
            stability_score,
            reason: format!(
                "VRAM: {}/{} MB ({} headroom), available: {} MB -> batch {}",
                used_vram, total_vram,
                format_bytes(headroom_mb),
                format_bytes(available_vram),
                safe_batch
            ),
            gpu_utilization_pct: snapshot.gpu.utilization_pct,
            vram_headroom_mb: Some(available_vram),
        })
    }
}

/// Format bytes as human-readable string
fn format_bytes(mb: u64) -> String {
    if mb >= 1024 {
        format!("{:.1}GB", mb as f64 / 1024.0)
    } else {
        format!("{}MB", mb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(512), "512MB");
        assert_eq!(format_bytes(1024), "1.0GB");
        assert_eq!(format_bytes(1536), "1.5GB");
    }

    #[tokio::test]
    async fn test_monitor_creation() {
        let config = StabilityConfig::default();
        let (monitor, mut receiver) = SystemMonitor::new(config);

        // Start monitoring - this consumes monitor but we don't need to call stop explicitly
        // as the monitor will be dropped when test ends
        monitor.start(Duration::from_millis(100));

        // Wait for a snapshot
        let event = tokio::time::timeout(Duration::from_secs(1), receiver.recv())
            .await
            .unwrap();

        assert!(event.is_some());
        match event {
            Some(MonitorEvent::Snapshot(s)) => {
                println!("Got snapshot: {:?}", s);
            }
            _ => panic!("Expected Snapshot event"),
        }
        // Monitor dropped here, stop signal sent automatically
    }
}
