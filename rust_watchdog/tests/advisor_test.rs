use rust_watchdog::advisor::recommend;
use rust_watchdog::config::Config;
use rust_watchdog::gpu_monitor::GpuStatus;

#[test]
fn test_recommend_oom() {
    let cfg = Config {
        batch_size: Some(64),
        learning_rate: Some(0.001),
        ..Default::default()
    };
    let gpu = GpuStatus { total_mb: Some(8192), used_mb: Some(8000), free_mb: Some(192), utilization_pct: Some(95), temperature_c: Some(75), simulated: false };
    let logs = vec!["Epoch 1".to_string(), "RuntimeError: CUDA out of memory. Tried to allocate 500MB.".to_string()];

    let reco = recommend(&cfg, &gpu, &logs);

    assert_eq!(reco.new_batch_size, Some(32)); // Halved
    assert_eq!(reco.new_learning_rate, Some(0.0005)); // Scaled linearly with batch
    assert_eq!(reco.grad_accum_steps, Some(2)); // Preserves effective batch
}

#[test]
fn test_recommend_optimizer_by_params() {
    let cfg = Config {
        param_count: Some(150_000_000),
        optimizer: Some("SGD".to_string()),
        ..Default::default()
    };
    let gpu = GpuStatus { total_mb: Some(8192), used_mb: Some(2000), free_mb: Some(6192), utilization_pct: Some(25), temperature_c: Some(45), simulated: false };
    let logs = vec![];

    let reco = recommend(&cfg, &gpu, &logs);

    assert_eq!(reco.new_optimizer.as_deref(), Some("AdamW"));
}
