#[test]
fn gpu_monitor_returns_values_or_simulated() {
    let s = rust_watchdog::gpu_monitor::poll();
    assert!(s.total_mb.is_some());
    assert!(s.used_mb.is_some());
    // simulated may be true on non-GPU systems; both branches acceptable
}
