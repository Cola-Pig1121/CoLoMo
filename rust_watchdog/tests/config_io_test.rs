#[test]
fn config_load_and_save_roundtrip() {
    let tmp = std::env::temp_dir().join("colomo_cfg_test.yaml");
    let cfg = rust_watchdog::config::Config {
        conda_env: Some("colomo".into()),
        backend: Some("cuda".into()),
        model_template: Some("pytorch".into()),
        train_script: Some("projects/demo/train.py".into()),
        tile_script: None,
        batch_size: Some(64),
        learning_rate: Some(1e-3),
        optimizer: Some("AdamW".into()),
        dataset_path: Some("./data".into()),
        param_count: Some(120_000_000),
    };

    rust_watchdog::config::save_config(&tmp, &cfg).expect("save");
    let loaded = rust_watchdog::config::load_config(&tmp).expect("load");

    assert_eq!(loaded.conda_env.as_deref(), Some("colomo"));
    assert_eq!(loaded.backend.as_deref(), Some("cuda"));
    assert_eq!(loaded.batch_size, Some(64));
    assert_eq!(loaded.optimizer.as_deref(), Some("AdamW"));
}
