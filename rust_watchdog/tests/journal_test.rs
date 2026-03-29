use std::fs;

#[test]
fn journal_append_writes_valid_jsonl() {
    let path = std::env::temp_dir().join("colomo_journal_test.jsonl");
    // ensure clean
    let _ = fs::remove_file(&path);

    rust_watchdog::journal::append(
        path.to_str().unwrap(),
        "modify_config",
        &serde_json::json!({"file":"/tmp/x.yaml","field":"batch_size","old":64,"new":32}),
        Some(serde_json::json!({"action":"modify_config","params":{"field":"batch_size","value":64}})),
    ).expect("append");

    let content = fs::read_to_string(&path).expect("read");
    let line = content.lines().next().expect("one line");
    let v: serde_json::Value = serde_json::from_str(line).expect("json");
    assert_eq!(v["action"], "modify_config");
    assert!(v["ts"].as_str().is_some());
    assert!(v["inverse"].is_object());
}
