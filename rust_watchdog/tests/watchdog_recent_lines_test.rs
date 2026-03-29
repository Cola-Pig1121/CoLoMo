use std::fs;
use std::io::Write;
use std::path::PathBuf;

#[test]
fn get_recent_lines_reads_last_log() {
    let dir = std::env::temp_dir().join("colomo_logs_test");
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("mkdir");

    let file = dir.join("run-0001.log");
    let mut f = fs::OpenOptions::new().create(true).write(true).open(&file).unwrap();
    writeln!(f, "line1").unwrap();
    writeln!(f, "line2").unwrap();

    let res = rust_watchdog::watchdog::Watchdog::get_recent_lines(PathBuf::from(&dir), 1).expect("read");
    assert_eq!(res.lines.len(), 1);
    assert!(res.lines[0].contains("line2"));
}
