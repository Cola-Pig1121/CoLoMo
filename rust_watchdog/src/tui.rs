use anyhow::Result;
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style, Color},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph, Row, Table, TableState},
    Frame,
};
use std::cell::RefCell;
use std::path::PathBuf;

use crate::config::{self, Config};
use crate::journal;
use crate::watchdog::Watchdog;
use crate::advisor;
use crate::gpu_monitor;
use crate::settings::{self, Settings};

pub struct GuidedState {
    pub _root_path: PathBuf,
    pub cfg_path: PathBuf,
    pub cfg: Config,
    pub table_state: RefCell<TableState>,
    pub message: Option<String>,
    pub last_reco: Option<advisor::Recommendation>,
    pub input_mode: bool,
    pub input_buf: String,
    pub settings: Settings,
    pub settings_path: PathBuf,
    pub autocomplete_items: Vec<String>,
    pub autocomplete_selected: usize,
    pub show_autocomplete: bool,
    pub lang: String,
    pub expert_mode: bool,
}

#[allow(dead_code)]
pub struct Labels {
    pub help_hint: String,
    pub no_logs: String,
    pub press_a: String,
    pub no_changes: String,
    pub saved: String,
    pub applied: String,
    pub no_reco: String,
    pub unknown_cmd: String,
    pub config_table: Vec<String>,
    pub guided_config: String,
    pub recommendation: String,
    pub interaction: String,
    pub recent_logs: String,
    pub summary: String,
}

impl GuidedState {
    pub fn new(root_path: PathBuf, cfg_path: PathBuf, cfg: Config) -> Self {
        let mut ts = TableState::default();
        ts.select(Some(0));
        let settings_path = PathBuf::from("settings.yaml");
        let settings = settings::load(&settings_path).unwrap_or_default();
        Self { _root_path: root_path, cfg_path, cfg, table_state: RefCell::new(ts), message: None, last_reco: None, input_mode: false, input_buf: String::new(), settings, settings_path, autocomplete_items: Vec::new(), autocomplete_selected: 0, show_autocomplete: false, lang: "zh-CN".into(), expert_mode: false }
    }

    /// Returns all available commands filtered by current input prefix, in the current language.
    pub fn get_commands(&self) -> Vec<String> {
        let prefix = self.input_buf.to_lowercase();
        let cmds_zh = vec![
            "/setting safety_alpha=",
            "/setting learning_mode=",
            "/setting acc=",
            "/setting lat=",
            "/setting mem=",
            "/setting thr=",
            "/setting energy=",
            "/rollback",
            "/language",
            "/create",
            "/open",
            "/expert",
            "/guided",
            "/save",
            "/run",
            "/apply",
            "/stop",
            "/edit-own",
        ];
        let cmds_en = vec![
            "/setting safety_alpha=",
            "/setting learning_mode=",
            "/setting acc=",
            "/setting lat=",
            "/setting mem=",
            "/setting thr=",
            "/setting energy=",
            "/rollback",
            "/language",
            "/create",
            "/open",
            "/expert",
            "/guided",
            "/save",
            "/run",
            "/apply",
            "/stop",
            "/edit-own",
        ];
        let cmds = if self.lang == "zh-CN" { &cmds_zh } else { &cmds_en };
        cmds.iter().filter(|c| c.to_lowercase().starts_with(&prefix)).map(|s| (*s).to_string()).collect()
    }

    /// Human-readable labels for the current language.
    pub fn labels(&self) -> Labels {
        if self.lang == "zh-CN" {
            Labels {
                help_hint: "输入命令（如 /setting safety_alpha=0.9），回车确认，Esc 取消".into(),
                no_logs: "(暂无日志)".into(),
                press_a: "按 A 应用推荐".into(),
                no_changes: "无需调整".into(),
                saved: "已保存配置并记录".into(),
                applied: "已应用推荐".into(),
                no_reco: "无可用推荐".into(),
                unknown_cmd: "未知命令: ".into(),
                config_table: vec!["字段".into(), "值".into()],
                guided_config: "引导配置".into(),
                recommendation: "推荐".into(),
                interaction: "交互".into(),
                recent_logs: "最近日志".into(),
                summary: "摘要".into(),
            }
        } else {
            Labels {
                help_hint: "Type command (e.g. /setting safety_alpha=0.9), Enter to confirm, Esc to cancel".into(),
                no_logs: "(no logs yet)".into(),
                press_a: "Press A to apply recommendation".into(),
                no_changes: "No changes suggested".into(),
                saved: "Saved config and journaled".into(),
                applied: "Applied recommendation".into(),
                no_reco: "No recommendation available".into(),
                unknown_cmd: "Unknown command: ".into(),
                config_table: vec!["Field".into(), "Value".into()],
                guided_config: "Guided Config".into(),
                recommendation: "Recommendation".into(),
                interaction: "Interaction".into(),
                recent_logs: "Recent Logs".into(),
                summary: "Summary".into(),
            }
        }
    }

    fn rows(&self) -> Vec<[String; 2]> {
        vec![
            ["conda_env".into(), self.cfg.conda_env.clone().unwrap_or_default()],
            ["backend".into(), self.cfg.backend.clone().unwrap_or_else(|| "cuda".into())],
            ["train_script".into(), self.cfg.train_script.clone().unwrap_or("projects/demo/train.py".into())],
            ["batch_size".into(), self.cfg.batch_size.map(|v| v.to_string()).unwrap_or_default()],
            ["learning_rate".into(), self.cfg.learning_rate.map(|v| v.to_string()).unwrap_or_default()],
            ["optimizer".into(), self.cfg.optimizer.clone().unwrap_or_default()],
            ["dataset_path".into(), self.cfg.dataset_path.clone().unwrap_or_default()],
        ]
    }

    pub fn on_save(&mut self) -> Result<()> {
        let abs = std::env::current_dir()?.join(&self.cfg_path);
        let old_cfg = crate::journal::snapshot_config(&abs);
        config::save_config(&abs, &self.cfg)?;
        crate::journal::append(
            "journal.jsonl",
            "modify_config",
            &serde_json::json!({ "file": abs }),
            Some(serde_json::json!({ "old_config": old_cfg })),
        )?;
        self.message = Some("Saved config and journaled".into());
        Ok(())
    }

    pub fn on_run(&mut self) {
        let project_root = PathBuf::from("..");
        let logs_dir = PathBuf::from("logs");
        let cfg = self.cfg.clone();
        let _ = std::thread::spawn(move || {
            let _ = Watchdog::run(&crate::watchdog::Config {
                conda_env: cfg.conda_env.clone(),
                backend: cfg.backend.clone(),
                train_script: cfg.train_script.clone(),
                tile_script: None,
            }, project_root, logs_dir);
        });
        self.message = Some("Launched training (see logs)".into());
    }

    pub fn compute_recommendation(&mut self, recent_logs: &[String]) {
        let gpu = gpu_monitor::poll();
        let reco = advisor::recommend(&self.cfg, &gpu, recent_logs);
        self.last_reco = Some(reco);
    }

    pub fn apply_recommendation(&mut self) -> Result<()> {
        if let Some(reco) = &self.last_reco {
            let abs = std::env::current_dir()?.join(&self.cfg_path);
            let old_cfg = crate::journal::snapshot_config(&abs);
            let mut changed = false;
            if let Some(b) = reco.new_batch_size { self.cfg.batch_size = Some(b); changed = true; }
            if let Some(lr) = reco.new_learning_rate { self.cfg.learning_rate = Some(lr); changed = true; }
            if let Some(ref opt) = reco.new_optimizer { self.cfg.optimizer = Some(opt.clone()); changed = true; }
            if changed {
                config::save_config(&abs, &self.cfg)?;
            }
            journal::append(
                "journal.jsonl",
                "apply_recommendation",
                &serde_json::json!({
                    "file": abs,
                    "applied": changed,
                    "grad_accum_steps": reco.grad_accum_steps,
                    "rationale": reco.rationale,
                }),
                Some(serde_json::json!({ "old_config": old_cfg })),
            )?;
            self.message = Some("Applied recommendation".into());
        } else {
            self.message = Some("No recommendation available".into());
        }
        Ok(())
    }

    pub fn handle_command(&mut self, cmd: &str) {
        let parts: Vec<&str> = cmd.split_whitespace().collect();
        if parts.is_empty() { return; }
        match parts[0] {
            "/setting" => {
                for kv in &parts[1..] {
                    if let Some((k, v)) = kv.split_once('=') {
                        match k {
                            "safety_alpha" => if let Ok(x) = v.parse() { self.settings.safety_alpha = x; },
                            "learning_mode" => if let Ok(x) = v.parse() { self.settings.learning_mode = x; },
                            "acc" => if let Ok(x) = v.parse() { self.settings.weights.acc = x; },
                            "lat" => if let Ok(x) = v.parse() { self.settings.weights.lat = x; },
                            "mem" => if let Ok(x) = v.parse() { self.settings.weights.mem = x; },
                            "thr" => if let Ok(x) = v.parse() { self.settings.weights.thr = x; },
                            "energy" => if let Ok(x) = v.parse() { self.settings.weights.energy = x; },
                            _ => {}
                        }
                    }
                }
                if let Err(e) = settings::save(&self.settings_path, &self.settings) {
                    self.message = Some(format!("Failed to save settings: {}", e));
                } else {
                    self.message = Some("Settings updated".into());
                }
            }
            "/rollback" => {
                let mut steps = 1;
                if parts.len() > 1 {
                    if let Ok(s) = parts[1].parse() { steps = s; }
                }
                // Placeholder: invoke rollback logic here
                self.message = Some(format!("Rollback {} step(s) pending implementation", steps));
            }
            "/language" => {
                let lang = self.lang.clone();
                self.lang = if lang == "zh-CN" { "en".into() } else { "zh-CN".into() };
                self.message = Some(format!("Language: {}", self.lang));
            }
            "/expert" => {
                self.expert_mode = true;
                self.message = Some("Expert mode: use /edit-own to edit files".into());
            }
            "/guided" => {
                self.expert_mode = false;
                self.message = Some("Guided mode: use /setting to modify parameters".into());
            }
            "/create" => {
                self.message = Some("Use: /create <name> <template>\nTemplates: resnet18, pytorch, tensorflow, lora, full-finetune".into());
            }
            "/open" => {
                self.message = Some("Use: /open <project_path>\nOpens an existing project from projects/<name>".into());
            }
            "/edit-own" => {
                if self.expert_mode {
                    let train_name = self.cfg.train_script.clone().unwrap_or_else(|| "train.py".into());
                    let train_path = std::env::current_dir()
                        .map(|d| d.join(train_name))
                        .unwrap_or_else(|_| PathBuf::from("train.py"));
                    let editor = std::env::var("EDITOR").unwrap_or_else(|_| "vi".into());
                    if let Err(e) = std::process::Command::new(&editor).arg(&train_path).status() {
                        self.message = Some(format!("Failed to open editor: {}", e));
                    } else {
                        self.message = Some(format!("Opened {} in {}", train_path.display(), editor));
                    }
                } else {
                    self.message = Some("Switch to expert mode first: /expert".into());
                }
            }
            "/save" => {
                if let Err(e) = self.on_save() {
                    self.message = Some(format!("Save failed: {}", e));
                } else {
                    self.message = Some("Config saved".into());
                }
            }
            "/run" => {
                self.on_run();
                self.message = Some("Training started".into());
            }
            "/apply" => {
                if let Err(e) = self.apply_recommendation() {
                    self.message = Some(format!("Apply failed: {}", e));
                }
            }
            "/stop" => {
                if let Err(e) = crate::watchdog::stop_last() {
                    self.message = Some(format!("Stop failed: {}", e));
                } else {
                    self.message = Some("Training stopped".into());
                }
            }
            _ => {
                self.message = Some(format!("{}{}", self.labels().unknown_cmd, parts[0]));
            }
        }
    }

    pub fn update_autocomplete(&mut self) {
        if !self.input_buf.starts_with('/') {
            self.show_autocomplete = false;
            self.autocomplete_items.clear();
            return;
        }
        self.autocomplete_items = self.get_commands();
        self.autocomplete_selected = 0;
        self.show_autocomplete = !self.autocomplete_items.is_empty();
    }

    pub fn autocomplete_selected_text(&self) -> Option<String> {
        self.autocomplete_items.get(self.autocomplete_selected).cloned()
    }
}

pub fn draw_guided(f: &mut Frame<'_>, area: Rect, state: &GuidedState) {
    let lbl = state.labels();
    let rows = state
        .rows()
        .into_iter()
        .map(|r| Row::new(vec![r[0].clone(), r[1].clone()]));
    let widths = [Constraint::Percentage(40), Constraint::Percentage(60)];
    let table = Table::new(rows, widths)
        .header(Row::new(vec![lbl.config_table[0].clone(), lbl.config_table[1].clone()]).style(Style::default().add_modifier(Modifier::BOLD)))
        .block(Block::default().title(lbl.guided_config.as_str()).borders(Borders::ALL));

    // Reserve bottom space for command input if in input mode
    let mut table_area = area;
    if state.input_mode {
        let mut constraints = vec![Constraint::Min(0), Constraint::Length(3)];
        if state.show_autocomplete && !state.autocomplete_items.is_empty() {
            let list_height = state.autocomplete_items.len().min(4);
            constraints.push(Constraint::Length(list_height as u16));
        }
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints(constraints)
            .split(area);
        table_area = chunks[0];

        let input_block = Paragraph::new(Line::from(vec![
            Span::styled("> ", Style::default().fg(Color::Yellow)),
            Span::raw(&state.input_buf),
            Span::styled("█", Style::default().add_modifier(Modifier::RAPID_BLINK)),
        ])).block(Block::default().borders(Borders::ALL).title("Command Input"));
        f.render_widget(input_block, chunks[1]);

        // Render autocomplete list below input block
        if state.show_autocomplete && !state.autocomplete_items.is_empty() {
            let list_height = state.autocomplete_items.len().min(4);
            let ac_lines: Vec<Line> = state.autocomplete_items
                .iter()
                .take(list_height)
                .enumerate()
                .map(|(i, item)| {
                    if i == state.autocomplete_selected {
                        Line::from(vec![
                            Span::styled("  > ", Style::default().fg(Color::Cyan)),
                            Span::styled(item, Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                        ])
                    } else {
                        Line::from(vec![
                            Span::raw("    "),
                            Span::raw(item),
                        ])
                    }
                })
                .collect();
            let ac_hint = Line::from(vec![
                Span::raw("  "),
                Span::styled("↑↓", Style::default().fg(Color::DarkGray)),
                Span::raw(" navigate  "),
                Span::styled("Enter", Style::default().fg(Color::DarkGray)),
                Span::raw(" select  "),
                Span::styled("Esc", Style::default().fg(Color::DarkGray)),
                Span::raw(" cancel"),
            ]);
            let ac_block = Paragraph::new({
                let mut lines = ac_lines;
                lines.push(ac_hint);
                lines
            }).block(Block::default().borders(Borders::ALL).title("Commands"));
            f.render_widget(ac_block, chunks[2]);
        }
    }

    let mut table_state = state.table_state.borrow_mut();
    f.render_stateful_widget(table, table_area, &mut table_state);

    // Recommendation panel below the table inside same area (only if not in input mode)
    if !state.input_mode {
        let sub = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(3), Constraint::Length(5)])
            .split(table_area);

        if let Some(reco) = &state.last_reco {
            let mut lines: Vec<Line> = Vec::new();
            if let Some(b) = reco.new_batch_size { lines.push(Line::from(format!("batch_size → {}", b))); }
            if let Some(lr) = reco.new_learning_rate { lines.push(Line::from(format!("learning_rate → {:.6}", lr))); }
            if let Some(ref opt) = reco.new_optimizer { lines.push(Line::from(format!("optimizer → {}", opt))); }
            if let Some(acc) = reco.grad_accum_steps { lines.push(Line::from(format!("grad_accum_steps → {} (advisory)", acc))); }
            if lines.is_empty() { lines.push(Line::from("No changes suggested")); }
            lines.push(Line::from("Press A to apply recommendation"));
            let recob = Paragraph::new(lines)
                .block(Block::default().title("Recommendation").borders(Borders::ALL));
            f.render_widget(recob, sub[1]);
        }
    }

    if let Some(msg) = &state.message {
        let popup = Paragraph::new(vec![Line::from(msg.as_str())])
            .block(Block::default().title("Info").borders(Borders::ALL));
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(40), Constraint::Percentage(20), Constraint::Percentage(40)])
            .split(area);
        let area_mid = chunks[1];
        f.render_widget(Clear, area_mid);
        f.render_widget(popup, area_mid);
    }
}
