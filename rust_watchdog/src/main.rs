use anyhow::Result;
use ratatui_kit::ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph, Row, Table, TableState, Wrap},
    Frame,
};
use ratatui_kit::{Component, ComponentDrawer, ComponentUpdater, Hooks, NoProps, UseState};
use std::cell::RefCell;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

mod advisor;
mod config;
mod env_check;
mod gpu_monitor;
mod journal;
mod llm_agent;
mod monitor;
mod project_creator;
mod settings;
mod tools;
mod watchdog;

// ============================================================================
// Types
// ============================================================================

/// Step in interactive project creation flow
#[derive(Debug, Clone, Copy, PartialEq)]
enum CreationStep {
    Name,    // Waiting for project name input
    Path,    // Waiting for project path input
    Confirm, // Confirm creation
}

impl Default for CreationStep {
    fn default() -> Self {
        CreationStep::Name
    }
}

#[derive(Debug, Clone)]
pub enum WatchdogCmd {
    Save,
    Run,
    ApplyRecommendation(advisor::Recommendation),
    Stop,
}

/// UI-bound events sent from spawned tasks to the UI loop via mpsc.
/// Using mpsc instead of tokio channel ensures non-blocking send from any thread.
#[derive(Debug, Clone)]
pub enum UxEvent {
    AppendOutput(String),
    SetActiveAgent(String),
    SetAgentTask(String),
    SetAgentProgress(String),
    SetSummaryLines(Vec<String>),
    SetTopRightLines(Vec<String>),
    SetOutputScroll(usize),
    SetRunning(bool),
    SetAgentLastSummary(String),
}

#[derive(Default)]
pub struct WatchdogState {
    pub gpu_info: String,
    pub recent_logs: Vec<String>,
    pub recommendation: Option<advisor::Recommendation>,
    pub summary_lines: Vec<String>,
    pub running: bool,
}

// ============================================================================
// Shared State
// ============================================================================

impl Default for AppState {
    fn default() -> Self {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        Self {
            lang: "zh-CN".into(),
            expert_mode: false,
            input_mode: false,
            input_buf: String::new(),
            cursor_pos: 0,
            selection_start: None,
            selection_end: None,
            message: String::new(),
            autocomplete_items: Vec::new(),
            autocomplete_selected: 0,
            autocomplete_scroll: 0,
            show_autocomplete: false,
            gpu_info: String::new(),
            recent_logs: Vec::new(),
            recommendation: None,
            summary_lines: Vec::new(),
            running: false,
            cfg: config::Config::default(),
            cfg_path: PathBuf::from("config.yaml"),
            table_state: RefCell::new(TableState::default()),
            consecutive_ctrl_c: 0,
            settings: settings::Settings::default(),
            active_agent: None,
            agent_task: None,
            agent_progress: None,
            monitor_receiver: None,
            current_snapshot: None,
            stable_recommendation: None,
            top_right_lines: Vec::new(),
            plan_exists: false,
            agent_last_summary: None,
            ux_tx: tx,
            ux_rx: rx,
            creating_project: false,
            creation_step: CreationStep::Name,
            new_project_name: String::new(),
            new_project_path: String::new(),
            output_lines: Vec::new(),
            output_scroll: 0,
            input_scroll_y: 0,
            input_scroll_x: 0,
        }
    }
}

pub struct AppState {
    pub lang: String,
    pub expert_mode: bool,
    pub input_mode: bool,
    pub input_buf: String,
    pub cursor_pos: usize,              // Cursor position in input_buf
    pub selection_start: Option<usize>, // Selection start (None = no selection)
    pub selection_end: Option<usize>,   // Selection end
    pub message: String,
    pub autocomplete_items: Vec<(String, String)>, // (command, hint)
    pub autocomplete_selected: usize,
    pub autocomplete_scroll: usize,
    pub show_autocomplete: bool,
    pub gpu_info: String,
    pub recent_logs: Vec<String>,
    pub recommendation: Option<advisor::Recommendation>,
    pub summary_lines: Vec<String>,
    pub running: bool,
    pub cfg: config::Config,
    pub cfg_path: PathBuf,
    pub table_state: RefCell<TableState>,
    pub consecutive_ctrl_c: u8,
    pub settings: settings::Settings,

    // Agent running status
    pub active_agent: Option<String>, // e.g., Some("planner".into())
    pub agent_task: Option<String>,   // e.g., Some("Analyzing requirement".into())
    pub agent_progress: Option<String>, // e.g., Some("3/5 steps".into())

    // System monitor fields
    pub monitor_receiver: Option<tokio::sync::mpsc::UnboundedReceiver<monitor::MonitorEvent>>,
    pub current_snapshot: Option<monitor::SystemSnapshot>,
    pub stable_recommendation: Option<monitor::StableRecommendation>,

    // Top-right panel: plan/todo display
    pub top_right_lines: Vec<String>,
    pub plan_exists: bool,
    pub agent_last_summary: Option<String>,

    // Non-blocking UI event channel (mpsc from spawned tasks to UI loop)
    pub ux_tx: tokio::sync::mpsc::UnboundedSender<UxEvent>,
    pub ux_rx: tokio::sync::mpsc::UnboundedReceiver<UxEvent>,

    // Project creation state
    pub creating_project: bool,
    pub creation_step: CreationStep,
    pub new_project_name: String,
    pub new_project_path: String,

    // Output textarea (replaces message popup)
    pub output_lines: Vec<String>,
    pub output_scroll: usize,
    pub input_scroll_y: usize,
    pub input_scroll_x: usize,
}

static APP_STATE: once_cell::sync::OnceCell<Arc<Mutex<AppState>>> =
    once_cell::sync::OnceCell::new();

/// Global UX event sender for spawned tasks to push UI updates without blocking.
/// Initialized in main() before the UI loop starts.
static UX_GLOBAL_TX: once_cell::sync::OnceCell<tokio::sync::mpsc::UnboundedSender<UxEvent>> =
    once_cell::sync::OnceCell::new();

/// Send a UX event to the UI loop (non-blocking from any thread).
/// Panics if not initialized.
pub fn send_ux(event: UxEvent) {
    if let Some(tx) = UX_GLOBAL_TX.get() {
        let _ = tx.send(event);
    }
}

fn get_state() -> &'static Arc<Mutex<AppState>> {
    APP_STATE.get_or_init(|| Arc::new(Mutex::new(AppState::default())))
}

// Shared monitor state that can be read synchronously
use std::sync::Arc as StdArc;
use std::sync::Mutex as StdMutex;

static MONITOR_STATE: once_cell::sync::OnceCell<StdArc<StdMutex<monitor::SharedMonitorState>>> =
    once_cell::sync::OnceCell::new();

fn get_monitor_state() -> &'static StdArc<StdMutex<monitor::SharedMonitorState>> {
    MONITOR_STATE.get_or_init(|| StdArc::new(StdMutex::new(monitor::SharedMonitorState::default())))
}

// ============================================================================
// CoLoMo Tui Component
// ============================================================================

pub struct CoLoMoTui;

impl Component for CoLoMoTui {
    type Props<'a> = NoProps;

    fn new(_props: &Self::Props<'_>) -> Self {
        Self
    }

    fn update(
        &mut self,
        _props: &mut Self::Props<'_>,
        mut hooks: Hooks,
        _updater: &mut ComponentUpdater,
    ) {
        let first_setup = hooks.use_state(|| false);
        let already_setup = *first_setup.read();
        if !already_setup {
            let cfg_path = PathBuf::from("../projects/demo/config.yaml");
            let cfg = config::load_config(&cfg_path).unwrap_or_default();
            let settings = settings::load(&PathBuf::from("settings.yaml")).unwrap_or_default();

            // Create shared monitor state and start system monitor
            let shared = get_monitor_state().clone();
            let stability = monitor::StabilityConfig::default();
            let (monitor_handle, receiver) =
                monitor::SystemMonitor::with_shared_state(stability, shared);
            let poll_interval = std::time::Duration::from_secs(5);
            monitor_handle.start(poll_interval);

            let mut state = get_state().lock().unwrap();
            *state = AppState {
                cfg,
                cfg_path,
                settings,
                table_state: RefCell::new(TableState::default()),
                lang: "zh-CN".into(),
                consecutive_ctrl_c: 0,
                creating_project: false,
                creation_step: CreationStep::Name,
                new_project_name: String::new(),
                new_project_path: String::new(),
                monitor_receiver: Some(receiver),
                current_snapshot: None,
                stable_recommendation: None,
                ..Default::default()
            };
        }
    }

    fn draw(&mut self, drawer: &mut ComponentDrawer<'_, '_>) {
        let area = drawer.area;
        let state = get_state().lock().unwrap();
        draw_tui(drawer.frame, area, &state);
    }
}

// ============================================================================
// Command Handling
// ============================================================================

fn get_commands(state: &AppState) -> Vec<(String, String)> {
    let prefix = state.input_buf.to_lowercase();
    let cmds = vec![
        ("/setting safety_alpha=", "设置安全系数 | Set safety factor"),
        (
            "/setting learning_mode=",
            "开启学习模式 | Enable learning mode",
        ),
        ("/setting acc=", "设置准确率权重 | Set accuracy weight"),
        ("/setting lat=", "设置延迟权重 | Set latency weight"),
        ("/setting mem=", "设置显存权重 | Set memory weight"),
        ("/setting thr=", "设置吞吐权重 | Set throughput weight"),
        ("/setting energy=", "设置能耗权重 | Set energy weight"),
        ("/setting debug=", "开启debug输出 | Enable debug output"),
        ("/rollback", "回滚配置变更 | Rollback config changes"),
        ("/language", "切换语言 | Toggle language"),
        ("/new", "新建项目 | Create new project"),
        ("/create", "直接创建项目 | Create project directly"),
        ("/open", "打开已有项目 | Open existing project"),
        ("/plan", "基于需求生成计划 | Generate plan from requirement"),
        ("/execute", "自主执行计划 | Execute plan autonomously"),
        ("/checker", "检查计划与代码 | Validate plan and code"),
        ("/summary", "生成结构化摘要 | Generate structured summary"),
        ("/tester", "运行测试 | Run tests"),
        ("/runner", "运行训练 | Run training in Conda"),
        ("/teacher", "算法讲解 | Explain ML algorithm"),
        ("/status", "查看状态 | Show pipeline status"),
        ("/auto-complete", "全流程自动运行 | Auto-run full pipeline"),
        ("/expert", "专家模式 | Expert mode"),
        ("/guided", "引导模式 | Guided mode"),
        ("/save", "保存配置 | Save configuration"),
        ("/run", "运行训练 | Run training"),
        ("/apply", "应用推荐 | Apply recommendation"),
        ("/stop", "停止训练 | Stop training"),
        ("/edit-own", "编辑文件 | Edit files in editor"),
    ];
    cmds.iter()
        .filter(|(cmd, _)| cmd.to_lowercase().starts_with(&prefix))
        .map(|(cmd, hint)| (cmd.to_string(), hint.to_string()))
        .collect()
}

fn update_autocomplete(state: &mut AppState) {
    if !state.input_buf.starts_with('/') {
        state.show_autocomplete = false;
        state.autocomplete_items.clear();
        return;
    }
    state.autocomplete_items = get_commands(state);
    state.autocomplete_selected = 0;
    state.autocomplete_scroll = 0;
    state.show_autocomplete = !state.autocomplete_items.is_empty();
}

/// Handle input during interactive project creation flow
fn handle_project_creation_input(
    key: crossterm::event::KeyEvent,
    state: &mut AppState,
) -> Option<String> {
    use crossterm::event::{KeyCode, KeyModifiers};

    match key.code {
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            // Cancel project creation on double Ctrl+C
            state.consecutive_ctrl_c += 1;
            if state.consecutive_ctrl_c >= 2 {
                state.creating_project = false;
                state.input_mode = false;
                state.input_buf.clear();
            }
        }
        KeyCode::Char(c) => {
            state.input_buf.push(c);
            state.consecutive_ctrl_c = 0;
        }
        KeyCode::Backspace => {
            state.input_buf.pop();
            state.consecutive_ctrl_c = 0;
        }
        KeyCode::Enter => {
            state.consecutive_ctrl_c = 0;
            match state.creation_step {
                CreationStep::Name => {
                    // Store project name and ask for path
                    state.new_project_name = state.input_buf.clone();
                    state.input_buf.clear();
                    state.creation_step = CreationStep::Path;
                    state.message = "Enter project path (e.g., projects/):".into();
                }
                CreationStep::Path => {
                    // Store path and create project
                    state.new_project_path = state.input_buf.clone();
                    state.input_buf.clear();
                    state.creating_project = false;
                    state.creation_step = CreationStep::Name;
                    // Return special command to create project
                    return Some(format!(
                        "/create {} {}",
                        state.new_project_name, state.new_project_path
                    ));
                }
                CreationStep::Confirm => {
                    // Should not reach here
                    state.creating_project = false;
                    state.creation_step = CreationStep::Name;
                }
            }
        }
        KeyCode::Esc => {
            // Cancel project creation
            state.creating_project = false;
            state.creation_step = CreationStep::Name;
            state.new_project_name.clear();
            state.new_project_path.clear();
            state.input_buf.clear();
            state.input_mode = false;
            state.consecutive_ctrl_c = 0;
        }
        _ => {
            state.consecutive_ctrl_c = 0;
        }
    }
    None
}

fn handle_input_mode_key(key: crossterm::event::KeyEvent, state: &mut AppState) -> Option<String> {
    use crossterm::event::{KeyCode, KeyModifiers};

    // Handle project creation flow
    if state.creating_project {
        return handle_project_creation_input(key, state);
    }

    match key.code {
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            // Single Ctrl+C to exit from normal mode (no command input)
            crossterm::execute!(
                std::io::stdout(),
                crossterm::terminal::LeaveAlternateScreen,
                crossterm::event::DisableMouseCapture,
                crossterm::terminal::Clear(crossterm::terminal::ClearType::All)
            )
            .ok();
            crossterm::terminal::disable_raw_mode().ok();
            std::process::exit(0);
        }
        KeyCode::Char('v') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            // Ctrl+V to paste
            if let Some(text) = get_clipboard() {
                insert_text_at_cursor(&mut state.input_buf, &mut state.cursor_pos, &text);
            }
            state.consecutive_ctrl_c = 0;
            update_autocomplete(state);
        }
        KeyCode::Char(c) => {
            // Clear selection if any, then insert character
            if state.selection_start.is_some() {
                delete_selection(
                    &mut state.input_buf,
                    &mut state.cursor_pos,
                    state.selection_start,
                    state.selection_end,
                );
            }
            // Ensure cursor is on a valid character boundary
            let cursor_byte = state.cursor_pos.min(state.input_buf.len());
            let valid_cursor = state.input_buf[..cursor_byte]
                .chars()
                .map(|c| c.len_utf8())
                .sum::<usize>();
            state.cursor_pos = valid_cursor.min(state.input_buf.len());
            if state.cursor_pos <= state.input_buf.len() {
                state.input_buf.insert(state.cursor_pos, c);
                state.cursor_pos += c.len_utf8();
            }
            state.consecutive_ctrl_c = 0;
            update_autocomplete(state);
        }
        KeyCode::Backspace => {
            if state.selection_start.is_some() {
                delete_selection(
                    &mut state.input_buf,
                    &mut state.cursor_pos,
                    state.selection_start,
                    state.selection_end,
                );
            } else if state.cursor_pos > 0 {
                // Find the start of the character before cursor
                let char_boundaries: Vec<usize> =
                    state.input_buf.char_indices().map(|(i, _)| i).collect();
                if let Some(pos) = char_boundaries.iter().rposition(|&b| b < state.cursor_pos) {
                    state.cursor_pos = char_boundaries[pos];
                    state.input_buf.remove(state.cursor_pos);
                }
            }
            state.consecutive_ctrl_c = 0;
            update_autocomplete(state);
        }
        KeyCode::Delete => {
            if state.selection_start.is_some() {
                delete_selection(
                    &mut state.input_buf,
                    &mut state.cursor_pos,
                    state.selection_start,
                    state.selection_end,
                );
            } else if state.cursor_pos < state.input_buf.len() {
                // Remove the character at cursor (not after)
                let char_boundaries: Vec<usize> =
                    state.input_buf.char_indices().map(|(i, _)| i).collect();
                if let Some(pos) = char_boundaries.iter().position(|&b| b >= state.cursor_pos) {
                    let next_pos = char_boundaries
                        .get(pos + 1)
                        .copied()
                        .unwrap_or(state.input_buf.len());
                    state.input_buf.drain(pos..next_pos);
                }
            }
            state.consecutive_ctrl_c = 0;
            update_autocomplete(state);
        }
        KeyCode::Left => {
            // Move cursor left by one character (not byte)
            if state.cursor_pos > 0 {
                let char_boundaries: Vec<usize> =
                    state.input_buf.char_indices().map(|(i, _)| i).collect();
                if let Some(pos) = char_boundaries.iter().rposition(|&b| b < state.cursor_pos) {
                    state.cursor_pos = char_boundaries[pos];
                } else {
                    state.cursor_pos = 0;
                }
            }
            state.selection_start = None;
            state.selection_end = None;
            state.consecutive_ctrl_c = 0;
        }
        KeyCode::Right => {
            // Move cursor right by one character (not byte)
            if state.cursor_pos < state.input_buf.len() {
                let char_boundaries: Vec<usize> =
                    state.input_buf.char_indices().map(|(i, _)| i).collect();
                if let Some(pos) = char_boundaries.iter().position(|&b| b > state.cursor_pos) {
                    state.cursor_pos = char_boundaries[pos];
                } else {
                    state.cursor_pos = state.input_buf.len();
                }
            }
            state.selection_start = None;
            state.selection_end = None;
            state.consecutive_ctrl_c = 0;
        }
        KeyCode::Home => {
            state.cursor_pos = 0;
            state.selection_start = None;
            state.selection_end = None;
            state.consecutive_ctrl_c = 0;
        }
        KeyCode::End => {
            state.cursor_pos = state.input_buf.len();
            state.selection_start = None;
            state.selection_end = None;
            state.consecutive_ctrl_c = 0;
        }
        KeyCode::Enter => {
            state.consecutive_ctrl_c = 0;
            if state.show_autocomplete {
                if let Some((cmd, _)) = state
                    .autocomplete_items
                    .get(state.autocomplete_selected)
                    .cloned()
                {
                    state.input_buf = cmd;
                    state.cursor_pos = state.input_buf.len();
                    state.show_autocomplete = false;
                    state.selection_start = None;
                    state.selection_end = None;
                }
            } else {
                let cmd = state.input_buf.clone();
                // Clear input after sending
                state.input_buf.clear();
                state.cursor_pos = 0;
                state.selection_start = None;
                state.selection_end = None;
                if !cmd.is_empty() {
                    state.output_lines.push(format!("user: {};agent:", cmd));
                    state.output_scroll = usize::MAX;
                }
                return Some(cmd);
            }
        }
        KeyCode::Up => {
            state.consecutive_ctrl_c = 0;
            if state.show_autocomplete && !state.autocomplete_items.is_empty() {
                if state.autocomplete_selected > 0 {
                    state.autocomplete_selected -= 1;
                    // Scroll up if selected is above visible area
                    if state.autocomplete_scroll > 0
                        && state.autocomplete_selected < state.autocomplete_scroll
                    {
                        state.autocomplete_scroll -= 1;
                    }
                }
            }
        }
        KeyCode::Down => {
            state.consecutive_ctrl_c = 0;
            if state.show_autocomplete && !state.autocomplete_items.is_empty() {
                let list_height = state.autocomplete_items.len().min(4);
                if state.autocomplete_selected < state.autocomplete_items.len() - 1 {
                    state.autocomplete_selected += 1;
                    // Scroll down if selected is below visible area
                    if state.autocomplete_selected >= state.autocomplete_scroll + list_height {
                        state.autocomplete_scroll += 1;
                    }
                }
            }
        }
        KeyCode::Tab => {
            // Tab: complete autocomplete and close popup
            state.consecutive_ctrl_c = 0;
            if state.show_autocomplete {
                if let Some((cmd, _)) = state
                    .autocomplete_items
                    .get(state.autocomplete_selected)
                    .cloned()
                {
                    state.input_buf = cmd;
                    state.cursor_pos = state.input_buf.len();
                }
                state.show_autocomplete = false;
                state.selection_start = None;
                state.selection_end = None;
            }
        }
        KeyCode::Esc => {
            state.input_mode = false;
            state.input_buf.clear();
            state.cursor_pos = 0;
            state.show_autocomplete = false;
            state.selection_start = None;
            state.selection_end = None;
            state.autocomplete_scroll = 0;
            state.consecutive_ctrl_c = 0;
        }
        _ => {
            state.consecutive_ctrl_c = 0;
        }
    }
    None
}

// Get clipboard text
fn get_clipboard() -> Option<String> {
    arboard::Clipboard::new().ok()?.get_text().ok()
}

// Insert text at cursor position (handles UTF-8 properly)
fn insert_text_at_cursor(buf: &mut String, cursor: &mut usize, text: &str) {
    // Ensure cursor is on a valid character boundary
    let cursor_byte = (*cursor).min(buf.len());
    let valid_cursor = buf[..cursor_byte]
        .chars()
        .map(|c| c.len_utf8())
        .sum::<usize>();
    *cursor = valid_cursor.min(buf.len());

    // Insert each character at cursor position
    for c in text.chars() {
        if *cursor <= buf.len() {
            buf.insert(*cursor, c);
            *cursor += c.len_utf8();
        }
    }
}

// Delete selected text
fn delete_selection(
    buf: &mut String,
    cursor: &mut usize,
    start: Option<usize>,
    end: Option<usize>,
) {
    if let (Some(s), Some(e)) = (start, end) {
        let (s, e) = if s < e { (s, e) } else { (e, s) };
        if e <= buf.len() {
            buf.drain(s..e);
            *cursor = s;
        }
    }
}

fn handle_normal_mode_key(key: crossterm::event::KeyEvent, state: &mut AppState) -> Option<String> {
    use crossterm::event::KeyCode;
    match key.code {
        KeyCode::Char('/') => {
            state.input_mode = true;
            state.input_buf.clear();
            state.show_autocomplete = false;
            state.autocomplete_items = get_commands(state);
            state.autocomplete_selected = 0;
            state.autocomplete_scroll = 0;
            state.consecutive_ctrl_c = 0;
        }
        KeyCode::Char('a') | KeyCode::Char('A') => {
            state.consecutive_ctrl_c = 0;
            if let Some(ref reco) = state.recommendation.clone() {
                apply_reco(reco, state);
                state.output_lines.push("Applied recommendation".into());
            }
        }
        _ => {
            state.consecutive_ctrl_c = 0;
        }
    }
    None
}

fn exec_command(cmd: &str, state: Arc<std::sync::Mutex<AppState>>) {
    let parts: Vec<&str> = cmd.split_whitespace().collect();
    if parts.is_empty() {
        return;
    }

    // If command doesn't start with /, treat it as a user request - invoke planner agent
    if !cmd.starts_with('/') {
        // Clone state Arc before locking, so we can move it later
        let state_clone = state.clone();
        let mut s = state.lock().unwrap();
        s.output_lines.push(format!("> {}", cmd));
        s.output_lines.push("Invoking planner agent...".into());
        s.output_scroll = usize::MAX;

        // Spawn planner agent in background using rig-core
        let requirement = cmd.to_string();
        let project_root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let runner = llm_agent::AgentRunner::new(state_clone);
        runner.spawn_planner(requirement, project_root);
        return;
    }

    match parts[0] {
        "/setting" => {
            let mut s = state.lock().unwrap();
            for kv in &parts[1..] {
                if let Some((k, v)) = kv.split_once('=') {
                    match k {
                        "safety_alpha" => {
                            if let Ok(x) = v.parse() {
                                s.settings.safety_alpha = x;
                            }
                        }
                        "learning_mode" => {
                            if let Ok(x) = v.parse() {
                                s.settings.learning_mode = x;
                            }
                        }
                        "acc" => {
                            if let Ok(x) = v.parse() {
                                s.settings.weights.acc = x;
                            }
                        }
                        "lat" => {
                            if let Ok(x) = v.parse() {
                                s.settings.weights.lat = x;
                            }
                        }
                        "mem" => {
                            if let Ok(x) = v.parse() {
                                s.settings.weights.mem = x;
                            }
                        }
                        "thr" => {
                            if let Ok(x) = v.parse() {
                                s.settings.weights.thr = x;
                            }
                        }
                        "energy" => {
                            if let Ok(x) = v.parse() {
                                s.settings.weights.energy = x;
                            }
                        }
                        "debug" => {
                            if let Ok(x) = v.parse() {
                                s.settings.debug = x;
                            }
                        }
                        _ => {}
                    }
                }
            }
            let _ = settings::save(&PathBuf::from("settings.yaml"), &s.settings);
            s.output_lines.push("Settings updated".into());
        }
        "/language" => {
            let mut s = state.lock().unwrap();
            let new_lang: String = if s.lang == "zh-CN" {
                "en".into()
            } else {
                "zh-CN".into()
            };
            s.lang = new_lang.clone();
            s.output_lines.push(format!("Language: {}", new_lang));
        }
        "/expert" => {
            let mut s = state.lock().unwrap();
            s.expert_mode = true;
            s.output_lines
                .push("Expert mode: use /edit-own to edit files".into());
        }
        "/guided" => {
            let mut s = state.lock().unwrap();
            s.expert_mode = false;
            s.output_lines
                .push("Guided mode: use /setting to modify parameters".into());
        }
        "/create" => {
            let mut s = state.lock().unwrap();
            if parts.len() < 3 {
                s.output_lines.push("Use: /create <name> <path>".into());
                s.output_lines
                    .push("Example: /create my_model projects/".into());
            } else {
                let name = parts[1];
                let path_str = parts[2];
                let base_path = PathBuf::from(path_str);

                match project_creator::create_project(name, &base_path) {
                    Ok(paths) => {
                        s.output_lines
                            .push(format!("Created project at: {}", paths.root.display()));
                        s.output_lines
                            .push(format!("Plan saved to: {}", paths.plan.display()));
                    }
                    Err(e) => {
                        s.output_lines
                            .push(format!("Failed to create project: {}", e));
                    }
                }
            }
        }
        "/new" => {
            let mut s = state.lock().unwrap();
            // Start interactive project creation flow
            s.creating_project = true;
            s.creation_step = CreationStep::Name;
            s.new_project_name.clear();
            s.new_project_path.clear();
            s.input_mode = true;
            s.input_buf.clear();
            s.output_lines.push("Enter project name:".into());
        }
        "/open" => {
            let mut s = state.lock().unwrap();
            if parts.len() < 2 {
                s.output_lines.push("Use: /open <project_path>".into());
                s.output_lines.push("Example: /open projects/mnist".into());
            } else {
                let project_path = PathBuf::from(parts[1]);
                let config_path = project_path.join("config.yaml");
                if !config_path.exists() {
                    s.output_lines
                        .push(format!("Config not found: {}", config_path.display()));
                } else {
                    match config::load_config(&config_path) {
                        Ok(cfg) => {
                            s.cfg = cfg;
                            s.cfg_path = config_path;
                            s.output_lines
                                .push(format!("Opened project: {}", project_path.display()));
                        }
                        Err(e) => {
                            s.output_lines.push(format!("Failed to load config: {}", e));
                        }
                    }
                }
            }
        }
        "/plan" => {
            let requirement = if cmd.len() > 5 { cmd[5..].trim() } else { "" };
            if requirement.is_empty() {
                let mut s = state.lock().unwrap();
                s.output_lines.push("Usage: /plan <requirement>".into());
                s.output_lines
                    .push("Example: /plan implement batch size auto-tuning".into());
            } else {
                let out_width = 80usize.saturating_sub(6);
                let log_lines = read_logger_log(out_width);
                let state_clone = state.clone();
                let mut s = state.lock().unwrap();
                s.output_lines.push(format!("user: /plan {}", requirement));
                s.active_agent = Some("planner".into());
                s.agent_task = Some("Running...".into());
                s.output_lines.push("agent: [planner] running...".into());
                if !log_lines.is_empty() {
                    s.output_lines.push("--- logger.log ---".into());
                    for line in log_lines {
                        s.output_lines.push(line);
                    }
                    s.output_lines.push("---".into());
                }
                s.output_scroll = usize::MAX;
                let req = requirement.to_string();
                let project_root = s
                    .cfg_path
                    .parent()
                    .map(|p| p.to_path_buf())
                    .unwrap_or_else(|| PathBuf::from("."));
                drop(s);
                let runner = llm_agent::AgentRunner::new(state_clone);
                runner.spawn_planner(req, project_root);
            }
        }
        "/edit-own" => {
            let mut s = state.lock().unwrap();
            if s.expert_mode {
                let train_name = s
                    .cfg
                    .train_script
                    .clone()
                    .unwrap_or_else(|| "train.py".into());
                let train_path = std::env::current_dir()
                    .map(|d| d.join(train_name))
                    .unwrap_or_else(|_| PathBuf::from("train.py"));
                let editor = std::env::var("EDITOR").unwrap_or_else(|_| "vi".into());
                if let Err(e) = std::process::Command::new(&editor)
                    .arg(&train_path)
                    .status()
                {
                    s.output_lines.push(format!("Failed to open editor: {}", e));
                } else {
                    s.output_lines
                        .push(format!("Opened {} in {}", train_path.display(), editor));
                }
            } else {
                s.output_lines
                    .push("Switch to expert mode first: /expert".into());
            }
        }
        "/save" => {
            let mut s = state.lock().unwrap();
            let abs = std::env::current_dir()
                .unwrap_or_default()
                .join(&s.cfg_path);
            let old_cfg = journal::snapshot_config(&abs);
            if let Err(e) = config::save_config(&abs, &s.cfg) {
                s.output_lines.push(format!("Save failed: {}", e));
            } else {
                let _ = journal::append(
                    "journal.jsonl",
                    "modify_config",
                    &serde_json::json!({ "file": abs }),
                    Some(serde_json::json!({ "old_config": old_cfg })),
                );
                s.output_lines.push("Config saved".into());
            }
        }
        "/run" => {
            let project_root = {
                let mut s = state.lock().unwrap();
                if s.running {
                    s.output_lines
                        .push("agent: Training already running. Use /stop first.".into());
                    s.output_scroll = usize::MAX;
                    return;
                }
                s.cfg_path
                    .parent()
                    .map(|p| p.to_path_buf())
                    .unwrap_or_else(|| PathBuf::from("."))
            };
            let mut s = state.lock().unwrap();
            s.active_agent = Some("runner".into());
            s.agent_task = Some("Training".into());
            s.output_lines
                .push("user: /run;agent: Starting training...".into());
            s.output_scroll = usize::MAX;
            drop(s);
            let agent = llm_agent::CoLoMoAgent::new(project_root);
            agent.run_runner_blocking();
            let mut s = state.lock().unwrap();
            s.agent_task = Some("Running".into());
            s.active_agent = None;
        }
        "/apply" => {
            let mut s = state.lock().unwrap();
            if let Some(ref reco) = s.recommendation.clone() {
                apply_reco(reco, &mut s);
                s.output_lines.push("Applied recommendation".into());
            } else {
                s.output_lines.push("No recommendation available".into());
            }
        }
        "/stop" => {
            let mut s = state.lock().unwrap();
            s.running = false;
            s.output_lines.push("Training stopped".into());
        }
        "/execute" => {
            let project_root = {
                let s = state.lock().unwrap();
                s.cfg_path
                    .parent()
                    .map(|p| p.to_path_buf())
                    .unwrap_or_else(|| PathBuf::from("."))
            };
            let plan_path = project_root.join("plan.md");
            if !plan_path.exists() {
                let mut s = state.lock().unwrap();
                s.output_lines
                    .push("agent: No plan.md found. Run /plan first.".into());
                s.output_scroll = usize::MAX;
            } else {
                // Estimate output panel width (50% of terminal, minus borders/padding)
                let out_width = 80usize.saturating_sub(6);
                let log_lines = read_logger_log(out_width);
                let mut s = state.lock().unwrap();
                // Show "running..." immediately
                s.active_agent = Some("executor".into());
                s.agent_task = Some("Running...".into());
                s.agent_progress = Some("0/1".into());
                s.output_lines.push("user: /execute".into());
                s.output_lines.push("agent: [executor] running...".into());
                if !log_lines.is_empty() {
                    s.output_lines.push("--- logger.log ---".into());
                    for line in log_lines {
                        s.output_lines.push(line);
                    }
                    s.output_lines.push("---".into());
                }
                s.output_scroll = usize::MAX;
                drop(s);
                let state_clone = state.clone();
                tokio::spawn(async move {
                    let agent = llm_agent::CoLoMoAgent::new(project_root.clone());
                    match agent.run_execute().await {
                        Ok(_) => {
                            // Read entry file from config.yaml
                            let config_path = project_root.join("config.yaml");
                            let entry =
                                if let Ok(cfg_content) = std::fs::read_to_string(&config_path) {
                                    if let Ok(cfg) =
                                        serde_yaml::from_str::<crate::config::Config>(&cfg_content)
                                    {
                                        cfg.train_script.unwrap_or_else(|| "train.py".into())
                                    } else {
                                        "train.py".into()
                                    }
                                } else {
                                    "train.py".into()
                                };
                            // Read logger.log and append to output
                            let out_width = 80usize.saturating_sub(6);
                            let log_lines = read_logger_log(out_width);
                            let mut s = state_clone.lock().unwrap();
                            s.output_lines.push("agent: [executor] completed".into());
                            if !log_lines.is_empty() {
                                s.output_lines.push("--- logger.log ---".into());
                                for line in log_lines {
                                    s.output_lines.push(line);
                                }
                                s.output_lines.push("---".into());
                            }
                            s.output_lines.push(format!(
                                "agent: Execute complete. Entry: {} (in {})",
                                entry,
                                project_root.display()
                            ));
                            s.output_lines
                                .push("agent: Run /runner to train, or /tester to verify.".into());
                            s.output_scroll = usize::MAX;
                            s.agent_task = Some("Execute complete".into());
                            s.agent_progress = None;
                            s.active_agent = None;
                        }
                        Err(e) => {
                            let mut s = state_clone.lock().unwrap();
                            s.output_lines
                                .push("agent: [executor] completed with error".into());
                            let out_width = 80usize.saturating_sub(6);
                            let log_lines = read_logger_log(out_width);
                            if !log_lines.is_empty() {
                                s.output_lines.push("--- logger.log ---".into());
                                for line in log_lines {
                                    s.output_lines.push(line);
                                }
                                s.output_lines.push("---".into());
                            }
                            s.output_lines.push(format!("agent: Execute error: {}", e));
                            s.output_scroll = usize::MAX;
                            s.agent_task = Some("Execute failed".into());
                            s.agent_progress = None;
                            s.active_agent = None;
                        }
                    }
                });
            }
        }
        "/checker" => {
            let project_root = {
                let s = state.lock().unwrap();
                s.cfg_path
                    .parent()
                    .map(|p| p.to_path_buf())
                    .unwrap_or_else(|| PathBuf::from("."))
            };
            let plan_path = project_root.join("plan.md");
            if !plan_path.exists() {
                let mut s = state.lock().unwrap();
                s.output_lines
                    .push("agent: No plan.md found. Run /plan first.".into());
                s.output_scroll = usize::MAX;
            } else {
                let out_width = 80usize.saturating_sub(6);
                let log_lines = read_logger_log(out_width);
                let mut s = state.lock().unwrap();
                s.active_agent = Some("checker".into());
                s.agent_task = Some("Running...".into());
                s.output_lines.push("user: /checker".into());
                s.output_lines.push("agent: [checker] running...".into());
                if !log_lines.is_empty() {
                    s.output_lines.push("--- logger.log ---".into());
                    s.output_lines.extend(log_lines);
                    s.output_lines.push("---".into());
                }
                s.output_scroll = usize::MAX;
                drop(s);
                let state_clone = state.clone();
                tokio::spawn(async move {
                    let agent = llm_agent::CoLoMoAgent::new(project_root);
                    match agent.run_checker().await {
                        Ok(_) => {
                            let out_width = 80usize.saturating_sub(6);
                            let log_lines = read_logger_log(out_width);
                            let mut s = state_clone.lock().unwrap();
                            s.output_lines.push("agent: [checker] completed".into());
                            if !log_lines.is_empty() {
                                s.output_lines.push("--- logger.log ---".into());
                                s.output_lines.extend(log_lines);
                                s.output_lines.push("---".into());
                            }
                            s.output_lines.push("agent: Check complete".into());
                            s.output_scroll = usize::MAX;
                            s.agent_task = Some("Check complete".into());
                            s.agent_progress = None;
                            s.active_agent = None;
                        }
                        Err(e) => {
                            let out_width = 80usize.saturating_sub(6);
                            let log_lines = read_logger_log(out_width);
                            let mut s = state_clone.lock().unwrap();
                            s.output_lines.push("agent: [checker] completed with error".into());
                            if !log_lines.is_empty() {
                                s.output_lines.push("--- logger.log ---".into());
                                s.output_lines.extend(log_lines);
                                s.output_lines.push("---".into());
                            }
                            s.output_lines.push(format!("agent: Check error: {}", e));
                            s.output_scroll = usize::MAX;
                            s.agent_task = Some("Check failed".into());
                            s.active_agent = None;
                        }
                    }
                });
            }
        }
        "/summary" => {
            let project_root = {
                let s = state.lock().unwrap();
                s.cfg_path
                    .parent()
                    .map(|p| p.to_path_buf())
                    .unwrap_or_else(|| PathBuf::from("."))
            };
            let plan_path = project_root.join("plan.md");
            if !plan_path.exists() {
                let mut s = state.lock().unwrap();
                s.output_lines
                    .push("agent: No plan.md found. Run /plan first.".into());
                s.output_scroll = usize::MAX;
            } else {
                let out_width = 80usize.saturating_sub(6);
                let log_lines = read_logger_log(out_width);
                let mut s = state.lock().unwrap();
                s.active_agent = Some("summary".into());
                s.agent_task = Some("Running...".into());
                s.output_lines.push("user: /summary".into());
                s.output_lines.push("agent: [summary] running...".into());
                if !log_lines.is_empty() {
                    s.output_lines.push("--- logger.log ---".into());
                    s.output_lines.extend(log_lines);
                    s.output_lines.push("---".into());
                }
                s.output_scroll = usize::MAX;
                drop(s);
                let state_clone = state.clone();
                tokio::spawn(async move {
                    let agent = llm_agent::CoLoMoAgent::new(project_root);
                    match agent.run_summary().await {
                        Ok(summary) => {
                            let out_width = 80usize.saturating_sub(6);
                            let log_lines = read_logger_log(out_width);
                            let mut s = state_clone.lock().unwrap();
                            s.output_lines.push("agent: [summary] completed".into());
                            if !log_lines.is_empty() {
                                s.output_lines.push("--- logger.log ---".into());
                                s.output_lines.extend(log_lines);
                                s.output_lines.push("---".into());
                            }
                            s.output_lines.push(format!(
                                "agent: Summary: {}",
                                &summary[..summary.len().min(200)]
                            ));
                            s.agent_last_summary = Some(summary);
                            s.output_scroll = usize::MAX;
                            s.agent_task = Some("Summary complete".into());
                            s.active_agent = None;
                        }
                        Err(e) => {
                            let out_width = 80usize.saturating_sub(6);
                            let log_lines = read_logger_log(out_width);
                            let mut s = state_clone.lock().unwrap();
                            s.output_lines.push("agent: [summary] completed with error".into());
                            if !log_lines.is_empty() {
                                s.output_lines.push("--- logger.log ---".into());
                                s.output_lines.extend(log_lines);
                                s.output_lines.push("---".into());
                            }
                            s.output_lines
                                .push(format!("agent: Summary error: {}", e));
                            s.output_scroll = usize::MAX;
                            s.agent_task = Some("Summary failed".into());
                            s.active_agent = None;
                        }
                    }
                });
            }
        }
        "/tester" => {
            let project_root = {
                let s = state.lock().unwrap();
                s.cfg_path
                    .parent()
                    .map(|p| p.to_path_buf())
                    .unwrap_or_else(|| PathBuf::from("."))
            };
            let config_path = project_root.join("config.yaml");
            if !config_path.exists() {
                let mut s = state.lock().unwrap();
                s.output_lines.push("agent: No config.yaml found.".into());
                s.output_scroll = usize::MAX;
            } else {
                let out_width = 80usize.saturating_sub(6);
                let log_lines = read_logger_log(out_width);
                let mut s = state.lock().unwrap();
                s.active_agent = Some("tester".into());
                s.agent_task = Some("Running...".into());
                s.output_lines.push("user: /tester".into());
                s.output_lines.push("agent: [tester] running...".into());
                if !log_lines.is_empty() {
                    s.output_lines.push("--- logger.log ---".into());
                    s.output_lines.extend(log_lines);
                    s.output_lines.push("---".into());
                }
                s.output_scroll = usize::MAX;
                drop(s);
                let state_clone = state.clone();
                tokio::spawn(async move {
                    let agent = llm_agent::CoLoMoAgent::new(project_root);
                    match agent.run_tester().await {
                        Ok(summary) => {
                            let out_width = 80usize.saturating_sub(6);
                            let log_lines = read_logger_log(out_width);
                            let mut s = state_clone.lock().unwrap();
                            s.output_lines.push("agent: [tester] completed".into());
                            if !log_lines.is_empty() {
                                s.output_lines.push("--- logger.log ---".into());
                                s.output_lines.extend(log_lines);
                                s.output_lines.push("---".into());
                            }
                            s.output_lines.push(format!(
                                "agent: Test: {}",
                                &summary[..summary.len().min(200)]
                            ));
                            s.agent_last_summary = Some(summary);
                            s.output_scroll = usize::MAX;
                            s.agent_task = Some("Test complete".into());
                            s.active_agent = None;
                        }
                        Err(e) => {
                            let out_width = 80usize.saturating_sub(6);
                            let log_lines = read_logger_log(out_width);
                            let mut s = state_clone.lock().unwrap();
                            s.output_lines.push("agent: [tester] completed with error".into());
                            if !log_lines.is_empty() {
                                s.output_lines.push("--- logger.log ---".into());
                                s.output_lines.extend(log_lines);
                                s.output_lines.push("---".into());
                            }
                            s.output_lines.push(format!("agent: Test error: {}", e));
                            s.output_scroll = usize::MAX;
                            s.agent_task = Some("Test failed".into());
                            s.active_agent = None;
                        }
                    }
                });
            }
        }
        "/runner" => {
            let project_root = {
                let mut s = state.lock().unwrap();
                if s.running {
                    s.output_lines
                        .push("agent: Training already running. Use /stop first.".into());
                    s.output_scroll = usize::MAX;
                    return;
                }
                s.cfg_path
                    .parent()
                    .map(|p| p.to_path_buf())
                    .unwrap_or_else(|| PathBuf::from("."))
            };
            let out_width = 80usize.saturating_sub(6);
            let log_lines = read_logger_log(out_width);
            let mut s = state.lock().unwrap();
            s.active_agent = Some("runner".into());
            s.agent_task = Some("Running...".into());
            s.output_lines.push("user: /runner".into());
            s.output_lines.push("agent: [runner] running...".into());
            if !log_lines.is_empty() {
                s.output_lines.push("--- logger.log ---".into());
                s.output_lines.extend(log_lines);
                s.output_lines.push("---".into());
            }
            s.output_scroll = usize::MAX;
            drop(s);
            let agent = llm_agent::CoLoMoAgent::new(project_root);
            agent.run_runner_blocking();
        }
        "/status" => {
            let project_root = {
                let s = state.lock().unwrap();
                s.cfg_path
                    .parent()
                    .map(|p| p.to_path_buf())
                    .unwrap_or_else(|| PathBuf::from("."))
            };
            let plan_path = project_root.join("plan.md");
            let running = {
                let s = state.lock().unwrap();
                s.running
            };
            let last_summary = {
                let s = state.lock().unwrap();
                s.agent_last_summary.clone()
            };
            let status = format!(
                "agent: status: plan_exists={}; running={}; summary={}",
                plan_path.exists(),
                running,
                last_summary
                    .as_ref()
                    .map(|s| &s[..s.len().min(50)])
                    .unwrap_or("none")
            );
            let mut s = state.lock().unwrap();
            s.output_lines.push(status);
            s.output_scroll = usize::MAX;
        }
        "/auto-complete" => {
            let requirement = if cmd.len() > 14 { cmd[14..].trim() } else { "" };
            if requirement.is_empty() {
                let mut s = state.lock().unwrap();
                s.output_lines
                    .push("Usage: /auto-complete <requirement>".into());
                s.output_lines
                    .push("Example: /auto-complete PyTorch CNN for MNIST".into());
                s.output_scroll = usize::MAX;
            } else {
                let out_width = 80usize.saturating_sub(6);
                let log_lines = read_logger_log(out_width);
                let mut s = state.lock().unwrap();
                s.output_lines.push(format!("user: /auto-complete {} ---分割线", requirement));
                s.output_lines.push("agent: [autocomplete] running...".into());
                if !log_lines.is_empty() {
                    s.output_lines.push("--- logger.log ---".into());
                    s.output_lines.extend(log_lines);
                    s.output_lines.push("---".into());
                }
                s.output_scroll = usize::MAX;
                drop(s);
                let state_clone = state.clone();
                let runner = llm_agent::AgentRunner::new(state_clone);
                runner.spawn_auto_complete(requirement.to_string());
            }
        }
        "/teacher" => {
            let topic = if cmd.len() > 8 { cmd[8..].trim() } else { "" };
            if topic.is_empty() {
                let mut s = state.lock().unwrap();
                s.output_lines
                    .push("Usage: /teacher <algorithm/topic>".into());
                s.output_lines
                    .push("Example: /teacher CNN, /teacher Adam optimizer".into());
                s.output_scroll = usize::MAX;
            } else {
                let out_width = 80usize.saturating_sub(6);
                let log_lines = read_logger_log(out_width);
                let mut s = state.lock().unwrap();
                s.active_agent = Some("teacher".into());
                s.agent_task = Some("Running...".into());
                s.output_lines.push(format!("user: /teacher {}", topic));
                s.output_lines.push("agent: [teacher] running...".into());
                if !log_lines.is_empty() {
                    s.output_lines.push("--- logger.log ---".into());
                    s.output_lines.extend(log_lines);
                    s.output_lines.push("---".into());
                }
                s.output_scroll = usize::MAX;
                drop(s);
                let state_clone = state.clone();
                let topic_str = topic.to_string();
                tokio::spawn(async move {
                    let project_root = {
                        let s = state_clone.lock().unwrap();
                        s.cfg_path
                            .parent()
                            .map(|p| p.to_path_buf())
                            .unwrap_or_else(|| PathBuf::from("."))
                    };
                    let agent = llm_agent::CoLoMoAgent::new(project_root);
                    match agent.run_teacher(&topic_str).await {
                        Ok(output) => {
                            let out_width = 80usize.saturating_sub(6);
                            let log_lines = read_logger_log(out_width);
                            let mut s = state_clone.lock().unwrap();
                            s.output_lines.push("agent: [teacher] completed".into());
                            if !log_lines.is_empty() {
                                s.output_lines.push("--- logger.log ---".into());
                                s.output_lines.extend(log_lines);
                                s.output_lines.push("---".into());
                            }
                            s.output_lines
                                .push(format!("agent: {}", &output[..output.len().min(300)]));
                            s.output_scroll = usize::MAX;
                            s.agent_task = Some("Explanation complete".into());
                            s.active_agent = None;
                        }
                        Err(e) => {
                            let out_width = 80usize.saturating_sub(6);
                            let log_lines = read_logger_log(out_width);
                            let mut s = state_clone.lock().unwrap();
                            s.output_lines.push("agent: [teacher] completed with error".into());
                            if !log_lines.is_empty() {
                                s.output_lines.push("--- logger.log ---".into());
                                s.output_lines.extend(log_lines);
                                s.output_lines.push("---".into());
                            }
                            s.output_lines
                                .push(format!("agent: Teacher error: {}", e));
                            s.output_scroll = usize::MAX;
                            s.agent_task = Some("Teacher failed".into());
                            s.active_agent = None;
                        }
                    }
                });
            }
        }
        "/rollback" => {
            let mut s = state.lock().unwrap();
            s.output_lines
                .push("Rollback pending implementation".into());
        }
        _ => {
            let mut s = state.lock().unwrap();
            s.output_lines
                .push(format!("Unknown command: {}", parts[0]));
        }
    }
}

fn apply_reco(reco: &advisor::Recommendation, state: &mut AppState) {
    let abs = std::env::current_dir()
        .unwrap_or_default()
        .join(&state.cfg_path);
    let old_cfg = journal::snapshot_config(&abs);
    let mut changed = false;
    if let Some(b) = reco.new_batch_size {
        state.cfg.batch_size = Some(b);
        changed = true;
    }
    if let Some(lr) = reco.new_learning_rate {
        state.cfg.learning_rate = Some(lr);
        changed = true;
    }
    if let Some(ref opt) = reco.new_optimizer {
        state.cfg.optimizer = Some(opt.clone());
        changed = true;
    }
    if changed {
        let _ = config::save_config(&abs, &state.cfg);
    }
    let _ = journal::append(
        "journal.jsonl",
        "apply_recommendation",
        &serde_json::json!({
            "file": abs,
            "applied": changed,
            "grad_accum_steps": reco.grad_accum_steps,
            "rationale": reco.rationale,
        }),
        Some(serde_json::json!({ "old_config": old_cfg })),
    );
}

// ============================================================================
// Log File Helpers
// ============================================================================

/// Strip leading [digits] timestamp and space prefix from a log line.
fn strip_timestamp_prefix(line: &str) -> &str {
    let mut s = line;
    // Keep stripping leading '[' / digits / ']' / spaces iteratively
    loop {
        let s2 = s
            .strip_prefix('[')
            .and_then(|t| t.strip_prefix(|c: char| c.is_ascii_digit()))
            .and_then(|t| {
                while t.starts_with(|c: char| c.is_ascii_digit()) {
                    if let Some(n) = t.strip_prefix(|c: char| c.is_ascii_digit()) {
                        s = t;
                    } else {
                        break;
                    }
                }
                Some(t)
            })
            .and_then(|t| t.strip_prefix(']'))
            .and_then(|t| t.strip_prefix(' '));
        match s2 {
            Some(next) => {
                if next == s {
                    break;
                }
                s = next;
            }
            None => break,
        }
    }
    // Also strip a leading `[` that wasn't part of a [digits] bracket
    if let Some(without_bracket) = s.strip_prefix('[') {
        s = without_bracket;
    }
    s.trim_start()
}

/// Read logger.log and return lines wrapped to `width` columns (char-based).
fn read_logger_log(width: usize) -> Vec<String> {
    let log_path = {
        let s = get_state().lock().unwrap();
        s.cfg_path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."))
            .join("logs")
            .join("logger.log")
    };
    let content = std::fs::read_to_string(&log_path).unwrap_or_default();

    let mut result = Vec::new();
    for line in content.lines() {
        let line = strip_timestamp_prefix(line);

        if line.trim().is_empty() {
            result.push(String::new());
            continue;
        }

        // Word-wrap at `width` chars (char-based to handle UTF-8 safely)
        let chars: Vec<char> = line.chars().collect();
        let mut pos = 0;
        while pos < chars.len() {
            let end = (pos + width).min(chars.len());
            if end - pos < width || end == chars.len() {
                // Last chunk or short enough, emit as-is
                result.push(chars[pos..].iter().collect());
                break;
            }
            // Find last space in [pos, end)
            let mut split = end;
            for i in (pos..end).rev() {
                if chars[i] == ' ' {
                    split = i;
                    break;
                }
            }
            if split == end {
                // No space found, hard wrap
                result.push(chars[pos..end].iter().collect());
                pos = end;
            } else {
                result.push(chars[pos..split].iter().collect());
                pos = split + 1; // skip the space
            }
        }
    }
    result
}

/// Push "running..." indicator to output during agent execution.
fn push_running(state: &Arc<Mutex<AppState>>, agent_name: &str) {
    let mut s = state.lock().unwrap();
    s.active_agent = Some(agent_name.into());
    s.agent_task = Some("Running...".into());
    s.output_lines
        .push(format!("agent: [{}] running...", agent_name));
    s.output_scroll = usize::MAX;
}

fn draw_tui(f: &mut Frame<'_>, area: Rect, state: &AppState) {
    let lbl = labels(&state.lang);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
        .split(area);

    let main_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(chunks[0]);

    if state.expert_mode {
        let content = vec![
            Line::from("Project: demo"),
            Line::from(""),
            Line::from("  train.py"),
            Line::from("  config.yaml"),
            Line::from("  requirements.txt"),
            Line::from("  logs/"),
        ];
        let para = Paragraph::new(content).block(
            Block::default()
                .title("Expert Mode - Project Files")
                .borders(Borders::ALL),
        );
        f.render_widget(para, main_chunks[0]);
    } else {
        // Guided config: table (top) + [sys monitor | stable reco] (bottom)
        let guided_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
            .split(main_chunks[0]);

        draw_config_table(f, guided_chunks[0], state, &lbl);

        // Bottom: System Monitor (left) + Stable Recommendation (right)
        let guided_bottom = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(guided_chunks[1]);

        // System Monitor (left)
        let sys_content = if let Some(ref snap) = state.current_snapshot {
            let mut lines = Vec::new();
            lines.push(Line::from(vec![Span::styled(
                "System Monitor",
                Style::default().add_modifier(Modifier::BOLD),
            )]));
            if let (Some(total), Some(used)) = (snap.gpu.total_mb, snap.gpu.used_mb) {
                lines.push(Line::from(format!(
                    "  GPU: {}/{} MB ({:.1}% used)",
                    used,
                    total,
                    (used as f64 / total as f64 * 100.0)
                )));
            } else if let Some(util) = snap.gpu.utilization_pct {
                lines.push(Line::from(format!("  GPU: {}%", util)));
            }
            lines.push(Line::from(format!(
                "  RAM: {}/{} MB ({:.1}% used)",
                snap.used_memory_mb,
                snap.system_memory_mb,
                (snap.used_memory_mb as f64 / snap.system_memory_mb as f64 * 100.0)
            )));
            lines.push(Line::from(format!("  CPU: {:.1}%", snap.cpu_usage_pct)));
            lines
        } else {
            vec![Line::from("(no system info)")]
        };
        let sys_block = Paragraph::new(sys_content)
            .block(
                Block::default()
                    .title("System Monitor")
                    .borders(Borders::ALL),
            )
            .wrap(Wrap { trim: true });
        f.render_widget(sys_block, guided_bottom[0]);

        // Stable Recommendation (right)
        let reco_content = if let Some(ref reco) = state.stable_recommendation {
            let mut lines = Vec::new();
            lines.push(Line::from(vec![Span::styled(
                "Stable Recommendation",
                Style::default()
                    .add_modifier(Modifier::BOLD)
                    .fg(Color::Cyan),
            )]));
            lines.push(Line::from(format!(
                "  batch: {} (was {})",
                reco.recommended_batch, reco.current_batch
            )));
            lines.push(Line::from(format!("  lr: {:.6}", reco.recommended_lr)));
            lines.push(Line::from(format!(
                "  stability: {:.0}%",
                reco.stability_score * 100.0
            )));
            if let Some(reason) = reco.reason.lines().next() {
                lines.push(Line::from(format!("  {}", reason)));
            }
            lines
        } else {
            vec![Line::from("(no recommendation)")]
        };
        let reco_block = Paragraph::new(reco_content)
            .block(
                Block::default()
                    .title("Recommendation")
                    .borders(Borders::ALL),
            )
            .wrap(Wrap { trim: true });
        f.render_widget(reco_block, guided_bottom[1]);
    }

    // Right panel: top-right = Plan/Todo, middle = Logs, bottom = Summary
    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(30), // Top-right: Plan/Todo
            Constraint::Percentage(35), // Middle: Recent Logs
            Constraint::Percentage(35), // Bottom: Summary
        ])
        .split(main_chunks[1]);

    // Top-right: Plan/Todo display
    let plan_lines: Vec<Line> = if state.top_right_lines.is_empty() {
        vec![Line::from(vec![Span::styled(
            "(no active plan)",
            Style::default().fg(Color::DarkGray),
        )])]
    } else {
        state
            .top_right_lines
            .iter()
            .map(|s| {
                if s.starts_with("[DONE]") {
                    Line::from(vec![Span::styled(s, Style::default().fg(Color::Green))])
                } else if s.starts_with("[...]") || s.starts_with("[>>]") {
                    Line::from(vec![Span::styled(s, Style::default().fg(Color::Cyan))])
                } else if s.starts_with("[ ]") {
                    Line::from(vec![Span::styled(s, Style::default().fg(Color::Yellow))])
                } else {
                    Line::from(vec![Span::raw(s)])
                }
            })
            .collect()
    };
    let plan_block = Paragraph::new(plan_lines)
        .block(Block::default().title("Plan / Todo").borders(Borders::ALL))
        .wrap(Wrap { trim: true });
    f.render_widget(plan_block, right_chunks[0]);

    let logs_content = if state.recent_logs.is_empty() {
        vec![Line::from(lbl.no_logs.as_str())]
    } else {
        state
            .recent_logs
            .iter()
            .map(|s| Line::from(s.as_str()))
            .collect()
    };
    let logs_block = Paragraph::new(logs_content)
        .block(
            Block::default()
                .title(lbl.recent_logs.as_str())
                .borders(Borders::ALL),
        )
        .wrap(Wrap { trim: true });
    f.render_widget(logs_block, right_chunks[1]);

    let summary_content: Vec<Line> = {
        let mut lines = Vec::new();

        // Fall back to summary_lines if no monitor data
        if state.current_snapshot.is_none() && !state.summary_lines.is_empty() {
            lines.extend(state.summary_lines.iter().map(|s| Line::from(s.as_str())));
        }

        lines
    };
    let summary_block = Paragraph::new(summary_content)
        .block(
            Block::default()
                .title(lbl.summary.as_str())
                .borders(Borders::ALL),
        )
        .wrap(Wrap { trim: true });
    f.render_widget(summary_block, right_chunks[2]);

    // Bottom area: Output (top) + Command Input/Status (bottom, combined)
    let bottom_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(3),    // Output textarea (scrollable)
            Constraint::Length(3), // Command Input + Status combined
        ])
        .split(chunks[1]);

    // Output textarea
    let output_height = bottom_chunks[0].height as usize;
    let visible_lines = output_height.saturating_sub(2); // Account for borders/padding
    let max_scroll = state.output_lines.len().saturating_sub(visible_lines);
    let scroll = state.output_scroll.min(max_scroll);
    // available width for text (minus 2 for border + 2 for padding)
    let avail_width = (bottom_chunks[0].width as usize).saturating_sub(4).max(10);
    // Word-wrap long lines so none exceed avail_width
    let display_lines: Vec<Line> = state
        .output_lines
        .iter()
        .skip(scroll)
        .take(visible_lines)
        .flat_map(|s| {
            let mut result = Vec::new();
            let clean = s.replace('\n', " ");
            // If line is short enough, emit as-is
            if clean.len() <= avail_width {
                result.push(Line::from(clean));
            } else {
                // Multi-line wrap (char-based to handle UTF-8 safely)
                let mut remaining = clean.as_str();
                loop {
                    if remaining.chars().count() <= avail_width {
                        result.push(Line::from(remaining.to_string()));
                        break;
                    }
                    // Find byte position for `avail_width` chars
                    let mut byte_end = 0;
                    let mut char_count = 0;
                    for (i, c) in remaining.char_indices() {
                        if char_count >= avail_width {
                            break;
                        }
                        byte_end = i + c.len_utf8();
                        char_count += 1;
                    }
                    if byte_end == 0 {
                        byte_end = remaining.len();
                    }
                    if let Some(pos) = remaining[..byte_end].rfind(' ') {
                        result.push(Line::from(remaining[..pos].to_string()));
                        remaining = &remaining[pos + 1..];
                    } else {
                        // No space, hard wrap at char boundary
                        result.push(Line::from(remaining[..byte_end].to_string()));
                        remaining = &remaining[byte_end..];
                    }
                }
            }
            result
        })
        .collect();
    let output_block = Paragraph::new(display_lines)
        .block(Block::default().title("Output").borders(Borders::ALL))
        .wrap(Wrap { trim: true })
        .scroll((scroll as u16, 0));
    f.render_widget(output_block, bottom_chunks[0]);

    // Command Input
    if state.input_mode {
        // Single-line: no scroll when text fits. Cursor (█) always appended at end.
        let visible_width = bottom_chunks[1].width.saturating_sub(4) as usize; // account for border/padding
        let content_len = state.input_buf.len();
        let scroll_x = if content_len > visible_width {
            state.input_scroll_x
        } else {
            0 // single line, no scroll
        };
        let input_text = format!("{}█", state.input_buf);
        let input_paragraph = Paragraph::new(input_text.as_str())
            .wrap(Wrap { trim: true })
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Command Input"),
            )
            .scroll((state.input_scroll_y as u16, scroll_x as u16));
        f.render_widget(input_paragraph, bottom_chunks[1]);
    } else {
        // Status lines when not in input mode
        let input_line = Line::from(vec![
            Span::raw("  "),
            Span::styled("/", Style::default().fg(Color::Yellow)),
            Span::raw(" for commands  |  "),
            Span::styled("Ctrl+C", Style::default().fg(Color::DarkGray)),
            Span::raw(" exit"),
        ]);

        let agent_info = if let (Some(agent), Some(task)) = (&state.active_agent, &state.agent_task)
        {
            format!("[{}] {}", agent, task)
        } else {
            String::new()
        };
        let status_text = if state.running { "Running" } else { "Idle" };
        let progress_info = state.agent_progress.as_deref().unwrap_or("");

        // Spinner frames for running animation
        const SPINNER_CHARS: [&str; 8] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧"];
        let spinner = if state.active_agent.is_some() {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as usize)
                .unwrap_or(0);
            Span::styled(
                format!("{} ", SPINNER_CHARS[now / 120 % SPINNER_CHARS.len()]),
                Style::default().fg(Color::Green),
            )
        } else {
            Span::raw("")
        };

        let combined_lines: Vec<Line> = vec![
            input_line,
            Line::from(vec![
                spinner,
                Span::raw(format!("{} ", status_text)),
                if !agent_info.is_empty() {
                    Span::styled(agent_info, Style::default().fg(Color::Cyan))
                } else {
                    Span::raw("")
                },
                if !progress_info.is_empty() {
                    Span::styled(
                        format!(" | {}", progress_info),
                        Style::default().fg(Color::DarkGray),
                    )
                } else {
                    Span::raw("")
                },
                Span::raw(" | Mouse wheel scrolls output | "),
                Span::styled("↑↓", Style::default().fg(Color::DarkGray)),
                Span::raw(" navigate autocomplete"),
            ]),
        ];

        let input_block = Paragraph::new(combined_lines)
            .block(Block::default().borders(Borders::ALL).title("Status"))
            .wrap(Wrap { trim: true });
        f.render_widget(input_block, bottom_chunks[1]);
    }

    // Autocomplete popup (overlays bottom area)
    if state.show_autocomplete && !state.autocomplete_items.is_empty() {
        let list_height = state.autocomplete_items.len().min(4);
        let scroll = state
            .autocomplete_scroll
            .min(state.autocomplete_items.len().saturating_sub(list_height));
        let ac_lines: Vec<Line> = state
            .autocomplete_items
            .iter()
            .skip(scroll)
            .take(list_height)
            .enumerate()
            .map(|(i, (cmd, hint))| {
                let global_idx = scroll + i;
                if global_idx == state.autocomplete_selected {
                    Line::from(vec![
                        Span::styled("  > ", Style::default().fg(Color::Cyan)),
                        Span::styled(
                            cmd,
                            Style::default()
                                .fg(Color::Yellow)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::raw(" "),
                        Span::styled(hint, Style::default().fg(Color::DarkGray)),
                    ])
                } else {
                    Line::from(vec![
                        Span::raw("    "),
                        Span::raw(cmd),
                        Span::raw(" "),
                        Span::styled(hint, Style::default().fg(Color::DarkGray)),
                    ])
                }
            })
            .collect();

        let ac_hint = Line::from(vec![
            Span::styled("↑↓", Style::default().fg(Color::DarkGray)),
            Span::raw(" navigate  "),
            Span::styled("Enter", Style::default().fg(Color::DarkGray)),
            Span::raw(" select  "),
            Span::styled("Esc", Style::default().fg(Color::DarkGray)),
            Span::raw(" cancel"),
        ]);

        let mut lines = ac_lines;
        lines.push(ac_hint);
        let ac_block =
            Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title("Commands"));

        // Position autocomplete above Command Input
        let ac_height = (list_height + 2) as u16;
        let ac_area = Rect::new(
            bottom_chunks[1].x,
            bottom_chunks[1].y.saturating_sub(ac_height),
            bottom_chunks[1].width,
            ac_height,
        );
        f.render_widget(Clear, ac_area);
        f.render_widget(ac_block, ac_area);
    }

    // Recommendation display (shown when not in input mode)
    if !state.input_mode {
        if let Some(ref reco) = state.recommendation {
            let mut reco_lines: Vec<Line> = Vec::new();
            if let Some(b) = reco.new_batch_size {
                reco_lines.push(Line::from(format!("batch_size → {}", b)));
            }
            if let Some(lr) = reco.new_learning_rate {
                reco_lines.push(Line::from(format!("learning_rate → {:.6}", lr)));
            }
            if let Some(ref opt) = reco.new_optimizer {
                reco_lines.push(Line::from(format!("optimizer → {}", opt)));
            }
            if let Some(acc) = reco.grad_accum_steps {
                reco_lines.push(Line::from(format!("grad_accum_steps → {} (advisory)", acc)));
            }
            if reco_lines.is_empty() {
                reco_lines.push(Line::from(lbl.no_changes.as_str()));
            }
            reco_lines.push(Line::from(lbl.press_a.as_str()));

            let reco_block = Paragraph::new(reco_lines).block(
                Block::default()
                    .title(lbl.recommendation.as_str())
                    .borders(Borders::ALL),
            );
            let reco_area = Rect::new(
                area.x + 1,
                area.y + area.height.saturating_sub(6),
                area.width.saturating_sub(2),
                5,
            );
            f.render_widget(Clear, reco_area);
            f.render_widget(reco_block, reco_area);
        }
    }
}

fn draw_config_table(f: &mut Frame<'_>, area: Rect, state: &AppState, lbl: &LabelsRef) {
    // Build config rows with recommended values from monitor
    let mut rows_data: Vec<[String; 3]> = vec![
        [
            "conda_env".into(),
            state.cfg.conda_env.clone().unwrap_or_default(),
            "".into(),
        ],
        [
            "backend".into(),
            state.cfg.backend.clone().unwrap_or_else(|| "cuda".into()),
            "".into(),
        ],
        [
            "train_script".into(),
            state.cfg.train_script.clone().unwrap_or_default(),
            "".into(),
        ],
    ];

    // Batch size with recommendation
    let batch_reco = state
        .stable_recommendation
        .as_ref()
        .map(|r| format!("→ {}", r.recommended_batch))
        .unwrap_or_default();
    rows_data.push([
        "batch_size".into(),
        state
            .cfg
            .batch_size
            .map(|v| v.to_string())
            .unwrap_or_default(),
        batch_reco,
    ]);

    // Learning rate with recommendation
    let lr_reco = state
        .stable_recommendation
        .as_ref()
        .map(|r| format!("→ {:.6}", r.recommended_lr))
        .unwrap_or_default();
    rows_data.push([
        "learning_rate".into(),
        state
            .cfg
            .learning_rate
            .map(|v| v.to_string())
            .unwrap_or_default(),
        lr_reco,
    ]);

    rows_data.push([
        "optimizer".into(),
        state.cfg.optimizer.clone().unwrap_or_default(),
        "".into(),
    ]);
    rows_data.push([
        "dataset_path".into(),
        state.cfg.dataset_path.clone().unwrap_or_default(),
        "".into(),
    ]);

    // Build table rows with 3 columns: field, current value, recommended
    let rows = rows_data.into_iter().map(|r| {
        let field = if r[2].is_empty() {
            Span::raw(r[0].clone())
        } else {
            Span::styled(r[0].clone(), Style::default().fg(Color::Cyan))
        };
        let value = Span::raw(r[1].clone());
        let reco = if r[2].is_empty() {
            Span::raw("")
        } else {
            Span::styled(r[2].clone(), Style::default().fg(Color::Green))
        };
        Row::new(vec![field, value, reco])
    });

    let widths = [
        Constraint::Percentage(35),
        Constraint::Percentage(35),
        Constraint::Percentage(30),
    ];
    let table = Table::new(rows, widths)
        .header(
            Row::new(vec![
                Span::styled(
                    lbl.config_table[0].to_string(),
                    Style::default().add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    "Current".to_string(),
                    Style::default().add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    "Monitor".to_string(),
                    Style::default().add_modifier(Modifier::BOLD),
                ),
            ])
            .style(Style::default().add_modifier(Modifier::BOLD)),
        )
        .block(
            Block::default()
                .title(lbl.guided_config.as_str())
                .borders(Borders::ALL),
        );

    let mut table_state = state.table_state.borrow_mut();
    f.render_stateful_widget(table, area, &mut table_state);
}

// ============================================================================
// Labels
// ============================================================================

struct LabelsRef {
    no_logs: String,
    press_a: String,
    no_changes: String,
    guided_config: String,
    recommendation: String,
    recent_logs: String,
    summary: String,
    config_table: Vec<String>,
}

fn labels(lang: &str) -> LabelsRef {
    if lang == "zh-CN" {
        LabelsRef {
            no_logs: "(暂无日志)".into(),
            press_a: "按 A 应用推荐".into(),
            no_changes: "无需调整".into(),
            guided_config: "引导配置".into(),
            recommendation: "推荐".into(),
            recent_logs: "最近日志".into(),
            summary: "摘要".into(),
            config_table: vec!["字段".into(), "值".into()],
        }
    } else {
        LabelsRef {
            no_logs: "(no logs yet)".into(),
            press_a: "Press A to apply recommendation".into(),
            no_changes: "No changes suggested".into(),
            guided_config: "Guided Config".into(),
            recommendation: "Recommendation".into(),
            recent_logs: "Recent Logs".into(),
            summary: "Summary".into(),
            config_table: vec!["Field".into(), "Value".into()],
        }
    }
}

// ============================================================================
// Main - Custom Render Loop
// ============================================================================

fn setup_terminal() -> Result<
    ratatui_kit::ratatui::Terminal<
        ratatui_kit::ratatui::backend::CrosstermBackend<std::io::Stdout>,
    >,
> {
    // Use ratatui's init which enters alternate screen + enables raw mode
    let terminal = ratatui_kit::ratatui::init();

    // crossterm mouse capture should be done AFTER ratatui::init
    // but ratatui::init already enables raw mode, so we just add mouse
    crossterm::execute!(std::io::stderr(), crossterm::event::EnableMouseCapture).ok();

    Ok(terminal)
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let _ = dotenvy::dotenv();

    // Check for project in current directory
    let default_cfg_path = PathBuf::from("config.yaml");
    let cfg_path = if default_cfg_path.exists() {
        default_cfg_path.clone()
    } else {
        eprintln!("[CoLoMo] 未在当前目录找到 config.yaml。是否创建一个新项目? (y/N)");
        let mut line = String::new();
        if std::io::stdin().read_line(&mut line).is_ok() && line.trim().eq_ignore_ascii_case("y") {
            // Use default project path
            let demo_cfg = PathBuf::from("../projects/demo/config.yaml");
            if demo_cfg.exists() {
                demo_cfg
            } else {
                // Create minimal config.yaml in current dir
                let minimal_cfg = r#"conda_env: colomo
backend: cuda
train_script: train.py
batch_size: 32
learning_rate: 0.001
optimizer: AdamW
dataset_path: ./data
param_count: 0
"#;
                let _ = std::fs::write("config.yaml", minimal_cfg);
                eprintln!("[CoLoMo] 已创建 config.yaml，请编辑后重新运行。");
                std::process::exit(0);
            }
        } else {
            eprintln!("[CoLoMo] 使用默认项目: ../projects/demo/config.yaml");
            PathBuf::from("../projects/demo/config.yaml")
        }
    };

    // Initialize terminal (handles alternate screen + raw mode)
    let terminal = setup_terminal()?;

    // Initialize state once
    {
        let cfg = config::load_config(&cfg_path).unwrap_or_default();
        let settings = settings::load(&PathBuf::from("settings.yaml")).unwrap_or_default();

        // Create and start system monitor with shared state
        let shared = get_monitor_state().clone();
        let stability = monitor::StabilityConfig::default();
        let (monitor_handle, receiver) =
            monitor::SystemMonitor::with_shared_state(stability, shared);
        let poll_interval = std::time::Duration::from_secs(5);
        monitor_handle.start(poll_interval);

        let mut state = AppState::default();
        state.cfg = cfg;
        state.cfg_path = cfg_path;
        state.settings = settings;
        state.table_state = RefCell::new(TableState::default());
        state.monitor_receiver = Some(receiver);

        // Initialize global UX sender for spawned tasks (shares the same channel as state.ux_rx)
        let ux_tx_for_global = state.ux_tx.clone();
        UX_GLOBAL_TX.get_or_init(|| ux_tx_for_global);

        let final_state = state;
        *get_state().lock().unwrap() = final_state;
    }

    // Run the UI loop in a blocking task so it doesn't block the async runtime
    tokio::task::spawn_blocking(move || run_ui_loop_inner(terminal)).await??;

    Ok(())
}

/// Inner UI loop - runs synchronously on a blocking thread
fn run_ui_loop_inner(
    mut terminal: ratatui_kit::ratatui::Terminal<
        ratatui_kit::ratatui::backend::CrosstermBackend<std::io::Stdout>,
    >,
) -> Result<()> {
    use ratatui_kit::crossterm::event::{Event, KeyEventKind};
    use std::time::Duration;

    // Drain all pending UX events from spawned tasks (non-blocking mpsc)
    fn poll_ux_events(state: &mut AppState) {
        while let Ok(event) = state.ux_rx.try_recv() {
            match event {
                UxEvent::AppendOutput(msg) => {
                    state.output_lines.push(msg);
                    state.output_scroll = usize::MAX;
                }
                UxEvent::SetActiveAgent(v) => {
                    state.active_agent = Some(v);
                }
                UxEvent::SetAgentTask(v) => {
                    state.agent_task = Some(v);
                }
                UxEvent::SetAgentProgress(v) => {
                    state.agent_progress = Some(v);
                }
                UxEvent::SetSummaryLines(v) => {
                    state.summary_lines = v;
                }
                UxEvent::SetTopRightLines(v) => {
                    state.top_right_lines = v;
                }
                UxEvent::SetOutputScroll(v) => {
                    state.output_scroll = v;
                }
                UxEvent::SetRunning(v) => {
                    state.running = v;
                }
                UxEvent::SetAgentLastSummary(v) => {
                    state.agent_last_summary = Some(v);
                }
            }
        }
    }

    loop {
        // Update monitor state from shared state before rendering
        {
            let mut state = get_state().lock().unwrap();
            poll_ux_events(&mut state);
            if let Ok(guard) = get_monitor_state().lock() {
                state.current_snapshot = guard.snapshot.clone();
                state.stable_recommendation = guard.recommendation.clone();
            }
        }

        // Render current state
        {
            let state = get_state().lock().unwrap();
            terminal.draw(|f| draw_tui(f, f.area(), &state))?;
        }

        // Poll for events with timeout
        if crossterm::event::poll(Duration::from_millis(50))? {
            match crossterm::event::read()? {
                Event::Key(key) => {
                    if key.kind != KeyEventKind::Press {
                        continue;
                    }

                    let pending_cmd: Option<String> = {
                        let mut state = get_state().lock().unwrap();

                        // Handle project creation flow
                        if state.creating_project {
                            handle_project_creation_input(key, &mut state)
                        } else if state.input_mode {
                            handle_input_mode_key(key, &mut state)
                        } else {
                            handle_normal_mode_key(key, &mut state)
                        }
                    };

                    // Execute pending commands
                    if let Some(ref cmd) = pending_cmd {
                        let state = get_state();
                        exec_command(cmd, (*state).clone());
                    }
                }
                Event::Mouse(mouse) => {
                    use crossterm::event::MouseEventKind;
                    let mut state = get_state().lock().unwrap();
                    match mouse.kind {
                        MouseEventKind::ScrollUp => {
                            if state.input_mode && state.input_buf.len() > 0 {
                                if state.input_scroll_y > 0 {
                                    state.input_scroll_y -= 1;
                                }
                            } else if state.output_scroll > 0 {
                                state.output_scroll -= 1;
                            }
                        }
                        MouseEventKind::ScrollDown => {
                            if state.input_mode && state.input_buf.len() > 0 {
                                state.input_scroll_y = state.input_scroll_y.saturating_add(1);
                            } else {
                                let visible_lines = 10.max(state.output_lines.len());
                                let max_scroll =
                                    state.output_lines.len().saturating_sub(visible_lines);
                                if state.output_scroll < max_scroll {
                                    state.output_scroll += 1;
                                }
                            }
                        }
                        MouseEventKind::Down(btn) if btn == crossterm::event::MouseButton::Left => {
                            if state.input_mode {
                                let text_start_x = 5;
                                let pos = if mouse.column > text_start_x {
                                    ((mouse.column - text_start_x) as usize)
                                        .min(state.input_buf.len())
                                } else {
                                    0
                                };
                                state.cursor_pos = pos;
                                state.selection_start = None;
                                state.selection_end = None;
                            }
                        }
                        MouseEventKind::Drag(btn) if btn == crossterm::event::MouseButton::Left => {
                            if state.input_mode {
                                let text_start_x = 5;
                                let pos = if mouse.column > text_start_x {
                                    ((mouse.column - text_start_x) as usize)
                                        .min(state.input_buf.len())
                                } else {
                                    0
                                };
                                if state.selection_end.is_none() {
                                    state.selection_start = Some(state.cursor_pos);
                                }
                                state.selection_end = Some(pos);
                            }
                        }
                        MouseEventKind::Up(btn) if btn == crossterm::event::MouseButton::Right => {
                            if state.input_mode {
                                if let Some(text) = get_clipboard() {
                                    let (start, end) = if let (Some(s), Some(e)) =
                                        (state.selection_start, state.selection_end)
                                    {
                                        (s.min(e), s.max(e))
                                    } else {
                                        (state.cursor_pos, state.cursor_pos)
                                    };
                                    let insert_pos = if state.selection_start.is_some() {
                                        start
                                    } else {
                                        state.cursor_pos
                                    };
                                    let delete_count = if state.selection_start.is_some() {
                                        end - start
                                    } else {
                                        0
                                    };
                                    // Replace selection with text
                                    state.input_buf.drain(insert_pos..insert_pos + delete_count);
                                    state.input_buf.insert_str(insert_pos, &text);
                                    state.cursor_pos = insert_pos + text.len();
                                    state.selection_start = None;
                                    state.selection_end = None;
                                }
                            }
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }

        // Yield to runtime periodically
        std::thread::sleep(std::time::Duration::from_micros(100));
    }
}
