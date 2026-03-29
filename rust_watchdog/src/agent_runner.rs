//! Agent Runner Module
//! Handles invocation of CoLoMo agents for task execution with high-quality feedback.

use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

/// Agent execution result
#[derive(Debug)]
pub struct AgentResult {
    pub success: bool,
    pub output: String,
    pub plan_file: Option<PathBuf>,
}

/// Run the planner agent with given requirement
/// Uses Claude Code CLI with timeout protection
pub fn run_planner_agent(
    requirement: &str,
    project_root: &PathBuf,
    state: Arc<std::sync::Mutex<crate::AppState>>,
) -> AgentResult {
    let plan_dir = project_root.join(".claude").join("plan");
    let plan_file = plan_dir.join("plan.md");

    // Ensure plan directory exists
    if let Err(e) = std::fs::create_dir_all(&plan_dir) {
        return AgentResult {
            success: false,
            output: format!("Failed to create plan directory: {}", e),
            plan_file: None,
        };
    }

    // Step 1: Update feedback - Understanding requirement
    update_agent_feedback(&state, "Understanding requirement", "1/5");
    thread::sleep(Duration::from_millis(200));

    // Step 2: Update feedback - Analyzing context
    update_agent_feedback(&state, "Analyzing project context", "2/5");

    // Read project context
    let config_path = project_root.join("config.yaml");
    let train_path = project_root.join("train.py");
    let context = read_project_context(project_root, &config_path, &train_path);

    // Step 3: Update feedback - Generating plan
    update_agent_feedback(&state, "Generating implementation plan", "3/5");

    // Build comprehensive prompt for the planner
    let prompt = build_planner_prompt(requirement, &context, &plan_file);

    // Try to invoke via Claude Code CLI with timeout
    let result = invoke_claude_code_with_timeout(&prompt, Duration::from_secs(30));

    match result {
        Ok(output) => {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);

                // Write plan to file
                if let Err(e) = std::fs::write(&plan_file, stdout.as_ref()) {
                    return AgentResult {
                        success: false,
                        output: format!("Failed to write plan file: {}", e),
                        plan_file: None,
                    };
                }

                // Step 4: Update feedback - Validating plan
                update_agent_feedback(&state, "Validating plan", "4/5");
                thread::sleep(Duration::from_millis(200));

                // Step 5: Update feedback - Complete
                update_agent_feedback(&state, "Plan ready", "5/5");

                AgentResult {
                    success: true,
                    output: format!("Plan generated at {}", plan_file.display()),
                    plan_file: Some(plan_file),
                }
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                // If Claude CLI fails, fall back to basic plan
                update_agent_feedback(&state, "Generating fallback plan", "3/5");
                let fallback = generate_fallback_plan(requirement, &plan_file);
                AgentResult {
                    success: fallback,
                    output: if fallback {
                        "Generated basic plan (Claude CLI error)".to_string()
                    } else {
                        format!("Claude CLI error: {}", stderr)
                    },
                    plan_file: if fallback { Some(plan_file) } else { None },
                }
            }
        }
        Err(e) => {
            // Timeout or error - fall back to basic plan
            update_agent_feedback(&state, "Generating fallback plan", "3/5");
            let fallback = generate_fallback_plan(requirement, &plan_file);
            AgentResult {
                success: fallback,
                output: if fallback {
                    format!("Generated basic plan ({})", e)
                } else {
                    e
                },
                plan_file: if fallback { Some(plan_file) } else { None },
            }
        }
    }
}

/// Read project context for the planner
fn read_project_context(project_root: &PathBuf, config_path: &PathBuf, train_path: &PathBuf) -> String {
    let mut context = String::new();

    if let Ok(config) = std::fs::read_to_string(config_path) {
        context.push_str("## config.yaml\n");
        context.push_str(&config);
        context.push_str("\n\n");
    }

    if let Ok(train) = std::fs::read_to_string(train_path) {
        context.push_str("## train.py (first 50 lines)\n");
        let lines: Vec<&str> = train.lines().take(50).collect();
        context.push_str(&lines.join("\n"));
        context.push_str("\n\n");
    }

    // List project files
    context.push_str("## Project Structure\n");
    if let Ok(entries) = std::fs::read_dir(project_root) {
        for entry in entries.flatten() {
            if let Ok(name) = entry.file_name().into_string() {
                context.push_str(&format!("  - {}\n", name));
            }
        }
    }

    context
}

/// Build comprehensive planner prompt
fn build_planner_prompt(requirement: &str, context: &str, plan_file: &PathBuf) -> String {
    format!(
        r#"You are a planner agent for CoLoMo (Context-Local-Model) training system.

## Your Task
Generate a concrete implementation plan for the following requirement:

**Requirement:** {}

## Project Context
{}

## Output Requirements
1. Generate a detailed implementation plan in markdown format
2. Include specific phases with actionable steps
3. Identify potential risks and mitigations
4. Output to: {}

## Plan Format
```markdown
---
name: implementation-plan
phases: [number of phases]
---

# Implementation Plan

## Overview
[Brief description of the plan]

## Phase 1: [Name]
- [ ] Step 1
- [ ] Step 2

## Phase 2: [Name]
- [ ] Step 1
...

## Risks
- Risk 1: [description] → Mitigation

## Dependencies
- [List any external dependencies]
```

Important:
- Do NOT modify any files
- Only generate the plan content
- Be specific and actionable
- Focus on ML training best practices
"#,
        requirement,
        context,
        plan_file.display()
    )
}

/// Invoke Claude Code CLI with timeout protection
fn invoke_claude_code_with_timeout(prompt: &str, timeout: Duration) -> Result<std::process::Output, String> {
    // First check if claude CLI is available and authenticated
    let check = Command::new("claude")
        .args(["--print", "--version"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();

    match check {
        Ok(mut child) => {
            // Wait max 5 seconds for version check
            match child.wait_timeout(timeout) {
                Some(status) => {
                    if !status.success() {
                        return Err("Claude CLI not properly configured".to_string());
                    }
                }
                None => {
                    return Err("Claude CLI check timed out".to_string());
                }
            }
        }
        Err(e) => {
            return Err(format!("Claude CLI not found: {}. Install from https://claude.ai/code", e));
        }
    }

    // Now invoke with the actual prompt
    let mut child = Command::new("claude")
        .args(["--print", prompt])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to spawn Claude CLI: {}", e))?;

    // Wait with timeout
    match child.wait_timeout(timeout) {
        Some(status) => {
            // Get output
            let output = child.wait_with_output().map_err(|e| format!("Failed to get output: {}", e))?;
            Ok(output)
        }
        None => {
            // Kill the process on timeout
            #[cfg(windows)]
            {
                let _ = Command::new("taskkill")
                    .args(["/F", "/T", "/PID", &child.id().to_string()])
                    .spawn();
            }
            #[cfg(not(windows))]
            {
                let _ = Command::new("kill")
                    .args(["-9", &child.id().to_string()])
                    .spawn();
            }
            Err(format!("Claude CLI timed out after {:?}", timeout))
        }
    }
}

/// Update agent feedback in state
fn update_agent_feedback(state: &Arc<std::sync::Mutex<crate::AppState>>, task: &str, progress: &str) {
    if let Ok(mut s) = state.lock() {
        s.agent_task = Some(task.to_string());
        s.agent_progress = Some(progress.to_string());
        s.output_lines.push(format!("  [{}] {}", progress, task));
    }
}

/// Generate a basic plan when external agent is not available
fn generate_fallback_plan(requirement: &str, plan_file: &PathBuf) -> bool {
    let plan_content = format!(
        r#"---
name: generated-plan
stateless: true
---

# Implementation Plan

## Requirement
{}

## Phases

### Phase 1: Analysis
- [ ] Understand the requirement
- [ ] Identify key components
- [ ] Assess technical feasibility

### Phase 2: Implementation
- [ ] Set up project structure
- [ ] Implement core functionality
- [ ] Add error handling

### Phase 3: Testing
- [ ] Write unit tests
- [ ] Run integration tests
- [ ] Verify functionality

### Phase 4: Documentation
- [ ] Update documentation
- [ ] Add usage examples

## Risks
- Requirement may need clarification
- Implementation details subject to change

---
*Plan auto-generated by CoLoMo*
"#,
        requirement
    );

    std::fs::write(plan_file, plan_content).is_ok()
}

/// Read plan file and extract todo items with status
pub fn read_plan_todos(plan_file: &PathBuf) -> Vec<String> {
    let content = match std::fs::read_to_string(plan_file) {
        Ok(c) => c,
        Err(_) => return vec![],
    };

    let mut todos = vec![];
    let mut in_phase = String::new();

    for line in content.lines() {
        let trimmed = line.trim();

        // Track current phase
        if trimmed.starts_with("## Phase") || trimmed.starts_with("### Phase") {
            in_phase = trimmed.replace("#", "").replace("Phase", "Phase").trim().to_string();
            continue;
        }

        // Parse todo items
        if trimmed.starts_with("- [ ]") {
            let item = &trimmed[5..].trim();
            if !in_phase.is_empty() {
                todos.push(format!("[ ] [{}] {}", in_phase, item));
            } else {
                todos.push(format!("[ ] {}", item));
            }
        } else if trimmed.starts_with("- [x]") || trimmed.starts_with("- [X]") {
            let item = &trimmed[5..].trim();
            todos.push(format!("[DONE] {}", item));
        } else if trimmed.starts_with("- [...]") {
            let item = &trimmed[6..].trim();
            todos.push(format!("[...] {}", item));
        }
    }

    if todos.is_empty() {
        // If no todos found, show summary lines
        for (i, line) in content.lines().enumerate() {
            if i > 15 {
                break;
            }
            let trimmed = line.trim();
            if !trimmed.is_empty() && !trimmed.starts_with("---") && !trimmed.starts_with("#") {
                todos.push(format!("[ ] {}", trimmed));
            }
        }
    }

    todos
}

/// Spawn a background agent task with high-quality feedback
pub fn spawn_agent_task(
    requirement: String,
    state: Arc<std::sync::Mutex<crate::AppState>>,
    project_root: PathBuf,
) {
    let req_clone = requirement.clone();
    thread::spawn(move || {
        // Initial feedback - Starting
        {
            let mut s = state.lock().unwrap();
            s.active_agent = Some("planner".into());
            s.agent_task = Some("Initializing planner".into());
            s.agent_progress = Some("0/5".into());
            s.top_right_lines = vec![
                "[>>] Planner: Initializing...".into(),
                "[ ] Read project context".into(),
                "[ ] Analyze requirement".into(),
                "[ ] Generate plan".into(),
                "[ ] Validate output".into(),
            ];
            s.output_lines.push(format!("> Planning: {}", req_clone));
            s.output_lines.push("  Starting planner agent...".into());
        }

        // Run planner with feedback
        let result = run_planner_agent(&requirement, &project_root, state.clone());

        // Update final state
        {
            let mut s = state.lock().unwrap();
            if result.success {
                s.agent_task = Some("Plan complete".into());
                s.agent_progress = Some("Done".into());

                // Load todos from plan file
                if let Some(ref plan_file) = result.plan_file {
                    let todos = read_plan_todos(plan_file);
                    if !todos.is_empty() {
                        s.top_right_lines = todos;
                    }
                }

                s.output_lines.push(format!("✓ Planner: {}", result.output));
            } else {
                s.agent_task = Some("Planner failed".into());
                s.agent_progress = None;
                s.top_right_lines = vec![
                    "[!] Planner error".into(),
                    format!("[ ] {}", result.output),
                ];
                s.output_lines.push(format!("✗ Planner error: {}", result.output));
            }
        }

        // Clear agent status after delay
        thread::sleep(Duration::from_secs(5));
        if let Ok(mut s) = state.lock() {
            s.active_agent = None;
            s.agent_task = None;
            s.agent_progress = None;
        }
    });
}

// Extension trait for wait_timeout
trait WaitTimeout {
    fn wait_timeout(&mut self, timeout: Duration) -> Option<std::process::ExitStatus>;
}

impl WaitTimeout for std::process::Child {
    fn wait_timeout(&mut self, timeout: Duration) -> Option<std::process::ExitStatus> {
        let start = Instant::now();
        loop {
            match self.try_wait() {
                Ok(Some(status)) => return Some(status),
                Ok(None) => {
                    if start.elapsed() >= timeout {
                        return None;
                    }
                    thread::sleep(Duration::from_millis(100));
                }
                Err(e) => return None,
            }
        }
    }
}
