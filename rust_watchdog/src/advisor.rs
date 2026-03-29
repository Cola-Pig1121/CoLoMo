use crate::config::Config as TrainConfig;
use crate::gpu_monitor::GpuStatus;

#[derive(Debug, Clone)]
pub struct Recommendation {
    pub new_batch_size: Option<u64>,
    pub new_learning_rate: Option<f64>,
    pub new_optimizer: Option<String>,
    pub grad_accum_steps: Option<u64>,
    pub rationale: String,
}

impl Recommendation {
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.new_batch_size.is_none()
            && self.new_learning_rate.is_none()
            && self.new_optimizer.is_none()
            && self.grad_accum_steps.is_none()
    }
}

#[allow(dead_code)]
pub fn recommend(cfg: &TrainConfig, gpu: &GpuStatus, recent_logs: &[String]) -> Recommendation {
    // Heuristics derived from Golden Rules (rough, safe defaults)
    let mut rationale_lines: Vec<String> = Vec::new();

    // 1) Batch size from memory/OOM signal
    let mut new_batch: Option<u64> = None;
    let old_batch = cfg.batch_size.unwrap_or(0);
    let oom = recent_logs.iter().any(|l| {
        let l = l.to_ascii_lowercase();
        l.contains("out of memory") || l.contains("cuda out of memory")
    });

    if oom && old_batch > 1 {
        let halved = (old_batch / 2).max(1);
        new_batch = Some(halved);
        rationale_lines.push(format!(
            "Detected OOM in logs → halve batch {} → {}",
            old_batch, halved
        ));
    } else if let (Some(total), Some(used)) = (gpu.total_mb, gpu.used_mb) {
        // If we're using < 50% of memory and have an explicit batch size, nudge up by 25%
        if old_batch > 0 && used * 2 < total {
            let increased = ((old_batch as f64) * 1.25).round() as u64;
            if increased > old_batch {
                new_batch = Some(increased);
                rationale_lines.push(format!(
                    "GPU usage low ({} / {} MB) → increase batch {} → {}",
                    used, total, old_batch, increased
                ));
            }
        }
    }

    // 2) Linear LR scaling with batch size
    let mut new_lr: Option<f64> = None;
    if let (Some(old_lr), Some(nb)) = (cfg.learning_rate, new_batch) {
        if old_batch > 0 && nb != old_batch {
            let scaled = old_lr * (nb as f64 / old_batch as f64);
            new_lr = Some(scaled);
            rationale_lines.push(format!(
                "Scale LR linearly with batch: {:.6} → {:.6}",
                old_lr, scaled
            ));
        }
    }

    // 3) Optimizer selection by parameter count thresholds
    let mut new_opt: Option<String> = None;
    if let Some(params) = cfg.param_count {
        let suggested = if params >= 100_000_000 {
            "AdamW"
        } else if params >= 10_000_000 {
            "Adam"
        } else {
            "SGD"
        };
        match cfg.optimizer.as_deref() {
            Some(current) if current.eq_ignore_ascii_case(suggested) => {}
            _ => {
                new_opt = Some(suggested.to_string());
                rationale_lines.push(format!(
                    "Param count ~{} → suggest {}",
                    params, suggested
                ));
            }
        }
    }

    // 4) Gradient accumulation to preserve effective batch when reducing
    let mut new_accum: Option<u64> = None;
    if let (Some(nb), Some(ob)) = (new_batch, if old_batch > 0 { Some(old_batch) } else { None }) {
        if nb < ob {
            // ceil(old/new)
            let accum = ob.div_ceil(nb);
            if accum > 1 {
                new_accum = Some(accum);
                rationale_lines.push(format!(
                    "Reduce batch {}→{} → set grad_accum_steps={} to preserve effective batch",
                    ob, nb, accum
                ));
            }
        }
    }

    let rationale = if rationale_lines.is_empty() {
        "No strong signals for adjustment; keeping current settings.".to_string()
    } else {
        rationale_lines.join("\n")
    };

    Recommendation {
        new_batch_size: new_batch,
        new_learning_rate: new_lr,
        new_optimizer: new_opt,
        grad_accum_steps: new_accum,
        rationale,
    }
}
