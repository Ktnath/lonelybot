use core::num::NonZeroU8;
use std::env;
use std::time::Instant;

use lonelybot::belief::ResearchGame;
use lonelybot::belief_agent::{ParticleBeliefConfig, ParticleBeliefDecision};
use lonelybot::convert::convert_moves;
use lonelybot::shuffler::default_shuffle;
use lonelybot::solver::solve;
use lonelybot::standard::{Pos, StandardMove, StandardSolitaire};
use lonelybot::state::Solitaire;
use rand::{rngs::SmallRng, SeedableRng};
use serde_json::{json, Value};

const BUDGETS: [usize; 3] = [64, 256, 2048];

#[derive(Default)]
struct BudgetAggregate {
    n: usize,
    oracle_agree: usize,
    oracle_representable: usize,
    decision_ms: Vec<f64>,
    invalid_selected: usize,
}

fn action_string(action: &StandardMove) -> String {
    match (action.from, action.to) {
        (Pos::Deck, Pos::Deck) => "DRAW".to_string(),
        (Pos::Deck, Pos::Pile(p)) => format!("DECK->P{p} {}", action.card),
        (Pos::Deck, Pos::Stack(s)) => format!("DECK->F{s} {}", action.card),
        (Pos::Pile(a), Pos::Pile(b)) => format!("P{a}->P{b} {}", action.card),
        (Pos::Pile(p), Pos::Stack(s)) => format!("P{p}->F{s} {}", action.card),
        (Pos::Stack(s), Pos::Pile(p)) => format!("F{s}->P{p} {}", action.card),
        _ => format!("{:?}->{:?} {}", action.from, action.to, action.card),
    }
}

fn percentile(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut tmp = values.to_vec();
    tmp.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let idx = ((tmp.len() - 1) as f64 * q) as usize;
    tmp[idx]
}

fn selected_json(decision: &ParticleBeliefDecision, elapsed_ms: f64) -> Value {
    let s = &decision.actions[decision.chosen_index];
    json!({
        "particles": decision.particles,
        "chosen_action": action_string(&decision.chosen_action),
        "valid_particles": s.valid_particles,
        "invalid_particles": s.invalid_particles,
        "mean_value": s.mean_value,
        "stderr_value": s.stderr_value,
        "value_lcb": s.value_lcb,
        "win_rate": s.win_rate,
        "win_lcb": s.win_lcb,
        "deadlock_rate": s.deadlock_rate,
        "information_gain_hint": s.information_gain_hint,
        "decision_ms": elapsed_ms,
    })
}

fn main() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    let n_seeds: u64 = args
        .get(1)
        .map_or(Ok(30), |s| s.parse())
        .map_err(|e| format!("invalid seeds: {e}"))?;
    let max_checkpoints: usize = args
        .get(2)
        .map_or(Ok(16), |s| s.parse())
        .map_err(|e| format!("invalid max_checkpoints: {e}"))?;
    let max_per_seed: usize = args
        .get(3)
        .map_or(Ok(2), |s| s.parse())
        .map_err(|e| format!("invalid max_per_seed: {e}"))?;
    let rollout_depth: usize = args
        .get(4)
        .map_or(Ok(24), |s| s.parse())
        .map_err(|e| format!("invalid rollout_depth: {e}"))?;

    let draw_step = NonZeroU8::new(3).unwrap();
    let cfg = ParticleBeliefConfig {
        rollout_depth,
        ..Default::default()
    };

    let mut checkpoint_rows = Vec::new();
    let mut aggregates: [BudgetAggregate; 3] = core::array::from_fn(|_| BudgetAggregate::default());
    let mut stability_64_256 = 0usize;
    let mut stability_256_2048 = 0usize;
    let mut stability_64_2048 = 0usize;
    let mut oracle_missing_from_public = 0usize;
    let mut solved_seeds = 0usize;
    let mut scanned_standard_actions = 0usize;

    'seeds: for seed in 0..n_seeds {
        let cards = default_shuffle(seed);
        let mut oracle = Solitaire::new(&cards, draw_step);
        let (_, solution) = solve(&mut oracle);
        let Some(solution) = solution else {
            continue;
        };
        solved_seeds += 1;

        let mut research = ResearchGame::new(&cards, draw_step)
            .map_err(|e| format!("seed {seed} research init: {e:?}"))?;
        let mut shadow = StandardSolitaire::new(&cards, draw_step);
        let mut seed_checkpoints = 0usize;
        let mut step_in_seed = 0usize;

        for mv in solution {
            let sequence = convert_moves(&mut shadow, &[mv])
                .map_err(|_| format!("seed {seed}: optimized-to-standard conversion failed"))?;

            for action in sequence {
                scanned_standard_actions += 1;
                let candidates = research.belief().public_actions();
                let oracle_representable = candidates.iter().any(|candidate| candidate == &action);
                if !oracle_representable {
                    oracle_missing_from_public += 1;
                }

                let eligible = step_in_seed >= 3
                    && seed_checkpoints < max_per_seed
                    && checkpoint_rows.len() < max_checkpoints
                    && research.belief().unknown_slot_count() > 0
                    && candidates.len() >= 2;

                if eligible {
                    let eval_seed = 0xB311_EF00_0000_0000u64
                        ^ seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        ^ (step_in_seed as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
                    let mut budget_rows = Vec::new();
                    let mut chosen = Vec::new();

                    for (budget_idx, budget) in BUDGETS.iter().copied().enumerate() {
                        // Resetting to the exact same seed makes Belief-64 an exact
                        // random prefix of Belief-256 and Belief-2048.
                        let mut rng = SmallRng::seed_from_u64(eval_seed);
                        let start = Instant::now();
                        let decision = research
                            .belief()
                            .particle_decision(budget, &cfg, &mut rng)
                            .map_err(|e| format!("seed {seed} step {step_in_seed} budget {budget}: {e:?}"))?;
                        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
                        let chosen_string = action_string(&decision.chosen_action);
                        let oracle_string = action_string(&action);
                        let chosen_stats = &decision.actions[decision.chosen_index];

                        let agg = &mut aggregates[budget_idx];
                        agg.n += 1;
                        agg.oracle_representable += usize::from(oracle_representable);
                        agg.oracle_agree += usize::from(chosen_string == oracle_string);
                        agg.invalid_selected += usize::from(chosen_stats.invalid_particles > 0);
                        agg.decision_ms.push(elapsed_ms);

                        chosen.push(chosen_string);
                        budget_rows.push(selected_json(&decision, elapsed_ms));
                    }

                    stability_64_256 += usize::from(chosen[0] == chosen[1]);
                    stability_256_2048 += usize::from(chosen[1] == chosen[2]);
                    stability_64_2048 += usize::from(chosen[0] == chosen[2]);

                    checkpoint_rows.push(json!({
                        "seed": seed,
                        "step_in_seed": step_in_seed,
                        "known_initial_slots": research.belief().known_slot_count(),
                        "unknown_initial_slots": research.belief().unknown_slot_count(),
                        "public_action_count": candidates.len(),
                        "oracle_next_action": action_string(&action),
                        "oracle_representable": oracle_representable,
                        "budgets": budget_rows,
                    }));
                    seed_checkpoints += 1;
                }

                research
                    .step(action)
                    .map_err(|e| format!("seed {seed} step {step_in_seed}: {e:?}"))?;
                step_in_seed += 1;

                if checkpoint_rows.len() >= max_checkpoints {
                    break 'seeds;
                }
            }
        }
    }

    let n_cp = checkpoint_rows.len();
    let budget_summary: Vec<Value> = BUDGETS
        .iter()
        .enumerate()
        .map(|(idx, &budget)| {
            let agg = &aggregates[idx];
            let mean_ms = if agg.decision_ms.is_empty() {
                0.0
            } else {
                agg.decision_ms.iter().sum::<f64>() / agg.decision_ms.len() as f64
            };
            json!({
                "particles": budget,
                "checkpoints": agg.n,
                "oracle_representable_rate": if agg.n == 0 { 0.0 } else { agg.oracle_representable as f64 / agg.n as f64 },
                "oracle_agreement_rate": if agg.n == 0 { 0.0 } else { agg.oracle_agree as f64 / agg.n as f64 },
                "selected_with_invalid_particles": agg.invalid_selected,
                "decision_ms_mean": mean_ms,
                "decision_ms_p50": percentile(&agg.decision_ms, 0.50),
                "decision_ms_p95": percentile(&agg.decision_ms, 0.95),
            })
        })
        .collect();

    let summary = json!({
        "agent": "particle_root_sampling_v1",
        "important_limit": "future rollout policy is determinized inside each sampled world; this is a belief-root baseline, not POMCP/information-set MCTS",
        "draw_step": 3,
        "seeds_requested": n_seeds,
        "solved_seeds_entered": solved_seeds,
        "rollout_depth": rollout_depth,
        "budgets": BUDGETS,
        "checkpoints": n_cp,
        "scanned_standard_actions": scanned_standard_actions,
        "oracle_missing_from_public_actions": oracle_missing_from_public,
        "stability_64_256": if n_cp == 0 { 0.0 } else { stability_64_256 as f64 / n_cp as f64 },
        "stability_256_2048": if n_cp == 0 { 0.0 } else { stability_256_2048 as f64 / n_cp as f64 },
        "stability_64_2048": if n_cp == 0 { 0.0 } else { stability_64_2048 as f64 / n_cp as f64 },
        "budget_summary": budget_summary,
        "checkpoint_rows": checkpoint_rows,
    });

    println!("{}", serde_json::to_string_pretty(&summary).map_err(|e| e.to_string())?);
    Ok(())
}
