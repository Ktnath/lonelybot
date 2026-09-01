use core::num::NonZeroU8;
use std::env;
use std::time::Instant;

use lonelybot::belief::ResearchGame;
use lonelybot::belief_agent::{AdaptiveParticleDecision, ParticleBeliefConfig, ParticleBeliefDecision};
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
    invalid_decisions: usize,
    decision_ms: Vec<f64>,
    stderr: Vec<f64>,
}

#[derive(Default)]
struct AdaptiveAggregate {
    n: usize,
    used_64: usize,
    used_256: usize,
    used_2048: usize,
    match_reference: usize,
    oracle_agree: usize,
    decision_ms: Vec<f64>,
    confidence_gaps: Vec<f64>,
    particle_sum: usize,
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
    let actions_with_invalid = decision
        .actions
        .iter()
        .filter(|stats| stats.invalid_particles > 0)
        .count();
    json!({
        "particles": decision.particles,
        "chosen_action": action_string(&decision.chosen_action),
        "candidate_actions": decision.actions.len(),
        "actions_with_invalid_particles": actions_with_invalid,
        "mean_value": s.mean_value,
        "stderr_value": s.stderr_value,
        "value_lcb": s.value_lcb,
        "value_ucb": s.value_ucb,
        "win_rate": s.win_rate,
        "deadlock_rate": s.deadlock_rate,
        "information_gain_hint": s.information_gain_hint,
        "foundation_progress": s.mean_foundation_progress,
        "tableau_reveal_progress": s.mean_tableau_reveal_progress,
        "stock_clear_progress": s.mean_stock_clear_progress,
        "mobility": s.mean_mobility,
        "empty_columns": s.mean_empty_columns,
        "stock_accessibility": s.mean_stock_accessibility,
        "reveal_options": s.mean_reveal_options,
        "decision_ms": elapsed_ms,
    })
}

fn adaptive_json(result: &AdaptiveParticleDecision, elapsed_ms: f64) -> Value {
    let s = &result.decision.actions[result.decision.chosen_index];
    json!({
        "particles": result.decision.particles,
        "chosen_action": action_string(&result.decision.chosen_action),
        "stopped_early": result.stopped_early,
        "confidence_gap": result.confidence_gap,
        "mean_value": s.mean_value,
        "stderr_value": s.stderr_value,
        "value_lcb": s.value_lcb,
        "value_ucb": s.value_ucb,
        "deadlock_rate": s.deadlock_rate,
        "information_gain_hint": s.information_gain_hint,
        "decision_ms": elapsed_ms,
    })
}

fn knowledge_stage(unknown: usize) -> usize {
    if unknown >= 36 {
        0 // early
    } else if unknown >= 18 {
        1 // mid
    } else {
        2 // late
    }
}

fn stage_name(stage: usize) -> &'static str {
    match stage {
        0 => "early",
        1 => "mid",
        _ => "late",
    }
}

fn main() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    let n_seeds: u64 = args
        .get(1)
        .map_or(Ok(40), |s| s.parse())
        .map_err(|e| format!("invalid seeds: {e}"))?;
    let max_checkpoints: usize = args
        .get(2)
        .map_or(Ok(24), |s| s.parse())
        .map_err(|e| format!("invalid checkpoints: {e}"))?;
    let rollout_depth: usize = args
        .get(3)
        .map_or(Ok(32), |s| s.parse())
        .map_err(|e| format!("invalid rollout depth: {e}"))?;

    let draw_step = NonZeroU8::new(3).unwrap();
    let cfg = ParticleBeliefConfig {
        rollout_depth,
        ..Default::default()
    };

    let mut aggregates: [BudgetAggregate; 3] = core::array::from_fn(|_| BudgetAggregate::default());
    let mut adaptive = AdaptiveAggregate::default();
    let mut checkpoint_rows = Vec::new();
    let mut stability_64_256 = 0usize;
    let mut stability_256_2048 = 0usize;
    let mut stability_64_2048 = 0usize;
    let mut oracle_missing = 0usize;
    let mut scanned_actions = 0usize;
    let mut solved_seeds = 0usize;
    let mut stage_counts = [0usize; 3];

    'seeds: for seed in 0..n_seeds {
        let cards = default_shuffle(seed);
        let mut oracle = Solitaire::new(&cards, draw_step);
        let (_, solution) = solve(&mut oracle);
        let Some(solution) = solution else {
            continue;
        };
        solved_seeds += 1;

        let mut research = ResearchGame::new(&cards, draw_step)
            .map_err(|e| format!("seed {seed} init: {e:?}"))?;
        let mut shadow = StandardSolitaire::new(&cards, draw_step);
        let mut collected_stage = [false; 3];
        let mut step_in_seed = 0usize;

        for mv in solution {
            let sequence = convert_moves(&mut shadow, &[mv])
                .map_err(|_| format!("seed {seed}: conversion failed"))?;

            for action in sequence {
                scanned_actions += 1;
                let candidates = research.belief().public_actions();
                let representable = candidates.iter().any(|c| c == &action);
                if !representable {
                    oracle_missing += 1;
                }

                let unknown = research.belief().unknown_slot_count();
                let stage = knowledge_stage(unknown);
                let eligible = step_in_seed >= 3
                    && unknown > 0
                    && candidates.len() >= 2
                    && !collected_stage[stage]
                    && checkpoint_rows.len() < max_checkpoints;

                if eligible {
                    let eval_seed = 0xE1A2_0000_0000_0000u64
                        ^ seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        ^ (step_in_seed as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
                    let oracle_string = action_string(&action);
                    let mut fixed_rows = Vec::new();
                    let mut choices = Vec::new();

                    for (idx, budget) in BUDGETS.iter().copied().enumerate() {
                        let mut rng = SmallRng::seed_from_u64(eval_seed);
                        let start = Instant::now();
                        let decision = research
                            .belief()
                            .particle_decision(budget, &cfg, &mut rng)
                            .map_err(|e| format!("seed {seed} step {step_in_seed} fixed {budget}: {e:?}"))?;
                        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                        let selected = &decision.actions[decision.chosen_index];
                        let chosen = action_string(&decision.chosen_action);

                        let agg = &mut aggregates[idx];
                        agg.n += 1;
                        agg.oracle_agree += usize::from(chosen == oracle_string);
                        agg.invalid_decisions += usize::from(decision.actions.iter().any(|s| s.invalid_particles > 0));
                        agg.decision_ms.push(elapsed);
                        agg.stderr.push(selected.stderr_value);

                        choices.push(chosen);
                        fixed_rows.push(selected_json(&decision, elapsed));
                    }

                    stability_64_256 += usize::from(choices[0] == choices[1]);
                    stability_256_2048 += usize::from(choices[1] == choices[2]);
                    stability_64_2048 += usize::from(choices[0] == choices[2]);

                    let mut rng = SmallRng::seed_from_u64(eval_seed);
                    let start = Instant::now();
                    let adaptive_result = research
                        .belief()
                        .adaptive_particle_decision(&BUDGETS, &cfg, &mut rng)
                        .map_err(|e| format!("seed {seed} step {step_in_seed} adaptive: {e:?}"))?;
                    let adaptive_ms = start.elapsed().as_secs_f64() * 1000.0;
                    let adaptive_choice = action_string(&adaptive_result.decision.chosen_action);
                    let used = adaptive_result.decision.particles;

                    adaptive.n += 1;
                    adaptive.particle_sum += used;
                    adaptive.used_64 += usize::from(used == 64);
                    adaptive.used_256 += usize::from(used == 256);
                    adaptive.used_2048 += usize::from(used == 2048);
                    adaptive.match_reference += usize::from(adaptive_choice == choices[2]);
                    adaptive.oracle_agree += usize::from(adaptive_choice == oracle_string);
                    adaptive.decision_ms.push(adaptive_ms);
                    adaptive.confidence_gaps.push(adaptive_result.confidence_gap);

                    checkpoint_rows.push(json!({
                        "seed": seed,
                        "step_in_seed": step_in_seed,
                        "knowledge_stage": stage_name(stage),
                        "known_initial_slots": research.belief().known_slot_count(),
                        "unknown_initial_slots": unknown,
                        "public_action_count": candidates.len(),
                        "oracle_next_action": oracle_string,
                        "oracle_representable": representable,
                        "fixed": fixed_rows,
                        "adaptive": adaptive_json(&adaptive_result, adaptive_ms),
                        "adaptive_matches_2048": adaptive_choice == choices[2],
                    }));
                    collected_stage[stage] = true;
                    stage_counts[stage] += 1;

                    if checkpoint_rows.len() >= max_checkpoints {
                        break 'seeds;
                    }
                }

                research
                    .step(action)
                    .map_err(|e| format!("seed {seed} step {step_in_seed}: {e:?}"))?;
                step_in_seed += 1;
            }
        }
    }

    let n_cp = checkpoint_rows.len();
    let fixed_summary: Vec<Value> = BUDGETS
        .iter()
        .enumerate()
        .map(|(idx, &budget)| {
            let agg = &aggregates[idx];
            let mean_ms = if agg.decision_ms.is_empty() { 0.0 } else { agg.decision_ms.iter().sum::<f64>() / agg.decision_ms.len() as f64 };
            let mean_stderr = if agg.stderr.is_empty() { 0.0 } else { agg.stderr.iter().sum::<f64>() / agg.stderr.len() as f64 };
            json!({
                "particles": budget,
                "checkpoints": agg.n,
                "oracle_agreement_rate": if agg.n == 0 {0.0} else {agg.oracle_agree as f64 / agg.n as f64},
                "invalid_decisions": agg.invalid_decisions,
                "stderr_mean": mean_stderr,
                "decision_ms_mean": mean_ms,
                "decision_ms_p50": percentile(&agg.decision_ms, 0.50),
                "decision_ms_p95": percentile(&agg.decision_ms, 0.95),
            })
        })
        .collect();

    let adaptive_summary = json!({
        "checkpoints": adaptive.n,
        "mean_particles": if adaptive.n == 0 {0.0} else {adaptive.particle_sum as f64 / adaptive.n as f64},
        "stopped_at_64": adaptive.used_64,
        "stopped_at_256": adaptive.used_256,
        "used_2048": adaptive.used_2048,
        "match_fixed_2048_rate": if adaptive.n == 0 {0.0} else {adaptive.match_reference as f64 / adaptive.n as f64},
        "oracle_agreement_rate": if adaptive.n == 0 {0.0} else {adaptive.oracle_agree as f64 / adaptive.n as f64},
        "decision_ms_mean": if adaptive.n == 0 {0.0} else {adaptive.decision_ms.iter().sum::<f64>() / adaptive.n as f64},
        "decision_ms_p50": percentile(&adaptive.decision_ms, 0.50),
        "decision_ms_p95": percentile(&adaptive.decision_ms, 0.95),
        "confidence_gap_median": percentile(&adaptive.confidence_gaps, 0.50),
    });

    let summary = json!({
        "agent": "particle_evaluator_v2_adaptive",
        "draw_step": 3,
        "rollout_depth": rollout_depth,
        "budgets": BUDGETS,
        "seeds_requested": n_seeds,
        "solved_seeds_entered": solved_seeds,
        "checkpoints": n_cp,
        "knowledge_stage_counts": {"early":stage_counts[0], "mid":stage_counts[1], "late":stage_counts[2]},
        "scanned_standard_actions": scanned_actions,
        "oracle_missing_from_public_actions": oracle_missing,
        "stability_64_256": if n_cp == 0 {0.0} else {stability_64_256 as f64 / n_cp as f64},
        "stability_256_2048": if n_cp == 0 {0.0} else {stability_256_2048 as f64 / n_cp as f64},
        "stability_64_2048": if n_cp == 0 {0.0} else {stability_64_2048 as f64 / n_cp as f64},
        "fixed_summary": fixed_summary,
        "adaptive_summary": adaptive_summary,
        "checkpoint_rows": checkpoint_rows,
    });

    println!("{}", serde_json::to_string_pretty(&summary).map_err(|e| e.to_string())?);
    Ok(())
}
