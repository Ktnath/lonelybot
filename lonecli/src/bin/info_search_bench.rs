use core::num::NonZeroU8;
use std::env;
use std::time::Instant;

use lonelybot::belief::ResearchGame;
use lonelybot::belief_agent::ParticleBeliefConfig;
use lonelybot::convert::convert_moves;
use lonelybot::info_search::{InformationSetDecision, InformationSetSearchConfig};
use lonelybot::shuffler::default_shuffle;
use lonelybot::solver::solve;
use lonelybot::standard::{Pos, StandardMove, StandardSolitaire};
use lonelybot::state::Solitaire;
use rand::{rngs::SmallRng, SeedableRng};
use serde_json::{json, Value};

const SEARCH_BUDGETS: [usize; 3] = [256, 1024, 4096];
const ROOT_EVAL_PARTICLES: usize = 2048;

#[derive(Default)]
struct SearchAggregate {
    n: usize,
    oracle_agree: usize,
    root_match: usize,
    invalid_transitions: usize,
    decision_ms: Vec<f64>,
    nodes: Vec<f64>,
    obs_children: Vec<f64>,
    branching_edges: Vec<f64>,
    max_obs_children: Vec<f64>,
    max_depth: Vec<f64>,
    chosen_root_eval_value: Vec<f64>,
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

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn knowledge_stage(unknown: usize) -> usize {
    if unknown >= 36 {
        0
    } else if unknown >= 18 {
        1
    } else {
        2
    }
}

fn stage_name(stage: usize) -> &'static str {
    match stage {
        0 => "early",
        1 => "mid",
        _ => "late",
    }
}

fn search_json(decision: &InformationSetDecision, elapsed_ms: f64, root_eval_value: f64) -> Value {
    let chosen = &decision.root_actions[decision.chosen_index];
    json!({
        "simulations": decision.diagnostics.simulations,
        "chosen_action": action_string(&decision.chosen_action),
        "chosen_visits": chosen.visits,
        "chosen_mean_value": chosen.mean_value,
        "chosen_observation_children": chosen.observation_children,
        "root_action_count": decision.root_actions.len(),
        "nodes_created": decision.diagnostics.nodes_created,
        "total_observation_children": decision.diagnostics.total_observation_children,
        "branching_action_edges": decision.diagnostics.branching_action_edges,
        "max_observation_children": decision.diagnostics.max_observation_children,
        "max_tree_depth_reached": decision.diagnostics.max_tree_depth_reached,
        "invalid_public_transitions": decision.diagnostics.invalid_public_transitions,
        "rollout_repeat_breaks": decision.diagnostics.rollout_repeat_breaks,
        "root_eval_value": root_eval_value,
        "decision_ms": elapsed_ms,
    })
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
    let tree_depth: usize = args
        .get(3)
        .map_or(Ok(8), |s| s.parse())
        .map_err(|e| format!("invalid tree depth: {e}"))?;
    let rollout_depth: usize = args
        .get(4)
        .map_or(Ok(20), |s| s.parse())
        .map_err(|e| format!("invalid rollout depth: {e}"))?;

    let draw_step = NonZeroU8::new(3).unwrap();
    let root_cfg = ParticleBeliefConfig {
        rollout_depth: 32,
        ..Default::default()
    };

    let mut aggregates: [SearchAggregate; 3] =
        core::array::from_fn(|_| SearchAggregate::default());
    let mut checkpoint_rows = Vec::new();
    let mut stability_256_1024 = 0usize;
    let mut stability_1024_4096 = 0usize;
    let mut stability_256_4096 = 0usize;
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
                    let eval_seed = 0x1357_9BDF_2468_ACE0u64
                        ^ seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        ^ (step_in_seed as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
                    let search_seed = eval_seed ^ 0xA11C_E5EA_7C00_0001u64;
                    let oracle_string = action_string(&action);

                    // Held-out root evaluator: one common particle sample scores
                    // every root action, including whatever the tree search picks.
                    let mut root_rng = SmallRng::seed_from_u64(eval_seed);
                    let root_decision = research
                        .belief()
                        .particle_decision(ROOT_EVAL_PARTICLES, &root_cfg, &mut root_rng)
                        .map_err(|e| format!("seed {seed} step {step_in_seed} root eval: {e:?}"))?;
                    let root_choice = action_string(&root_decision.chosen_action);
                    let root_selected_value = root_decision.actions[root_decision.chosen_index].mean_value;

                    let mut fixed_rows = Vec::new();
                    let mut choices = Vec::new();

                    for (idx, simulations) in SEARCH_BUDGETS.iter().copied().enumerate() {
                        let cfg = InformationSetSearchConfig {
                            simulations,
                            tree_depth,
                            rollout_depth,
                            ..Default::default()
                        };
                        let mut rng = SmallRng::seed_from_u64(search_seed);
                        let start = Instant::now();
                        let decision = research
                            .belief()
                            .information_set_decision(&cfg, &mut rng)
                            .map_err(|e| format!("seed {seed} step {step_in_seed} info {simulations}: {e:?}"))?;
                        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                        let chosen_string = action_string(&decision.chosen_action);
                        let chosen_root_eval_value = root_decision
                            .actions
                            .iter()
                            .find(|s| s.action == decision.chosen_action)
                            .map_or(-1.0, |s| s.mean_value);

                        let agg = &mut aggregates[idx];
                        agg.n += 1;
                        agg.oracle_agree += usize::from(chosen_string == oracle_string);
                        agg.root_match += usize::from(chosen_string == root_choice);
                        agg.invalid_transitions += decision.diagnostics.invalid_public_transitions;
                        agg.decision_ms.push(elapsed);
                        agg.nodes.push(decision.diagnostics.nodes_created as f64);
                        agg.obs_children.push(decision.diagnostics.total_observation_children as f64);
                        agg.branching_edges.push(decision.diagnostics.branching_action_edges as f64);
                        agg.max_obs_children.push(decision.diagnostics.max_observation_children as f64);
                        agg.max_depth.push(decision.diagnostics.max_tree_depth_reached as f64);
                        agg.chosen_root_eval_value.push(chosen_root_eval_value);

                        choices.push(chosen_string);
                        fixed_rows.push(search_json(&decision, elapsed, chosen_root_eval_value));
                    }

                    stability_256_1024 += usize::from(choices[0] == choices[1]);
                    stability_1024_4096 += usize::from(choices[1] == choices[2]);
                    stability_256_4096 += usize::from(choices[0] == choices[2]);

                    checkpoint_rows.push(json!({
                        "seed": seed,
                        "step_in_seed": step_in_seed,
                        "knowledge_stage": stage_name(stage),
                        "known_initial_slots": research.belief().known_slot_count(),
                        "unknown_initial_slots": unknown,
                        "public_action_count": candidates.len(),
                        "oracle_next_action": oracle_string,
                        "oracle_representable": representable,
                        "root_baseline_action": root_choice,
                        "root_baseline_value": root_selected_value,
                        "search": fixed_rows,
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
    let search_summary: Vec<Value> = SEARCH_BUDGETS
        .iter()
        .enumerate()
        .map(|(idx, &simulations)| {
            let agg = &aggregates[idx];
            json!({
                "simulations": simulations,
                "checkpoints": agg.n,
                "oracle_agreement_rate": if agg.n == 0 {0.0} else {agg.oracle_agree as f64 / agg.n as f64},
                "root_baseline_match_rate": if agg.n == 0 {0.0} else {agg.root_match as f64 / agg.n as f64},
                "invalid_public_transitions": agg.invalid_transitions,
                "decision_ms_mean": mean(&agg.decision_ms),
                "decision_ms_p50": percentile(&agg.decision_ms, 0.50),
                "decision_ms_p95": percentile(&agg.decision_ms, 0.95),
                "nodes_created_mean": mean(&agg.nodes),
                "observation_children_mean": mean(&agg.obs_children),
                "branching_action_edges_mean": mean(&agg.branching_edges),
                "max_observation_children_mean": mean(&agg.max_obs_children),
                "max_tree_depth_mean": mean(&agg.max_depth),
                "chosen_root_eval_value_mean": mean(&agg.chosen_root_eval_value),
            })
        })
        .collect();

    let summary = json!({
        "agent": "information_set_observation_tree_v1",
        "draw_step": 3,
        "tree_depth": tree_depth,
        "rollout_depth": rollout_depth,
        "search_budgets": SEARCH_BUDGETS,
        "root_eval_particles": ROOT_EVAL_PARTICLES,
        "seeds_requested": n_seeds,
        "solved_seeds_entered": solved_seeds,
        "checkpoints": n_cp,
        "knowledge_stage_counts": {"early":stage_counts[0], "mid":stage_counts[1], "late":stage_counts[2]},
        "scanned_standard_actions": scanned_actions,
        "oracle_missing_from_public_actions": oracle_missing,
        "stability_256_1024": if n_cp == 0 {0.0} else {stability_256_1024 as f64 / n_cp as f64},
        "stability_1024_4096": if n_cp == 0 {0.0} else {stability_1024_4096 as f64 / n_cp as f64},
        "stability_256_4096": if n_cp == 0 {0.0} else {stability_256_4096 as f64 / n_cp as f64},
        "search_summary": search_summary,
        "checkpoint_rows": checkpoint_rows,
    });

    println!("{}", serde_json::to_string_pretty(&summary).map_err(|e| e.to_string())?);
    Ok(())
}
