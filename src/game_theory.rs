//! Simplified MCTS based move selection working on partial information.

use rand::prelude::*;

use crate::analysis::{ranked_moves, HeuristicConfig, PlayStyle, RankedMove};
use crate::engine::SolitaireEngine;
use crate::partial::PartialState;
use crate::pruning::FullPruner;
use crate::standard::StandardSolitaire;

const ROLLOUT_EXPLORATION_RATE: f64 = 0.10;
const HEURISTIC_PRIOR_WEIGHT: f64 = 4.0;

/// Dense leaf evaluation for guided Monte Carlo rollouts.
///
/// A full win remains overwhelmingly best. A terminal deadlock is always a
/// loss, but late deadlocks receive slightly less-negative values than early
/// ones so that all failed rollouts are not collapsed to the same constant.
/// Non-terminal leaves reward foundation progress, revealed tableau cards and
/// mobility.
fn leaf_value(engine: &SolitaireEngine<FullPruner>) -> f64 {
    if engine.state().is_win() {
        return 1000.0;
    }

    let legal = engine.list_moves_dom();
    let foundation = f64::from(engine.state().get_stack().len());
    let hidden_down = f64::from(engine.state().get_hidden().total_down_cards());

    if legal.is_empty() {
        // Keep every proven loss below a live position in ordinary ranges,
        // while retaining information about how far the rollout progressed.
        return -300.0 + foundation * 3.0 - hidden_down;
    }

    let mobility = legal.len() as f64;
    foundation * 10.0 - hidden_down * 4.0 + mobility * 2.0
}

fn round_to_i32(value: f64) -> i32 {
    if value >= 0.0 {
        (value + 0.5) as i32
    } else {
        (value - 0.5) as i32
    }
}

/// Choose a rollout action using the existing expert heuristic most of the
/// time, with a small amount of random exploration.
///
/// The rollout operates inside one sampled determinization. Converting that
/// determinized state back to `PartialState` is intentional here: this remains
/// a determinization baseline, not the final information-set solver.
fn choose_rollout_move<R: Rng>(
    engine: &SolitaireEngine<FullPruner>,
    style: PlayStyle,
    cfg: &HeuristicConfig,
    rng: &mut R,
) -> Option<crate::moves::Move> {
    let legal = engine.list_moves_dom();
    if legal.is_empty() {
        return None;
    }

    if rng.random_bool(ROLLOUT_EXPLORATION_RATE) {
        return legal.choose(rng).copied();
    }

    let std = StandardSolitaire::from(engine.state());
    let partial = PartialState::from(&std);
    ranked_moves(engine, &partial, style, cfg)
        .into_iter()
        .next()
        .map(|ranked| ranked.mv)
        .or_else(|| legal.choose(rng).copied())
}

/// Run a light Monte Carlo search over sampled determinizations.
///
/// Phase 0.2 changes two things compared with the previous baseline:
///  * rollouts are heuristic-guided instead of uniformly random;
///  * the heuristic root score acts as a meaningful prior rather than a tiny
///    tie-breaker.
///
/// This is still not a true belief-state / information-set MCTS. Unknown cards
/// are independently sampled for each rollout and the final Phase-1 solver will
/// replace this root model with a particle belief over legitimate histories.
#[must_use]
pub fn best_move_mcts<R: Rng>(
    state: &PartialState,
    style: PlayStyle,
    cfg: &HeuristicConfig,
    n_playouts: usize,
    max_depth: usize,
    rng: &mut R,
) -> Option<RankedMove> {
    let probs = state.column_probabilities();
    let filled = state.fill_unknowns_weighted(&probs, rng);
    let solitaire: crate::state::Solitaire = (&filled).into();
    let engine: SolitaireEngine<FullPruner> = solitaire.into();
    let mut moves = ranked_moves(&engine, state, style, cfg);

    let mut best: Option<(RankedMove, f64)> = None;

    for m in &mut moves {
        let mut total = 0.0f64;
        let mut wins = 0usize;
        let mut valid_playouts = 0usize;

        for _ in 0..n_playouts {
            let filled = state.fill_unknowns_weighted(&probs, rng);
            let solitaire_child: crate::state::Solitaire = (&filled).into();
            let mut child: SolitaireEngine<FullPruner> = solitaire_child.into();

            // A move exposed by one determinization may not exist in another.
            // This penalty is deliberately retained because it reveals where
            // the current action representation is not information-set safe.
            if !child.do_move(m.mv) {
                total -= 300.0;
                continue;
            }
            valid_playouts += 1;

            let mut tmp: SolitaireEngine<FullPruner> = child.state().clone().into();
            let mut depth = 0usize;
            while depth < max_depth && !tmp.state().is_win() {
                let Some(mv) = choose_rollout_move(&tmp, style, cfg, rng) else {
                    break;
                };
                if !tmp.do_move(mv) {
                    break;
                }
                depth += 1;
            }

            if tmp.state().is_win() {
                wins += 1;
            }
            total += leaf_value(&tmp);
        }

        let avg = if n_playouts == 0 {
            f64::from(m.heuristic_score)
        } else {
            total / n_playouts as f64
        };

        m.simulation_score = round_to_i32(avg);
        m.win_rate = if valid_playouts == 0 {
            0.0
        } else {
            wins as f64 / valid_playouts as f64
        };

        let combined = avg + f64::from(m.heuristic_score) * HEURISTIC_PRIOR_WEIGHT;
        if let Some((_, best_score)) = &mut best {
            if combined > *best_score {
                *best_score = combined;
                best = Some((m.clone(), combined));
            }
        } else {
            best = Some((m.clone(), combined));
        }
    }

    best.map(|b| b.0)
}
