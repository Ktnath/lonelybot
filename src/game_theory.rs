//! Simplified MCTS based move selection working on partial information.

use rand::prelude::*;

use crate::analysis::{ranked_moves, HeuristicConfig, PlayStyle, RankedMove};
use crate::engine::SolitaireEngine;
use crate::partial::PartialState;
use crate::pruning::FullPruner;

/// Dense leaf evaluation for Monte Carlo rollouts.
///
/// The previous implementation only rewarded a rollout if it reached a full
/// win before `max_depth`. In Klondike that makes almost every short rollout
/// worth zero. This evaluator gives partial credit for genuine progress while
/// keeping terminal wins/deadlocks dominant.
fn leaf_value(engine: &SolitaireEngine<FullPruner>) -> f64 {
    if engine.state().is_win() {
        return 1000.0;
    }

    let legal = engine.list_moves_dom();
    if legal.is_empty() {
        return -250.0;
    }

    let foundation = f64::from(engine.state().get_stack().len());
    let hidden_down = f64::from(engine.state().get_hidden().total_down_cards());
    let mobility = legal.len() as f64;

    // Foundation progress is the strongest positive signal. Revealing tableau
    // cards is next, followed by a smaller mobility bonus. The absolute scale
    // is intentionally far below the terminal win value.
    foundation * 12.0 - hidden_down * 5.0 + mobility * 1.5
}

fn round_to_i32(value: f64) -> i32 {
    if value >= 0.0 {
        (value + 0.5) as i32
    } else {
        (value - 0.5) as i32
    }
}

/// Run a light Monte Carlo tree search to pick the best move.
///
/// This is still a determinization-based baseline rather than the final belief
/// solver. Unknown cards are sampled from the current partial-state model, and
/// each candidate move is evaluated over multiple sampled continuations.
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

            // A move produced by one determinization can be invalid in another.
            // Do not silently score such a world as if the move had happened.
            if !child.do_move(m.mv) {
                total -= 250.0;
                continue;
            }
            valid_playouts += 1;

            let mut tmp: SolitaireEngine<FullPruner> = child.state().clone().into();
            let mut depth = 0usize;
            while depth < max_depth && !tmp.state().is_win() {
                let list = tmp.list_moves_dom();
                let Some(mv) = list.choose(rng).copied() else {
                    break;
                };
                tmp.do_move(mv);
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

        // Dense rollout value drives the decision. Existing expert heuristics
        // only break near-ties rather than replacing simulation evidence.
        let combined = avg + f64::from(m.heuristic_score) * 0.05;
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
