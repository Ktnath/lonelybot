//! Simplified MCTS based move selection working on partial information.

use rand::prelude::*;

use crate::analysis::{ranked_moves, HeuristicConfig, PlayStyle, RankedMove};
use crate::engine::SolitaireEngine;
use crate::pruning::FullPruner;
use crate::partial::PartialState;

/// Run a light Monte Carlo tree search to pick the best move.
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
        let mut total = 0f64;
        let mut wins = 0usize;

        // Monte Carlo playouts with weighted unknowns
        for _ in 0..n_playouts {
            let filled = state.fill_unknowns_weighted(&probs, rng);
            let solitaire_child: crate::state::Solitaire = (&filled).into();
            let mut child: SolitaireEngine<FullPruner> = solitaire_child.into();
            child.do_move(m.mv);

            let mut tmp: SolitaireEngine<FullPruner> = child.state().clone().into();
            let mut depth = 0usize;
            while depth < max_depth {
                let list = tmp.list_moves_dom();
                if list.is_empty() {
                    break;
                }
                let mv = *list.choose(rng).unwrap();
                tmp.do_move(mv);
                depth += 1;
                if tmp.state().is_win() {
                    wins += 1;
                    total += 10.0;
                    break;
                }
            }
        }

        let avg = if n_playouts == 0 { 0.0 } else { total / n_playouts as f64 };
        // round() may not be available in core for no_std; emulate simple rounding
        m.simulation_score = (avg + 0.5) as i32;
        m.win_rate = if n_playouts == 0 { 0.0 } else { wins as f64 / n_playouts as f64 };
        if let Some((_, best_score)) = &mut best {
            if avg > *best_score {
                *best_score = avg;
                best = Some((m.clone(), avg));
            }
        } else {
            best = Some((m.clone(), avg));
        }
    }

    best.map(|b| b.0)
}
