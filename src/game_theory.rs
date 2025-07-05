//! Simplified MCTS based move selection working on partial information.

use rand::prelude::*;

use crate::analysis::{ranked_moves, HeuristicConfig, PlayStyle, RankedMove};
use crate::engine::SolitaireEngine;
use crate::partial::PartialState;
use crate::pruning::FullPruner;

/// Run a light Monte Carlo tree search to pick the best move.
#[must_use]
pub fn best_move_mcts<R: Rng>(
    engine: &mut SolitaireEngine<FullPruner>,
    state: &PartialState,
    style: PlayStyle,
    cfg: &HeuristicConfig,
    rng: &mut R,
) -> Option<RankedMove> {
    let filled = state.fill_unknowns_randomly(rng);
    let solitaire: crate::state::Solitaire = (&filled).into();
    let engine: SolitaireEngine<FullPruner> = solitaire.into();
    let mut moves = ranked_moves(&engine, state, style, cfg);

    let probs = state.column_probabilities();
    let mut best: Option<(RankedMove, f64)> = None;

    for m in &mut moves {
        let mut total = 0f64;

        for _ in 0..3 {
            let filled = state.fill_unknowns_weighted(&probs, rng);
            let solitaire_child: crate::state::Solitaire = (&filled).into();
            let mut child: SolitaireEngine<FullPruner> = solitaire_child.into();
            child.do_move(m.mv);

            let mut score = 0;
            for _ in 0..3 {
                let mut tmp: SolitaireEngine<FullPruner> = child.state().clone().into();
                let mut depth = 0;

                while depth < 10 {
                    let list = tmp.list_moves_dom();
                    if list.is_empty() {
                        break;
                    }
                    let mv = *list.choose(rng).unwrap();
                    tmp.do_move(mv);
                    depth += 1;

                    if tmp.state().is_win() {
                        score += 10;
                        break;
                    }
                }
            }

            total += score as f64;
        }

        let avg = total / 3.0;
        m.win_rate = avg;

        if let Some((_, ref mut best_score)) = best {
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
