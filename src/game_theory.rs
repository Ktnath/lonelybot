//! Simplified MCTS based move selection working on partial information.

use rand::prelude::*;

use crate::analysis::{ranked_moves, HeuristicConfig, PlayStyle, RankedMove};
use crate::partial::PartialState;
use crate::engine::SolitaireEngine;
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
    let moves = ranked_moves(engine, state, style, cfg);
    // perform a very small random playout for each move
    let mut best: Option<(RankedMove, i32)> = None;
    for m in moves {
        let mut child: SolitaireEngine<FullPruner> = engine.state().clone().into();
        child.do_move(m.mv);
        let mut score = 0;
        for _ in 0..3 {
            let mut tmp: SolitaireEngine<FullPruner> = child.state().clone().into();
            let mut depth = 0;
            while depth < 10 {
                let mv = {
                    let list = tmp.list_moves_dom();
                    if list.is_empty() {
                        break;
                    }
                    *list.choose(rng).unwrap()
                };
                tmp.do_move(mv);
                depth += 1;
                if tmp.state().is_win() {
                    score += 10;
                    break;
                }
            }
        }
        if let Some((_, best_score)) = &mut best {
            if score > *best_score {
                *best_score = score;
                best = Some((m.clone(), score));
            }
        } else {
            best = Some((m.clone(), score));
        }
    }
    best.map(|b| b.0)
}

