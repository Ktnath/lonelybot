//! Heuristic evaluation and move ranking utilities.
//!
//! This module provides a very small set of expert inspired heuristics and
//! facilities to rank legal moves of a game state.

use crate::engine::SolitaireEngine;
use crate::moves::Move;
use crate::pruning::FullPruner;

extern crate alloc;
use alloc::vec::Vec;

/// Player style used to influence the evaluation of moves.
#[derive(Clone, Copy, Debug)]
pub enum PlayStyle {
    Conservative,
    Neutral,
    Aggressive,
}

/// Weights for the different heuristics used during evaluation.
#[derive(Clone, Debug)]
pub struct HeuristicConfig {
    pub reveal_bonus: i32,
    pub empty_column_bonus: i32,
    pub early_foundation_penalty: i32,
    pub keep_king_bonus: i32,
    pub deadlock_penalty: i32,
}

impl Default for HeuristicConfig {
    fn default() -> Self {
        Self {
            reveal_bonus: 5,
            empty_column_bonus: 2,
            early_foundation_penalty: -3,
            keep_king_bonus: 1,
            deadlock_penalty: -10,
        }
    }
}

/// Result of a ranked move.
#[derive(Clone, Debug)]
pub struct RankedMove {
    pub mv: Move,
    pub heuristic_score: i32,
    pub simulation_score: i32,
    pub will_block: bool,
}

/// Evaluate a move using very small heuristics.
fn evaluate_move(style: PlayStyle, engine: &SolitaireEngine<FullPruner>, m: Move, cfg: &HeuristicConfig) -> i32 {
    let mut score = 0;
    match m {
        Move::Reveal(_) => score += cfg.reveal_bonus,
        Move::PileStack(c) => {
            if c.rank() < 5 {
                score += cfg.early_foundation_penalty;
            }
        }
        Move::DeckPile(c) | Move::StackPile(c) => {
            if c.is_king() && engine.state().get_hidden().len(6) == 0 {
                score += cfg.keep_king_bonus;
            }
        }
        _ => {}
    }

    // style modifier
    score += match style {
        PlayStyle::Aggressive => 1,
        PlayStyle::Conservative => -1,
        PlayStyle::Neutral => 0,
    };
    score
}

/// Return a sorted list of legal moves with heuristic scores.
#[must_use]
pub fn ranked_moves(
    engine: &SolitaireEngine<FullPruner>,
    style: PlayStyle,
    cfg: &HeuristicConfig,
) -> Vec<RankedMove> {
    let moves = engine.list_moves_dom();
    let mut res: Vec<RankedMove> = moves
        .iter()
        .map(|&m| RankedMove {
            mv: m,
            heuristic_score: evaluate_move(style, engine, m, cfg),
            simulation_score: 0,
            will_block: false,
        })
        .collect();
    res.sort_by_key(|m| -m.heuristic_score);
    res
}

