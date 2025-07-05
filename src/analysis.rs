//! Heuristic evaluation and move ranking utilities.
//!
//! This module provides a very small set of expert inspired heuristics and
//! facilities to rank legal moves of a game state.

use crate::engine::SolitaireEngine;
use crate::moves::Move;
use crate::partial::PartialState;
use crate::pruning::FullPruner;
use crate::card::{Card, N_CARDS};
use crate::state::{Solitaire, ExtraInfo};
use crate::deck::N_PILES;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use alloc::collections::BTreeSet;

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
    pub aggressive_coef: i32,
    pub conservative_coef: i32,
    pub neutral_coef: i32,
}

impl Default for HeuristicConfig {
    fn default() -> Self {
        Self {
            reveal_bonus: 5,
            empty_column_bonus: 2,
            early_foundation_penalty: -3,
            keep_king_bonus: 1,
            deadlock_penalty: -10,
            aggressive_coef: 1,
            conservative_coef: 1,
            neutral_coef: 1,
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
    pub revealed_cards: Vec<Card>,
    pub columns_freed: usize,
    pub win_rate: f64,
}

/// Basic information about a partial game state.
#[derive(Clone, Debug)]
pub struct StateAnalysis {
    /// Number of cards that are still unknown.
    pub unknown_cards: usize,
    /// Cards that are not present in the current information set.
    pub remaining_cards: Vec<Card>,
    /// Number of tableau columns where the top card cannot currently move.
    pub blocked_columns: usize,
    /// Number of legal moves in a sampled filled state.
    pub mobility: usize,
    /// Heuristic estimation of deadlock risk in \[0.0,1.0\].
    pub deadlock_risk: f64,
}

/// Evaluate a move using very small heuristics.
fn evaluate_move(style: PlayStyle, engine: &SolitaireEngine<FullPruner>, m: Move, cfg: &HeuristicConfig) -> i32 {
    let coeff = match style {
        PlayStyle::Aggressive => cfg.aggressive_coef,
        PlayStyle::Conservative => cfg.conservative_coef,
        PlayStyle::Neutral => cfg.neutral_coef,
    };
    let mut score = 0;
    match m {
        Move::Reveal(_) => score += cfg.reveal_bonus * coeff,
        Move::PileStack(c) => {
            if c.rank() < 5 {
                score += cfg.early_foundation_penalty * coeff;
            }
        }
        Move::DeckPile(c) | Move::StackPile(c) => {
            if c.is_king() && engine.state().get_hidden().len(6) == 0 {
                score += cfg.keep_king_bonus * coeff;
            }
        }
        _ => {}
    }
    score
}

fn count_empty_columns(game: &Solitaire) -> usize {
    let piles = game.compute_visible_piles();
    let hidden = game.get_hidden();
    let mut count = 0usize;
    for i in 0..N_PILES {
        if piles[i as usize].is_empty() && hidden.len(i) == 0 {
            count += 1;
        }
    }
    count
}

/// Return a sorted list of legal moves with heuristic scores.
#[must_use]
pub fn ranked_moves(
    engine: &SolitaireEngine<FullPruner>,
    style: PlayStyle,
    cfg: &HeuristicConfig,
) -> Vec<RankedMove> {
    let moves = engine.list_moves_dom();
    let base_empty = count_empty_columns(engine.state());
    let mut res: Vec<RankedMove> = moves
        .iter()
        .map(|&m| {
            let mut st = engine.state().clone();
            let (_, (_, extra)) = st.do_move(m);
            let columns_freed = count_empty_columns(&st).saturating_sub(base_empty);
            let revealed_cards = match extra {
                ExtraInfo::Card(c) => alloc::vec![c],
                _ => Vec::new(),
            };
            RankedMove {
                mv: m,
                heuristic_score: evaluate_move(style, engine, m, cfg),
                simulation_score: 0,
                will_block: false,
                revealed_cards,
                columns_freed,
                win_rate: 0.0,
            }
        })
        .collect();
    res.sort_by_key(|m| -m.heuristic_score);
    res
}

/// Analyze a partial state and return basic metrics.
#[must_use]
pub fn analyze_state(state: &PartialState) -> StateAnalysis {
    let mut used = BTreeSet::new();
    let mut unknown = 0usize;
    for col in &state.columns {
        for c in &col.visible {
            used.insert(c.mask_index());
        }
        for c in &col.hidden {
            match c {
                Some(card) => {
                    used.insert(card.mask_index());
                }
                None => unknown += 1,
            }
        }
    }
    for c in &state.deck {
        match c {
            Some(card) => {
                used.insert(card.mask_index());
            }
            None => unknown += 1,
        }
    }
    let remaining_cards: Vec<Card> = (0..N_CARDS)
        .filter(|i| !used.contains(i))
        .map(Card::from_mask_index)
        .collect();

    // Compute mobility using a deterministic fill of unknowns
    let mut rng = SmallRng::seed_from_u64(0);
    let filled = state.fill_unknowns_randomly(&mut rng);
    let solitaire: crate::state::Solitaire = (&filled).into();
    let engine: SolitaireEngine<FullPruner> = solitaire.into();
    let mobility = engine.list_moves_dom().len();

    // Blocked columns heuristics
    let mut blocked = 0usize;
    for (i, col) in state.columns.iter().enumerate() {
        let top = col.visible.last().copied();
        if top.is_none() {
            if !col.hidden.is_empty() {
                blocked += 1;
            }
            continue;
        }
        let top = top.unwrap();
        let mut movable = false;
        for (j, other) in state.columns.iter().enumerate() {
            if i == j {
                continue;
            }
            let dest = other.visible.last().copied();
            if top.go_after(dest) {
                movable = true;
                break;
            }
            if dest.is_none() && top.is_king() {
                movable = true;
                break;
            }
        }
        if !movable {
            blocked += 1;
        }
    }
    let deadlock_risk = if mobility == 0 && unknown == 0 {
        1.0
    } else {
        blocked as f64 / state.columns.len() as f64
    };

    StateAnalysis {
        unknown_cards: unknown,
        remaining_cards,
        blocked_columns: blocked,
        mobility,
        deadlock_risk,
    }
}

