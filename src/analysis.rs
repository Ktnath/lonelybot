//! Heuristic evaluation and move ranking utilities.
//!
//! This module provides a very small set of expert inspired heuristics and
//! facilities to rank legal moves of a game state.

use crate::engine::SolitaireEngine;
use crate::moves::Move;
use crate::partial::PartialState;
use crate::pruning::FullPruner;
use crate::card::{Card, N_CARDS};
use crate::deck::N_PILES;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use alloc::collections::BTreeSet;

extern crate alloc;
use alloc::vec::Vec;

const LONG_COLUMN_THRESHOLD: u8 = 3;

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
    pub long_column_bonus: i32,
    pub chain_bonus: i32,
}

impl Default for HeuristicConfig {
    fn default() -> Self {
        Self {
            reveal_bonus: 5,
            empty_column_bonus: 2,
            early_foundation_penalty: -3,
            keep_king_bonus: 1,
            deadlock_penalty: -10,
            long_column_bonus: 3,
            chain_bonus: 2,
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

fn move_enables_chain(engine: &SolitaireEngine<FullPruner>, m: Move, col: u8) -> bool {
    let mut tmp: SolitaireEngine<FullPruner> = engine.state().clone().into();
    if !tmp.do_move(m) {
        return false;
    }
    let next = tmp.state().get_hidden().peek(col).copied();
    let moves = tmp.list_moves_dom();
    if let Some(card) = next {
        moves.iter().any(|mv| match *mv {
            Move::DeckPile(c)
            | Move::DeckStack(c)
            | Move::PileStack(c)
            | Move::StackPile(c)
            | Move::Reveal(c) => c == card,
        })
    } else {
        false
    }
}

/// Evaluate a move using very small heuristics.
fn evaluate_move(style: PlayStyle, engine: &SolitaireEngine<FullPruner>, m: Move, cfg: &HeuristicConfig) -> i32 {
    let hidden = engine.state().get_hidden();
    let has_empty = (0..N_PILES).any(|i| hidden.len(i as u8) == 0);
    let mut score = 0;
    match m {
        Move::Reveal(c) => {
            score += cfg.reveal_bonus;
            let col = hidden.find(c);
            let down = hidden.len(col).saturating_sub(1);
            if down > LONG_COLUMN_THRESHOLD {
                score += cfg.long_column_bonus;
            }
            if has_empty && c.is_king() {
                score += cfg.empty_column_bonus;
            }
            if move_enables_chain(engine, m, col) {
                score += cfg.chain_bonus;
            }
        }
        Move::PileStack(c) => {
            if c.rank() < 5 {
                score += cfg.early_foundation_penalty;
            }
            let col = hidden.find(c);
            let down = hidden.len(col).saturating_sub(1);
            if down > 0 && down > LONG_COLUMN_THRESHOLD {
                score += cfg.long_column_bonus;
            }
            if move_enables_chain(engine, m, col) {
                score += cfg.chain_bonus;
            }
        }
        Move::DeckPile(c) | Move::StackPile(c) => {
            if c.is_king() && has_empty {
                score += cfg.empty_column_bonus;
            }
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

