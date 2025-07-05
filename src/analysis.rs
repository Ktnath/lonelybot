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
    /// Penalty applied when a move leaves the engine with no mobility.
    pub deadlock_penalty: i32,
    pub long_column_bonus: i32,
    pub chain_bonus: i32,
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
            long_column_bonus: 3,
            chain_bonus: 2,
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
    pub unknown_cards: usize,
    pub remaining_cards: Vec<Card>,
    pub blocked_columns: usize,
    pub mobility: usize,
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

fn evaluate_move(
    style: PlayStyle,
    engine: &SolitaireEngine<FullPruner>,
    state: &PartialState,
    m: Move,
    cfg: &HeuristicConfig,
) -> i32 {
    let coeff = match style {
        PlayStyle::Aggressive => cfg.aggressive_coef,
        PlayStyle::Conservative => cfg.conservative_coef,
        PlayStyle::Neutral => cfg.neutral_coef,
    };

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
                score += cfg.early_foundation_penalty * coeff;
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
            if c.is_king() && hidden.len(6) == 0 {
                score += cfg.keep_king_bonus * coeff;
            }
        }
        _ => {}
    }

    // Penalize moves that immediately lead to no available follow-up moves.
    // This prevents ranking moves highly if they would dead-end the game state.
    let mut next: SolitaireEngine<FullPruner> = engine.state().clone().into();
    if next.do_move(m) && next.list_moves_dom().is_empty() {
        score += cfg.deadlock_penalty;
    }

    // Bonus/penalité par style
    score += match style {
        PlayStyle::Aggressive => 1,
        PlayStyle::Conservative => -1,
        PlayStyle::Neutral => 0,
    };

    // Poids de probabilité
    let probabilities = state.column_probabilities();
    let prob = match m {
        Move::Reveal(c) => {
            let idx = hidden.find(c) as usize;
            if state.columns[idx].hidden.iter().any(|h| *h == Some(c)) {
                1.0
            } else {
                probabilities
                    .get(idx)
                    .and_then(|col_probs| {
                        col_probs
                            .iter()
                            .find_map(|(card, p)| if *card == c { Some(*p) } else { None })
                    })
                    .unwrap_or(0.0)
            }
        }
        _ => 1.0,
    };

    // round() may not be available in core for no_std; emulate simple rounding
    ((score as f64) * prob + 0.5) as i32
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
    state: &PartialState,
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

            let heuristic_score = evaluate_move(style, engine, state, m, cfg);

            RankedMove {
                mv: m,
                heuristic_score,
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

    let mut rng = SmallRng::seed_from_u64(0);
    let filled = state.fill_unknowns_randomly(&mut rng);
    let solitaire: crate::state::Solitaire = (&filled).into();
    let engine: SolitaireEngine<FullPruner> = solitaire.into();
    let mobility = engine.list_moves_dom().len();

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
