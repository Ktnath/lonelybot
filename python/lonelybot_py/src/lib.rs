use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use lonelybot::analysis::{ranked_moves, analyze_state, HeuristicConfig, PlayStyle, StateAnalysis};
use lonelybot::game_theory::best_move_mcts;
use lonelybot::partial::{PartialState, PartialColumn};
use lonelybot::engine::SolitaireEngine;
use lonelybot::pruning::FullPruner;
use lonelybot::standard::StandardSolitaire;
use lonelybot::card::{Card, N_SUITS, N_RANKS};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use serde_json::Value;

#[pyclass]
#[derive(Clone)]
pub struct MovePy {
    mv: lonelybot::moves::Move,
}

#[pymethods]
impl MovePy {
    fn __repr__(&self) -> String {
        self.mv.to_string()
    }
}

#[pyclass]
#[derive(Clone)]
pub struct HeuristicConfigPy {
    #[pyo3(get, set)]
    pub reveal_bonus: i32,
    #[pyo3(get, set)]
    pub empty_column_bonus: i32,
    #[pyo3(get, set)]
    pub early_foundation_penalty: i32,
    #[pyo3(get, set)]
    pub keep_king_bonus: i32,
    #[pyo3(get, set)]
    pub deadlock_penalty: i32,
}

#[pymethods]
impl HeuristicConfigPy {
    #[new]
    fn new(
        reveal_bonus: Option<i32>,
        empty_column_bonus: Option<i32>,
        early_foundation_penalty: Option<i32>,
        keep_king_bonus: Option<i32>,
        deadlock_penalty: Option<i32>,
    ) -> Self {
        let d = HeuristicConfig::default();
        Self {
            reveal_bonus: reveal_bonus.unwrap_or(d.reveal_bonus),
            empty_column_bonus: empty_column_bonus.unwrap_or(d.empty_column_bonus),
            early_foundation_penalty: early_foundation_penalty.unwrap_or(d.early_foundation_penalty),
            keep_king_bonus: keep_king_bonus.unwrap_or(d.keep_king_bonus),
            deadlock_penalty: deadlock_penalty.unwrap_or(d.deadlock_penalty),
        }
    }
}

impl From<&HeuristicConfigPy> for HeuristicConfig {
    fn from(p: &HeuristicConfigPy) -> Self {
        Self {
            reveal_bonus: p.reveal_bonus,
            empty_column_bonus: p.empty_column_bonus,
            early_foundation_penalty: p.early_foundation_penalty,
            keep_king_bonus: p.keep_king_bonus,
            deadlock_penalty: p.deadlock_penalty,
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct GameState {
    state: PartialState,
}

fn parse_card(s: &str) -> PyResult<Card> {
    const RANKS: [&str; N_RANKS as usize] = ["A","2","3","4","5","6","7","8","9","10","J","Q","K"];
    const SUITS: [&str; N_SUITS as usize] = ["H","D","C","S"];
    let s = s.trim();
    if s.len() < 2 { return Err(PyValueError::new_err("invalid card")); }
    let (rank_str, suit_str) = s.split_at(s.len()-1);
    let rank = RANKS.iter().position(|&r| r.eq_ignore_ascii_case(rank_str))
        .ok_or_else(|| PyValueError::new_err("invalid rank"))? as u8;
    let suit = SUITS.iter().position(|&r| r.eq_ignore_ascii_case(suit_str))
        .ok_or_else(|| PyValueError::new_err("invalid suit"))? as u8;
    Ok(Card::new(rank, suit))
}

fn parse_json_state(txt: &str) -> PyResult<PartialState> {
    let v: Value = serde_json::from_str(txt).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let draw_step = v.get("draw_step").and_then(|x| x.as_u64()).unwrap_or(1) as u8;
    let mut columns: [PartialColumn;7] = core::array::from_fn(|_| PartialColumn { hidden: Vec::new(), visible: lonelybot::standard::PileVec::new() });
    if let Some(cols) = v.get("columns").and_then(|c| c.as_array()) {
        for (i,col) in cols.iter().enumerate().take(7) {
            if let Some(hid) = col.get("hidden").and_then(|h| h.as_array()) {
                columns[i].hidden = hid.iter().map(|c| {
                    if c == "unknown" || c.as_i64()==Some(-1) { None } else { c.as_str().map(|s| parse_card(s).unwrap()).map(Some).unwrap_or(None) }
                }).collect();
            }
            if let Some(vis) = col.get("visible").and_then(|h| h.as_array()) {
                for card in vis {
                    if let Some(s) = card.as_str() {
                        columns[i].visible.push(parse_card(s)?);
                    }
                }
            }
        }
    }
    let mut deck = Vec::new();
    if let Some(d) = v.get("deck").and_then(|d| d.as_array()) {
        for card in d {
            if card == "unknown" || card.as_i64()==Some(-1) {
                deck.push(None);
            } else if let Some(s) = card.as_str() {
                deck.push(Some(parse_card(s)?));
            }
        }
    }
    Ok(PartialState { columns, deck, draw_step })
}

#[pymethods]
impl GameState {
    #[new]
    fn new() -> Self {
        use lonelybot::shuffler::default_shuffle;
        use core::num::NonZeroU8;
        let deck = default_shuffle(0);
        let std = StandardSolitaire::new(&deck, NonZeroU8::new(1).unwrap());
        Self { state: PartialState::from(&std) }
    }

    #[staticmethod]
    fn from_json(txt: &str) -> PyResult<Self> {
        Ok(Self { state: parse_json_state(txt)? })
    }
}

fn get_style(style: &str) -> PlayStyle {
    match style {
        "aggressive" => PlayStyle::Aggressive,
        "conservative" => PlayStyle::Conservative,
        _ => PlayStyle::Neutral,
    }
}

#[pyfunction]
fn ranked_moves_py(
    state: &GameState,
    style: &str,
    cfg: Option<&HeuristicConfigPy>,
) -> PyResult<Vec<(MovePy, i32)>> {
    let mut rng = SmallRng::seed_from_u64(0);
    let g = state.state.fill_unknowns_randomly(&mut rng);
    let engine: SolitaireEngine<FullPruner> = g.into();
    let cfg = cfg.map_or_else(HeuristicConfig::default, |c| c.into());
    let moves = ranked_moves(&engine, get_style(style), &cfg);
    Ok(moves.into_iter().map(|m| (MovePy{mv:m.mv}, m.heuristic_score)).collect())
}

#[pyfunction]
fn best_move_py(
    state: &GameState,
    style: &str,
    cfg: Option<&HeuristicConfigPy>,
) -> PyResult<Option<MovePy>> {
    let mut rng = SmallRng::seed_from_u64(0);
    let g = state.state.fill_unknowns_randomly(&mut rng);
    let engine: SolitaireEngine<FullPruner> = g.into();
    let cfg = cfg.map_or_else(HeuristicConfig::default, |c| c.into());
    let mv = ranked_moves(&engine, get_style(style), &cfg).into_iter().next();
    Ok(mv.map(|m| MovePy{mv:m.mv}))
}

#[pyfunction]
fn best_move_mcts_py(
    state: &GameState,
    style: &str,
    cfg: Option<&HeuristicConfigPy>,
) -> PyResult<Option<MovePy>> {
    let mut rng = SmallRng::seed_from_u64(0);
    let mut g = state.state.fill_unknowns_randomly(&mut rng);
    let mut engine: SolitaireEngine<FullPruner> = g.into();
    let cfg = cfg.map_or_else(HeuristicConfig::default, |c| c.into());
    let mv = best_move_mcts(&mut engine, get_style(style), &cfg, &mut rng);
    Ok(mv.map(|m| MovePy{mv:m.mv}))
}

#[pyfunction]
fn column_probabilities_py(state: &GameState) -> PyResult<Vec<Vec<(String, f64)>>> {
    Ok(state.state.column_probabilities().into_iter()
        .map(|col| col.into_iter().map(|(c,p)| (c.to_string(), p)).collect()).collect())
}

#[pyfunction]
fn analyze_state_py(state: &GameState) -> PyResult<(usize, Vec<String>, usize, usize, f64)> {
    let info: StateAnalysis = analyze_state(&state.state);
    Ok((
        info.unknown_cards,
        info.remaining_cards.into_iter().map(|c| c.to_string()).collect(),
        info.blocked_columns,
        info.mobility,
        info.deadlock_risk,
    ))
}

#[pymodule]
fn lonelybot_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<GameState>()?;
    m.add_class::<MovePy>()?;
    m.add_class::<HeuristicConfigPy>()?;
    m.add_function(wrap_pyfunction!(ranked_moves_py, m)?)?;
    m.add_function(wrap_pyfunction!(best_move_py, m)?)?;
    m.add_function(wrap_pyfunction!(best_move_mcts_py, m)?)?;
    m.add_function(wrap_pyfunction!(column_probabilities_py, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_state_py, m)?)?;
    Ok(())
}
