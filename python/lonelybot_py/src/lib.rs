use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyIOError};
use pyo3::wrap_pyfunction;
use pyo3::Bound;

use lonelybot::analysis::{ranked_moves, ranked_moves_from_partial, analyze_state, HeuristicConfig, PlayStyle, StateAnalysis};
use lonelybot::game_theory::best_move_mcts;
use lonelybot::partial::{PartialState, PartialColumn};
use lonelybot::engine::SolitaireEngine;
use lonelybot::pruning::FullPruner;
use lonelybot::standard::StandardSolitaire;
use lonelybot::card::{Card, N_SUITS, N_RANKS};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use pyo3::types::PyDict;
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
    #[pyo3(get, set)]
    pub long_column_bonus: i32,
    #[pyo3(get, set)]
    pub chain_bonus: i32,
    #[pyo3(get, set)]
    pub aggressive_coef: i32,
    #[pyo3(get, set)]
    pub conservative_coef: i32,
    #[pyo3(get, set)]
    pub neutral_coef: i32,
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
        long_column_bonus: Option<i32>,
        chain_bonus: Option<i32>,
        aggressive_coef: Option<i32>,
        conservative_coef: Option<i32>,
        neutral_coef: Option<i32>,
    ) -> Self {
        let d = HeuristicConfig::default();
        Self {
            reveal_bonus: reveal_bonus.unwrap_or(d.reveal_bonus),
            empty_column_bonus: empty_column_bonus.unwrap_or(d.empty_column_bonus),
            early_foundation_penalty: early_foundation_penalty.unwrap_or(d.early_foundation_penalty),
            keep_king_bonus: keep_king_bonus.unwrap_or(d.keep_king_bonus),
            deadlock_penalty: deadlock_penalty.unwrap_or(d.deadlock_penalty),
            long_column_bonus: long_column_bonus.unwrap_or(d.long_column_bonus),
            chain_bonus: chain_bonus.unwrap_or(d.chain_bonus),
            aggressive_coef: aggressive_coef.unwrap_or(d.aggressive_coef),
            conservative_coef: conservative_coef.unwrap_or(d.conservative_coef),
            neutral_coef: neutral_coef.unwrap_or(d.neutral_coef),
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
            long_column_bonus: p.long_column_bonus,
            chain_bonus: p.chain_bonus,
            aggressive_coef: p.aggressive_coef,
            conservative_coef: p.conservative_coef,
            neutral_coef: p.neutral_coef,
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct GameState {
    state: PartialState,
}

fn parse_card(s: &str) -> PyResult<Card> {
    const RANKS: [&str; N_RANKS as usize] = [
        "A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K",
    ];
    const SUITS: [&str; N_SUITS as usize] = ["H", "D", "C", "S"];
    let s = s.trim();
    if s.len() < 2 {
        return Err(PyValueError::new_err("invalid card"));
    }
    let mut chars = s.chars();
    let suit_ch = chars
        .next_back()
        .ok_or_else(|| PyValueError::new_err("invalid card"))?;
    let rank_str: String = chars.collect();
    let rank = RANKS
        .iter()
        .position(|&r| r.eq_ignore_ascii_case(&rank_str))
        .ok_or_else(|| PyValueError::new_err("invalid rank"))? as u8;
    let suit = match suit_ch {
        'H' | 'h' | '♥' => 0,
        'D' | 'd' | '♦' => 1,
        'C' | 'c' | '♣' => 2,
        'S' | 's' | '♠' => 3,
        _ => return Err(PyValueError::new_err("invalid suit")),
    };
    Ok(Card::new(rank, suit))
}

fn parse_move_str(s: &str) -> PyResult<lonelybot::moves::Move> {
    let mut it = s.split_whitespace();
    let action = it
        .next()
        .ok_or_else(|| PyValueError::new_err("invalid move"))?;
    let card_str = it
        .next()
        .ok_or_else(|| PyValueError::new_err("invalid move"))?;
    let card = parse_card(card_str)?;
    match action.to_uppercase().as_str() {
        "DS" => Ok(lonelybot::moves::Move::DeckStack(card)),
        "PS" => Ok(lonelybot::moves::Move::PileStack(card)),
        "DP" => Ok(lonelybot::moves::Move::DeckPile(card)),
        "SP" => Ok(lonelybot::moves::Move::StackPile(card)),
        "R" => Ok(lonelybot::moves::Move::Reveal(card)),
        _ => Err(PyValueError::new_err("unknown move type")),
    }
}

fn parse_json_state(txt: &str) -> PyResult<PartialState> {
    let v: Value = serde_json::from_str(txt).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let draw_step = v.get("draw_step").and_then(|x| x.as_u64()).unwrap_or(1) as u8;
    let mut columns: [PartialColumn;7] = core::array::from_fn(|_| PartialColumn { hidden: Vec::new(), visible: lonelybot::standard::PileVec::new() });
    if let Some(cols) = v.get("columns").and_then(|c| c.as_array()) {
        for (i,col) in cols.iter().enumerate().take(7) {
            if let Some(hid) = col.get("hidden").and_then(|h| h.as_array()) {
                columns[i].hidden = hid
                    .iter()
                    .map(|c| {
                        if c == "unknown" || c.as_i64() == Some(-1) {
                            Ok(None)
                        } else if let Some(s) = c.as_str() {
                            parse_card(s).map(Some)
                        } else {
                            Ok(None)
                        }
                    })
                    .collect::<PyResult<Vec<Option<Card>>>>()?;
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
            if card == "unknown" || card.as_i64() == Some(-1) {
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
) -> PyResult<Vec<PyObject>> {
    let cfg = cfg.map_or_else(HeuristicConfig::default, |c| c.into());
    let moves = ranked_moves_from_partial(&state.state, get_style(style), &cfg);

    Python::with_gil(|py| {
        let mut res = Vec::new();
        for m in moves {
            let dict = PyDict::new_bound(py);
            dict.set_item("move", MovePy { mv: m.mv }.into_py(py))?;
            dict.set_item("heuristic_score", m.heuristic_score)?;
            dict.set_item("simulation_score", m.simulation_score)?;
            dict.set_item("will_block", m.will_block)?;
            let revealed: Vec<String> = m.revealed_cards.iter().map(|c| c.to_string()).collect();
            dict.set_item("revealed_cards", revealed)?;
            dict.set_item("columns_freed", m.columns_freed)?;
            dict.set_item("win_rate", m.win_rate)?;
            res.push(dict.into());
        }
        Ok(res)
    })
}

#[pyfunction]
fn best_move_py(
    state: &GameState,
    style: &str,
    cfg: Option<&HeuristicConfigPy>,
) -> PyResult<Option<MovePy>> {
    let mut rng = SmallRng::seed_from_u64(0);
    let probs = state.state.column_probabilities();
    let g = state.state.fill_unknowns_weighted(&probs, &mut rng);
    let solitaire: lonelybot::state::Solitaire = (&g).into();
    let engine: SolitaireEngine<FullPruner> = solitaire.into();
    let cfg = cfg.map_or_else(HeuristicConfig::default, |c| c.into());
    let mv = ranked_moves(&engine, &state.state, get_style(style), &cfg)
        .into_iter()
        .next();
    Ok(mv.map(|m| MovePy { mv: m.mv }))
}

#[pyfunction(signature = (state, style, n_playouts, max_depth, cfg=None))]
fn best_move_mcts_py(
    state: &GameState,
    style: &str,
    n_playouts: usize,
    max_depth: usize,
    cfg: Option<&HeuristicConfigPy>,
) -> PyResult<Option<PyObject>> {
    let mut rng = SmallRng::seed_from_u64(0);
    let cfg = cfg.map_or_else(HeuristicConfig::default, |c| c.into());
    let mv = best_move_mcts(
        &state.state,
        get_style(style),
        &cfg,
        n_playouts,
        max_depth,
        &mut rng,
    );

    Python::with_gil(|py| {
        Ok(mv.map(|m| {
            let dict = PyDict::new_bound(py);
            dict.set_item("move", MovePy { mv: m.mv }.into_py(py)).unwrap();
            dict.set_item("heuristic_score", m.heuristic_score).unwrap();
            dict.set_item("simulation_score", m.simulation_score).unwrap();
            dict.set_item("will_block", m.will_block).unwrap();
            let revealed: Vec<String> = m.revealed_cards.iter().map(|c| c.to_string()).collect();
            dict.set_item("revealed_cards", revealed).unwrap();
            dict.set_item("columns_freed", m.columns_freed).unwrap();
            dict.set_item("win_rate", m.win_rate).unwrap();
            dict.into()
        }))
    })
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

#[pyfunction]
fn collect_training_data_py(n_games: usize) -> PyResult<()> {
    lonecli::training::collect_training_data(n_games)
        .map_err(|e| PyIOError::new_err(e.to_string()))
}

#[pyfunction]
fn generate_random_state_py() -> PyResult<GameState> {
    use std::time::{SystemTime, UNIX_EPOCH};
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let mut rng = SmallRng::seed_from_u64(seed);
    let solitaire = lonelybot::state::Solitaire::deal_with_rng(&mut rng);
    let std: StandardSolitaire = (&solitaire).into();
    Ok(GameState {
        state: PartialState::from(&std),
    })
}

fn to_engine(state: &PartialState) -> SolitaireEngine<FullPruner> {
    let mut rng = SmallRng::seed_from_u64(0);
    let std = state.fill_unknowns_randomly(&mut rng);
    let sol: lonelybot::state::Solitaire = (&std).into();
    sol.into()
}

#[pyfunction]
fn legal_actions_py(state: &GameState) -> PyResult<Vec<String>> {
    let engine = to_engine(&state.state);
    Ok(engine
        .list_moves_dom()
        .iter()
        .map(|m| m.to_string())
        .collect())
}

#[pyfunction]
fn is_terminal_py(state: &GameState) -> PyResult<bool> {
    let mut engine = to_engine(&state.state);
    Ok(engine.state().is_win() || engine.list_moves_dom().is_empty())
}

#[pyfunction]
fn step_py(state: &GameState, mv: &str) -> PyResult<(GameState, bool, i32)> {
    let mut engine = to_engine(&state.state);
    let parsed = parse_move_str(mv)?;
    let valid = engine.do_move(parsed);
    if !valid {
        return Ok((state.clone(), false, -1));
    }
    let done = engine.state().is_win() || engine.list_moves_dom().is_empty();
    let reward = if engine.state().is_win() {
        100
    } else if done {
        -1
    } else {
        1
    };
    let st: StandardSolitaire = engine.state().into();
    let next_state = GameState {
        state: PartialState::from(&st),
    };
    Ok((next_state, done, reward))
}

#[pyfunction]
fn encode_observation_py(state: &GameState) -> PyResult<Vec<i32>> {
    let mut rng = SmallRng::seed_from_u64(0);
    let std = state.state.fill_unknowns_randomly(&mut rng);
    let mut obs: Vec<i32> = Vec::with_capacity(100);
    for col in &std.get_piles()[..] {
        for i in 0..13 {
            if let Some(&c) = col.get(i) {
                obs.push(i32::from(c.mask_index()) + 1);
            } else {
                obs.push(0);
            }
        }
    }
    for i in 0..7 {
        let hidden_len = std.get_hidden()[i].len();
        obs.push(hidden_len as i32);
    }
    if let Some(c) = std.get_deck().peek_current() {
        obs.push(i32::from(c.mask_index()) + 1);
    } else {
        obs.push(0);
    }
    obs.push(std.get_deck().deck_iter().len() as i32);
    Ok(obs)
}

#[pymodule]
fn lonelybot_py(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GameState>()?;
    m.add_class::<MovePy>()?;
    m.add_class::<HeuristicConfigPy>()?;
    m.add_function(wrap_pyfunction!(ranked_moves_py, m)?)?;
    m.add_function(wrap_pyfunction!(best_move_py, m)?)?;
    m.add_function(wrap_pyfunction!(best_move_mcts_py, m)?)?;
    m.add_function(wrap_pyfunction!(column_probabilities_py, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_state_py, m)?)?;
    m.add_function(wrap_pyfunction!(collect_training_data_py, m)?)?;
    m.add_function(wrap_pyfunction!(generate_random_state_py, m)?)?;
    m.add_function(wrap_pyfunction!(step_py, m)?)?;
    m.add_function(wrap_pyfunction!(legal_actions_py, m)?)?;
    m.add_function(wrap_pyfunction!(is_terminal_py, m)?)?;
    m.add_function(wrap_pyfunction!(encode_observation_py, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_card_in_json_returns_error() {
        let data = r#"{"columns":[{"hidden":["ZZ"],"visible":[]}],"deck":[]}"#;
        assert!(parse_json_state(data).is_err());
    }
}
