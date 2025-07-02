use lonelybot::analysis::{ranked_moves, HeuristicConfig, PlayStyle};
use lonelybot::engine::SolitaireEngine;
use lonelybot::pruning::FullPruner;
use lonelybot::standard::StandardSolitaire;
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
struct GameState {
    inner: StandardSolitaire,
}

#[pymethods]
impl GameState {
    #[new]
    fn new() -> Self {
        use lonelybot::shuffler::default_shuffle;
        use core::num::NonZeroU8;
        let deck = default_shuffle(0);
        Self { inner: StandardSolitaire::new(&deck, NonZeroU8::new(1).unwrap()) }
    }
}

#[pyfunction]
fn py_ranked_moves(state: &GameState, style: &str) -> PyResult<Vec<(String, i32)>> {
    let style = match style {
        "aggressive" => PlayStyle::Aggressive,
        "conservative" => PlayStyle::Conservative,
        _ => PlayStyle::Neutral,
    };
    let mut engine: SolitaireEngine<FullPruner> = state.inner.clone().into();
    let moves = ranked_moves(&engine, style, &HeuristicConfig::default());
    Ok(moves
        .into_iter()
        .map(|m| (m.mv.to_string(), m.heuristic_score))
        .collect())
}

#[pymodule]
fn lonelybot_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<GameState>()?;
    m.add_function(wrap_pyfunction!(py_ranked_moves, m)?)?;
    Ok(())
}

