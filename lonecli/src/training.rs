use lonelybot::analysis::{ranked_moves, HeuristicConfig, PlayStyle};
use lonelybot::engine::SolitaireEngine;
use lonelybot::partial::PartialState;
use lonelybot::pruning::FullPruner;
use lonelybot::state::Solitaire;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde_json::{json, to_string, Value};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::collections::HashSet;

fn state_to_json(state: &PartialState) -> Value {
    let columns: Vec<Value> = state
        .columns
        .iter()
        .map(|c| {
            json!({
                "hidden": c
                    .hidden
                    .iter()
                    .map(|o| o.map(|x| x.to_string()).unwrap_or_else(|| "unknown".into()))
                    .collect::<Vec<_>>(),
                "visible": c
                    .visible
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>(),
            })
        })
        .collect();
    let deck: Vec<String> = state
        .deck
        .iter()
        .map(|o| o.map(|x| x.to_string()).unwrap_or_else(|| "unknown".into()))
        .collect();
    json!({
        "draw_step": state.draw_step,
        "columns": columns,
        "deck": deck,
    })
}

pub fn collect_training_data(n_games: usize) -> std::io::Result<()> {
    let file = File::create("training_data.jsonl")?;
    let mut writer = BufWriter::new(file);
    let mut rng = SmallRng::seed_from_u64(0);

    for i in 0..n_games {
        if i % 1000 == 0 && i > 0 {
            eprintln!("generated {}/{} games", i, n_games);
        }
        let solitaire = Solitaire::deal_with_rng(&mut rng);
        let mut engine: SolitaireEngine<FullPruner> = solitaire.into();
        let mut seen = HashSet::new();
        let mut turn = 0usize;
        while !engine.state().is_win() {
            let enc = engine.state().encode();
            if !seen.insert(enc) {
                break;
            }
            let state = PartialState::from_blind(engine.state());
            let moves = engine.list_moves_dom();
            if moves.is_empty() {
                break;
            }
            let ranked = ranked_moves(&engine, &state, PlayStyle::Neutral, &HeuristicConfig::default());
            let mv = ranked.first().map(|m| m.mv).unwrap_or(moves[0]);
            engine.do_move(mv);
            let record = json!({
                "turn": turn,
                "partial_state": state_to_json(&state),
                "available_moves": moves.iter().map(|m| m.to_string()).collect::<Vec<_>>(),
                "selected_move": mv.to_string(),
                "win": engine.state().is_win(),
                "style": "neutral",
            });
            writer.write_all(to_string(&record)?.as_bytes())?;
            writer.write_all(b"\n")?;
            turn += 1;
        }
    }

    writer.flush()
}

