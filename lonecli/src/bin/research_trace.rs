use core::num::NonZeroU8;
use std::env;

use lonelybot::card::{Card, N_RANKS};
use lonelybot::engine::SolitaireEngine;
use lonelybot::moves::Move;
use lonelybot::pruning::FullPruner;
use lonelybot::shuffler::default_shuffle;
use lonelybot::standard::StandardSolitaire;
use lonelybot::state::Solitaire;
use serde_json::{json, Value};

fn parse_card(s: &str) -> Result<Card, String> {
    const RANKS: [&str; N_RANKS as usize] = [
        "A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K",
    ];
    let s = s.trim();
    if s.len() < 2 {
        return Err(format!("invalid card: {s}"));
    }
    let mut chars = s.chars();
    let suit_ch = chars.next_back().ok_or_else(|| format!("invalid card: {s}"))?;
    let rank_str: String = chars.collect();
    let rank = RANKS
        .iter()
        .position(|r| r.eq_ignore_ascii_case(&rank_str))
        .ok_or_else(|| format!("invalid rank: {rank_str}"))? as u8;
    let suit = match suit_ch {
        'H' | 'h' | '♥' => 0,
        'D' | 'd' | '♦' => 1,
        'C' | 'c' | '♣' => 2,
        'S' | 's' | '♠' => 3,
        _ => return Err(format!("invalid suit: {suit_ch}")),
    };
    Ok(Card::new(rank, suit))
}

fn parse_move(s: &str) -> Result<Move, String> {
    let mut parts = s.split_whitespace();
    let kind = parts.next().ok_or_else(|| format!("invalid move: {s}"))?;
    let card = parse_card(parts.next().ok_or_else(|| format!("invalid move: {s}"))?)?;
    match kind.to_ascii_uppercase().as_str() {
        "DS" => Ok(Move::DeckStack(card)),
        "PS" => Ok(Move::PileStack(card)),
        "DP" => Ok(Move::DeckPile(card)),
        "SP" => Ok(Move::StackPile(card)),
        "R" => Ok(Move::Reveal(card)),
        _ => Err(format!("unknown move type: {kind}")),
    }
}

fn masked_snapshot(
    engine: &SolitaireEngine<FullPruner>,
    draw_step: u8,
    step: usize,
    oracle_move: &str,
) -> Value {
    let std = StandardSolitaire::from(engine.state());
    let piles = std.get_piles();
    let hidden = std.get_hidden();
    let deck = std.get_deck();

    let columns: Vec<Value> = (0..7)
        .map(|i| {
            let visible: Vec<String> = piles[i].iter().map(ToString::to_string).collect();
            json!({
                "hidden": vec!["unknown"; hidden[i].len()],
                "visible": visible,
            })
        })
        .collect();

    let deck_len = deck.deck_iter().len();
    let legal_moves: Vec<String> = engine
        .list_moves_dom()
        .iter()
        .map(ToString::to_string)
        .collect();

    json!({
        "step": step,
        "oracle_move": oracle_move,
        "true_legal_moves": legal_moves,
        "foundation_cards": engine.state().get_stack().len(),
        "hidden_down_cards": engine.state().get_hidden().total_down_cards(),
        "partial": {
            "draw_step": draw_step,
            "columns": columns,
            "deck": vec!["unknown"; deck_len],
        }
    })
}

fn main() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        return Err("usage: research_trace <seed> <draw_step> '<moves_json>'".to_string());
    }

    let seed: u64 = args[1].parse().map_err(|e| format!("invalid seed: {e}"))?;
    let draw_step: u8 = args[2]
        .parse()
        .map_err(|e| format!("invalid draw_step: {e}"))?;
    let draw_step_nz = NonZeroU8::new(draw_step).ok_or_else(|| "draw_step must be > 0".to_string())?;
    let move_strings: Vec<String> = serde_json::from_str(&args[3])
        .map_err(|e| format!("invalid moves JSON: {e}"))?;

    let deck = default_shuffle(seed);
    let game = Solitaire::new(&deck, draw_step_nz);
    let mut engine: SolitaireEngine<FullPruner> = game.into();
    let mut snapshots = Vec::with_capacity(move_strings.len());

    for (step, move_string) in move_strings.iter().enumerate() {
        snapshots.push(masked_snapshot(&engine, draw_step, step, move_string));
        let mv = parse_move(move_string)?;
        if !engine.do_move(mv) {
            return Err(format!("oracle move became illegal at step {step}: {move_string}"));
        }
    }

    println!("{}", serde_json::to_string(&snapshots).map_err(|e| e.to_string())?);
    Ok(())
}
