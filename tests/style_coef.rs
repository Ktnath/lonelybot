use lonelybot::analysis::{ranked_moves, HeuristicConfig, PlayStyle};
use lonelybot::engine::SolitaireEngine;
use lonelybot::pruning::FullPruner;
use lonelybot::standard::StandardSolitaire;
use lonelybot::shuffler::default_shuffle;
use lonelybot::partial::PartialState;
use std::num::NonZeroU8;

#[test]
fn test_style_coefficient_scales_total_score() {
    let deck = default_shuffle(0);
    let game = StandardSolitaire::new(&deck, NonZeroU8::new(3).unwrap());
    let solitaire: lonelybot::state::Solitaire = (&game).into();
    let engine: SolitaireEngine<FullPruner> = solitaire.into();
    let state: PartialState = (&game).into();

    let mut cfg1 = HeuristicConfig::default();
    cfg1.neutral_coef = 1;
    let moves1 = ranked_moves(&engine, &state, PlayStyle::Neutral, &cfg1);

    let mut cfg2 = cfg1.clone();
    cfg2.neutral_coef = 2;
    let moves2 = ranked_moves(&engine, &state, PlayStyle::Neutral, &cfg2);

    assert_eq!(moves1.len(), moves2.len());
    for (a, b) in moves1.iter().zip(moves2.iter()) {
        assert_eq!(b.heuristic_score, a.heuristic_score * 2);
    }
}
