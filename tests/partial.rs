use lonelybot::partial::{PartialColumn, PartialState};
use lonelybot::card::Card;
use lonelybot::standard::PileVec;
use rand::rngs::SmallRng;
use rand::SeedableRng;

#[test]
fn test_fill_unknown() {
    let col = PartialColumn { hidden: vec![None], visible: {
        let mut p = PileVec::new();
        p.push(Card::new(0,0));
        p
    }};
    let state = PartialState { columns: [col.clone(), col.clone(), col.clone(), col.clone(), col.clone(), col.clone(), col], deck: vec![None], draw_step: 1 };
    let mut rng = SmallRng::seed_from_u64(0);
    let g = state.fill_unknowns_randomly(&mut rng);
    assert_eq!(g.get_deck().len(), 24);
}
