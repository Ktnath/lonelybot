use crate::{
    card::Card,
    engine::{Encode, Move, Solitaire},
    tracking::{DefaultSearchSignal, EmptySearchStats, SearchSignal, SearchStatistics},
    traverse::{traverse_game, GraphCallback, TpTable, TraverseResult},
};

extern crate alloc;
use alloc::vec::Vec;

#[derive(Clone, Copy, Debug)]
pub enum EdgeType {
    DeckPile,
    DeckStack,
    PileStack,
    PileStackReveal,
    StackPile,
    Reveal,
}

pub type Edge = (Encode, Encode, EdgeType);
pub type Graph = Vec<Edge>;

struct BuilderCallback<'a, S: SearchStatistics, T: SearchSignal> {
    graph: Graph,
    stats: &'a S,
    sign: &'a T,
    depth: usize,
    prev_enc: Encode,
    last_move: Move,
    rev_move: Option<Move>,
}

const fn get_edge_type(m: &Move, rm: &Option<Move>) -> EdgeType {
    match m {
        Move::DeckStack(_) => EdgeType::DeckStack,
        Move::PileStack(_) => {
            if rm.is_some() {
                EdgeType::PileStack
            } else {
                EdgeType::PileStackReveal
            }
        }
        Move::DeckPile(_) => EdgeType::DeckPile,
        Move::StackPile(_) => EdgeType::StackPile,
        Move::Reveal(_) => EdgeType::Reveal,
    }
}

impl<'a, S: SearchStatistics, T: SearchSignal> GraphCallback for BuilderCallback<'a, S, T> {
    fn on_win(&mut self, _: &Solitaire, rm: &Option<Move>) -> TraverseResult {
        // win state
        self.graph
            .push((self.prev_enc, !0, get_edge_type(&self.last_move, rm)));
        TraverseResult::Ok
    }

    fn on_visit(&mut self, _: &Solitaire, e: Encode) -> TraverseResult {
        if self.sign.is_terminated() {
            return TraverseResult::Halted;
        }

        self.stats.hit_a_state(self.depth);
        self.graph.push((
            self.prev_enc,
            e,
            get_edge_type(&self.last_move, &self.rev_move),
        ));

        TraverseResult::Ok
    }

    fn on_move_gen(&mut self, m: &crate::engine::MoveVec, _: Encode) {
        self.stats.hit_unique_state(self.depth, m.len());
    }

    fn on_do_move(&mut self, _: &Solitaire, m: &Move, e: Encode, rev: &Option<Move>) {
        self.last_move = *m;
        self.rev_move = *rev;
        self.prev_enc = e;
        self.depth += 1;
    }

    fn on_undo_move(&mut self, _: &Move, _: Encode) {
        self.depth -= 1;
        self.stats.finish_move(self.depth);
    }

    fn on_start(&mut self) {}

    fn on_finish(&mut self, _: &TraverseResult) {
        self.sign.search_finish();
    }
}

pub fn graph_game_with_tracking(
    g: &mut Solitaire,
    stats: &impl SearchStatistics,
    sign: &impl SearchSignal,
) -> (TraverseResult, Graph) {
    let mut tp = TpTable::default();
    let mut callback = BuilderCallback {
        graph: Graph::new(),
        stats,
        sign,
        depth: 0,
        prev_enc: g.encode(),
        last_move: Move::DeckPile(Card::FAKE),
        rev_move: None,
    };

    let finished = traverse_game(g, &mut tp, &mut callback);
    (finished, callback.graph)
}

pub fn graph_game(g: &mut Solitaire) -> (TraverseResult, Graph) {
    graph_game_with_tracking(g, &EmptySearchStats {}, &DefaultSearchSignal {})
}
