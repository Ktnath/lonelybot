use crate::{
    card::{Card, ALT_MASK, KING_MASK},
    engine::{Move, Solitaire},
};

pub struct PruneInfo {
    rev_move: Option<Move>,
    last_move: Move,
    last_draw: Option<Card>,
}

impl Default for PruneInfo {
    fn default() -> Self {
        Self {
            rev_move: None,
            last_move: Move::FAKE,
            last_draw: None,
        }
    }
}

/*
let last_draw = match m {
    Move::DeckPile(c) => Some(c),
    Move::DeckStack(_) => None,
    Move::StackPile(c) => {
        if last_draw.is_some_and(|cc| cc.go_before(&c)) {
            None
        } else {
            last_draw
        }
    }
    Move::Reveal(c) => {
        if last_draw.is_some_and(|cc| !cc.go_before(&c)) {
            continue;
        } else {
            None
        }
    }
    Move::PileStack(c) => {
        if last_draw.is_some_and(|cc| cc.rank() != c.rank() || cc.suit() ^ c.suit() != 1) {
            continue;
        } else {
            None
        }
    }
};
 */

impl PruneInfo {
    pub fn new(game: &Solitaire, prev: &PruneInfo, m: &Move) -> Self {
        Self {
            rev_move: game.get_rev_move(&m),
            last_move: *m,
            last_draw: match m {
                Move::DeckPile(c) => Some(*c),
                Move::StackPile(c) => {
                    if prev.last_draw.is_some_and(|cc| cc.go_before(c)) {
                        None
                    } else {
                        prev.last_draw
                    }
                }
                _ => None,
            },
        }
    }

    pub fn rev_move(&self) -> Option<Move> {
        return self.rev_move;
    }

    pub fn last_move(&self) -> Move {
        return self.last_move;
    }

    pub fn prune_moves(&self, game: &Solitaire) -> [u64; 5] {
        // [pile_stack - 0, deck_stack - 1, stack_pile - 2, deck_pile - 3, reveal - 4]
        let first_layer = game.get_hidden().first_layer_mask();
        let mut filter = match self.last_move {
            Move::Reveal(c) => {
                if first_layer & c.mask() > 0 {
                    [!0, !0, !KING_MASK, !KING_MASK, !KING_MASK]
                } else {
                    [0; 5]
                }
            }
            _ => [0; 5],
        };

        if let Some(last_draw) = self.last_draw {
            // pruning deck :)
            let m = last_draw.mask();
            let mm = ((m | m >> 1) & ALT_MASK) * 0b11;
            filter[0] |= !mm | m;

            // need | first layer because of this case , DP 8♠, R 10♥, DP K♠,
            // if you reveal 10 first then you forced to get K, which might prevent you from getting 8
            // if you get 8 first, you can't reveal 10, because it expects you to reveal it before
            // to get the required card to put under 8, but since it doesn't reveal anything, it's not doing it``
            filter[4] |= !((mm >> 4) | first_layer);
        }

        match self.rev_move {
            Some(Move::PileStack(c)) => filter[0] |= c.mask(),
            Some(Move::DeckStack(c)) => filter[1] |= c.mask(),
            Some(Move::StackPile(c)) => filter[2] |= c.mask(),
            Some(Move::DeckPile(c)) => filter[3] |= c.mask(),
            Some(Move::Reveal(c)) => filter[4] |= c.mask(),
            None => {}
        }

        filter
    }
}
