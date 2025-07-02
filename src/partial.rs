//! Partial information state support for Klondike
//!
//! This module allows representing hidden cards as `Option<Card>` and provides
//! helpers for filling unknown cards randomly as well as computing simple
//! probability estimates for hidden columns.

use rand::seq::SliceRandom;
use rand::Rng;

use crate::card::{Card, N_CARDS};
use crate::shuffler::CardDeck;
use crate::standard::{PileVec, StandardSolitaire};

extern crate alloc;
use alloc::vec::Vec;
use alloc::collections::BTreeSet;

/// Representation of a single tableau column with partially known cards.
#[derive(Clone, Debug)]
pub struct PartialColumn {
    /// Hidden cards from top to bottom. `None` represents an unknown card.
    pub hidden: Vec<Option<Card>>,
    /// Visible cards from bottom to top.
    pub visible: PileVec,
}

impl PartialColumn {
    /// Number of hidden cards.
    #[must_use]
    pub fn hidden_len(&self) -> usize {
        self.hidden.iter().filter(|c| c.is_none()).count()
    }
}

/// Representation of a partial Klondike state.
#[derive(Clone, Debug)]
pub struct PartialState {
    pub columns: [PartialColumn; 7],
    pub deck: Vec<Option<Card>>, // top of deck is the end of the vec
    pub draw_step: u8,
}

impl From<&StandardSolitaire> for PartialState {
    fn from(g: &StandardSolitaire) -> Self {
        let columns: [PartialColumn; 7] = core::array::from_fn(|i| PartialColumn {
            hidden: g.get_hidden()[i].iter().map(|&c| Some(c)).collect(),
            visible: g.get_piles()[i].clone(),
        });
        let deck: Vec<Option<Card>> = g.get_deck().iter().map(Some).collect();
        Self {
            columns,
            deck,
            draw_step: g.get_deck().draw_step().get(),
        }
    }
}

impl PartialState {
    /// Fill the unknown cards using a random permutation of the remaining
    /// cards. The returned `StandardSolitaire` can then be solved using the
    /// existing engine.
    #[must_use]
    pub fn fill_unknowns_randomly<R: Rng>(&self, rng: &mut R) -> StandardSolitaire {
        let mut used = BTreeSet::new();
        for col in &self.columns {
            for c in &col.visible {
                used.insert(c.mask_index());
            }
            for c in &col.hidden {
                if let Some(card) = c {
                    used.insert(card.mask_index());
                }
            }
        }
        for c in &self.deck {
            if let Some(card) = c {
                used.insert(card.mask_index());
            }
        }

        let mut remaining: Vec<Card> = (0..N_CARDS)
            .filter(|i| !used.contains(i))
            .map(Card::from_mask_index)
            .collect();
        remaining.shuffle(rng);
        let mut rem_iter = remaining.into_iter();

        // Build the card deck in the format expected by StandardSolitaire
        let mut cards = Vec::with_capacity(N_CARDS as usize);
        for col in &self.columns {
            for h in &col.hidden {
                if let Some(c) = h {
                    cards.push(*c);
                } else {
                    cards.push(rem_iter.next().unwrap());
                }
            }
            for &v in &col.visible {
                cards.push(v);
            }
        }
        for c in &self.deck {
            if let Some(card) = c.clone() {
                cards.push(card);
            } else {
                cards.push(rem_iter.next().unwrap());
            }
        }
        while cards.len() < N_CARDS as usize {
            cards.push(rem_iter.next().unwrap());
        }
        let mut array: CardDeck = [Card::DEFAULT; N_CARDS as usize];
        array.copy_from_slice(&cards);
        use core::num::NonZeroU8;
        StandardSolitaire::new(&array, NonZeroU8::new(self.draw_step).unwrap())
    }

    /// Compute simplistic probability estimates for every hidden column.
    #[must_use]
    pub fn column_probabilities(&self) -> Vec<Vec<(Card, f64)>> {
        let mut used = BTreeSet::new();
        let mut total_unknown = 0usize;
        for col in &self.columns {
            for c in &col.visible {
                used.insert(c.mask_index());
            }
            for c in &col.hidden {
                match c {
                    Some(card) => {
                        used.insert(card.mask_index());
                    }
                    None => total_unknown += 1,
                }
            }
        }
        for c in &self.deck {
            if let Some(card) = c {
                used.insert(card.mask_index());
            } else {
                total_unknown += 1;
            }
        }
        let remaining: Vec<Card> = (0..N_CARDS)
            .filter(|i| !used.contains(i))
            .map(Card::from_mask_index)
            .collect();
        let n_remaining = remaining.len() as f64;
        let mut res = Vec::new();
        for col in &self.columns {
            let n_unknown = col.hidden.iter().filter(|c| c.is_none()).count();
            let prob = if total_unknown == 0 {
                0.0
            } else {
                n_unknown as f64 / total_unknown as f64
            };
            res.push(remaining.iter().map(|&c| (c, prob / n_remaining)).collect());
        }
        res
    }
}

