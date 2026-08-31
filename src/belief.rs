//! Belief-state support for dynamic imperfect-information Klondike research.
//!
//! The legacy `PartialState` is useful for static determinization, but it does
//! not preserve a complete mid-game public state. `BeliefState` instead keeps
//! a public snapshot plus constraints on the *initial deal slots*. Particles
//! are sampled as complete initial deals consistent with every legitimately
//! observed card, then the public action history is replayed. This preserves
//! foundations, tableau structure, stock cursor/waste state, and all previous
//! observations without ever storing unrevealed card identities in the agent
//! state.

extern crate alloc;

use alloc::vec::Vec;
use core::num::NonZeroU8;

use rand::seq::SliceRandom;
use rand::Rng;

use crate::card::{Card, N_CARDS, N_SUITS};
use crate::deck::{N_PILES, N_PILE_CARDS};
use crate::shuffler::CardDeck;
use crate::standard::{InvalidMove, PileVec, Pos, StandardMove, StandardSolitaire};

const N_SLOTS: usize = N_CARDS as usize;
const STOCK_START: usize = N_PILE_CARDS as usize;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublicColumn {
    pub hidden_count: u8,
    pub visible: PileVec,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublicSnapshot {
    pub columns: [PublicColumn; N_PILES as usize],
    /// Number of cards currently in each foundation suit. A value of 0 means
    /// the ace of that suit is the next stackable card.
    pub foundation_counts: [u8; N_SUITS as usize],
    pub deck_len: u8,
    /// Public stock/waste cursor. Zero means no current waste card.
    pub deck_cursor: u8,
    pub waste_top: Option<Card>,
    pub draw_step: u8,
}

impl PublicSnapshot {
    fn from_truth(game: &StandardSolitaire) -> Self {
        let columns = core::array::from_fn(|i| PublicColumn {
            #[allow(clippy::cast_possible_truncation)]
            hidden_count: game.get_hidden()[i].len() as u8,
            visible: game.get_piles()[i].clone(),
        });
        let foundation_counts = core::array::from_fn(|s| game.get_stack().get(s as u8));
        Self {
            columns,
            foundation_counts,
            deck_len: game.get_deck().len(),
            deck_cursor: game.get_deck().get_offset(),
            waste_top: game.get_deck().peek_current(),
            draw_step: game.get_deck().draw_step().get(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BeliefError {
    InvalidDrawStep,
    InvalidPublicAction,
    InconsistentObservation,
    DuplicateObservedCard,
    ParticleReplayFailed,
}

impl From<InvalidMove> for BeliefError {
    fn from(_: InvalidMove) -> Self {
        Self::InvalidPublicAction
    }
}

/// Agent-visible belief state.
///
/// `known_slots[i]` is populated only when the identity of initial deal slot
/// `i` has legitimately become visible. Unknown initial slots remain `None`.
/// This lets us preserve all observations across stock cycles and tableau
/// reveals without ever copying hidden identities from the true environment.
pub struct BeliefState {
    draw_step: NonZeroU8,
    known_slots: [Option<Card>; N_SLOTS],
    hidden_slots: [Vec<u8>; N_PILES as usize],
    deck_slots: Vec<u8>,
    deck_cursor: u8,
    history: Vec<StandardMove>,
    snapshot: PublicSnapshot,
}

impl BeliefState {
    fn new_from_initial_public(game: &StandardSolitaire) -> Result<Self, BeliefError> {
        let draw_step = game.get_deck().draw_step();
        let mut known_slots = [None; N_SLOTS];
        let hidden_slots: [Vec<u8>; N_PILES as usize] = core::array::from_fn(|pile| {
            let base = pile * (pile + 1) / 2;
            (0..pile).map(|j| (base + j) as u8).collect()
        });

        // At the initial deal exactly one tableau card per column is public.
        for pile in 0..N_PILES as usize {
            let top_slot = pile * (pile + 1) / 2 + pile;
            let Some(card) = game.get_piles()[pile].first().copied() else {
                return Err(BeliefError::InconsistentObservation);
            };
            known_slots[top_slot] = Some(card);
        }

        let deck_slots = (STOCK_START..N_SLOTS).map(|i| i as u8).collect();
        let snapshot = PublicSnapshot::from_truth(game);

        Ok(Self {
            draw_step,
            known_slots,
            hidden_slots,
            deck_slots,
            deck_cursor: 0,
            history: Vec::new(),
            snapshot,
        })
    }

    #[must_use]
    pub const fn snapshot(&self) -> &PublicSnapshot {
        &self.snapshot
    }

    #[must_use]
    pub fn history(&self) -> &[StandardMove] {
        &self.history
    }

    #[must_use]
    pub fn known_slot_count(&self) -> usize {
        self.known_slots.iter().filter(|c| c.is_some()).count()
    }

    #[must_use]
    pub fn unknown_slot_count(&self) -> usize {
        N_SLOTS - self.known_slot_count()
    }

    #[must_use]
    pub const fn deck_cursor(&self) -> u8 {
        self.deck_cursor
    }

    fn observe_slot(&mut self, slot: u8, card: Card) -> Result<(), BeliefError> {
        let slot = usize::from(slot);
        if let Some(existing) = self.known_slots[slot] {
            if existing != card {
                return Err(BeliefError::InconsistentObservation);
            }
            return Ok(());
        }
        if self
            .known_slots
            .iter()
            .enumerate()
            .any(|(i, c)| i != slot && *c == Some(card))
        {
            return Err(BeliefError::DuplicateObservedCard);
        }
        self.known_slots[slot] = Some(card);
        Ok(())
    }

    fn advance_draw(&mut self) {
        let len = self.deck_slots.len() as u8;
        self.deck_cursor = if self.deck_cursor >= len {
            0
        } else {
            core::cmp::min(self.deck_cursor + self.draw_step.get(), len)
        };
    }

    fn observe_current_waste(&mut self, card: Card) -> Result<(), BeliefError> {
        if self.deck_cursor == 0 {
            return Err(BeliefError::InconsistentObservation);
        }
        let idx = usize::from(self.deck_cursor - 1);
        let Some(&slot) = self.deck_slots.get(idx) else {
            return Err(BeliefError::InconsistentObservation);
        };
        self.observe_slot(slot, card)
    }

    fn remove_current_deck_slot(&mut self, expected: Card) -> Result<(), BeliefError> {
        if self.deck_cursor == 0 {
            return Err(BeliefError::InvalidPublicAction);
        }
        self.observe_current_waste(expected)?;
        let idx = usize::from(self.deck_cursor - 1);
        if idx >= self.deck_slots.len() {
            return Err(BeliefError::InconsistentObservation);
        }
        self.deck_slots.remove(idx);
        self.deck_cursor -= 1;
        Ok(())
    }

    fn note_tableau_reveal(
        &mut self,
        pile: u8,
        card: Card,
    ) -> Result<(), BeliefError> {
        let slots = self
            .hidden_slots
            .get_mut(usize::from(pile))
            .ok_or(BeliefError::InconsistentObservation)?;
        let slot = slots.pop().ok_or(BeliefError::InconsistentObservation)?;
        self.observe_slot(slot, card)
    }

    fn update_snapshot(&mut self, truth: &StandardSolitaire) -> Result<(), BeliefError> {
        let observed = PublicSnapshot::from_truth(truth);
        if observed.deck_len as usize != self.deck_slots.len()
            || observed.deck_cursor != self.deck_cursor
        {
            return Err(BeliefError::InconsistentObservation);
        }
        if let Some(card) = observed.waste_top {
            self.observe_current_waste(card)?;
        }
        self.snapshot = observed;
        Ok(())
    }

    /// Sample one complete initial deal consistent with all observations and
    /// replay the public history to reconstruct the current particle.
    pub fn sample_particle<R: Rng>(&self, rng: &mut R) -> Result<StandardSolitaire, BeliefError> {
        let mut cards = [Card::DEFAULT; N_SLOTS];
        let mut used = [false; N_SLOTS];
        let mut unknown_slots = Vec::new();

        for (slot, known) in self.known_slots.iter().enumerate() {
            if let Some(card) = known {
                let idx = card.mask_index() as usize;
                if used[idx] {
                    return Err(BeliefError::DuplicateObservedCard);
                }
                used[idx] = true;
                cards[slot] = *card;
            } else {
                unknown_slots.push(slot);
            }
        }

        let mut remaining: Vec<Card> = (0..N_CARDS)
            .filter(|i| !used[*i as usize])
            .map(Card::from_mask_index)
            .collect();
        remaining.shuffle(rng);
        if remaining.len() != unknown_slots.len() {
            return Err(BeliefError::InconsistentObservation);
        }
        for (slot, card) in unknown_slots.into_iter().zip(remaining) {
            cards[slot] = card;
        }

        let mut particle = StandardSolitaire::new(&cards, self.draw_step);
        for action in &self.history {
            particle
                .do_move(action)
                .map_err(|_| BeliefError::ParticleReplayFailed)?;
        }
        if PublicSnapshot::from_truth(&particle) != self.snapshot {
            return Err(BeliefError::ParticleReplayFailed);
        }
        Ok(particle)
    }

    /// Validate that independently sampled worlds all reconstruct exactly the
    /// same current public state.
    pub fn validate_particles<R: Rng>(
        &self,
        n: usize,
        rng: &mut R,
    ) -> Result<usize, BeliefError> {
        let mut valid = 0usize;
        for _ in 0..n {
            self.sample_particle(rng)?;
            valid += 1;
        }
        Ok(valid)
    }
}

/// Research-only environment that keeps hidden truth and agent-visible belief
/// in separate fields. The agent should receive `belief()`; `oracle_state()` is
/// deliberately named and intended only for evaluator/oracle code.
pub struct ResearchGame {
    true_state: StandardSolitaire,
    belief: BeliefState,
}

impl ResearchGame {
    pub fn new(cards: &CardDeck, draw_step: NonZeroU8) -> Result<Self, BeliefError> {
        let true_state = StandardSolitaire::new(cards, draw_step);
        let belief = BeliefState::new_from_initial_public(&true_state)?;
        Ok(Self { true_state, belief })
    }

    #[must_use]
    pub const fn belief(&self) -> &BeliefState {
        &self.belief
    }

    /// Evaluator-only access to the hidden truth.
    #[must_use]
    pub const fn oracle_state(&self) -> &StandardSolitaire {
        &self.true_state
    }

    /// Apply one *public* StandardMove. Hidden information is never copied to
    /// the belief state; only automatic reveals and the actually visible waste
    /// top are observed after the move.
    pub fn step(&mut self, action: StandardMove) -> Result<(), BeliefError> {
        if !self.true_state.validate_move(&action) {
            return Err(BeliefError::InvalidPublicAction);
        }

        let source_pile = match action.from {
            Pos::Pile(p) => Some(p),
            _ => None,
        };
        let hidden_before = source_pile.map(|p| self.true_state.get_hidden()[usize::from(p)].len());

        match (action.from, action.to, action.card) {
            (Pos::Deck, Pos::Deck, _) => self.belief.advance_draw(),
            (Pos::Deck, _, card) => self.belief.remove_current_deck_slot(card)?,
            _ => {}
        }

        self.true_state.do_move(&action)?;

        if let (Some(pile), Some(before)) = (source_pile, hidden_before) {
            let after = self.true_state.get_hidden()[usize::from(pile)].len();
            if after + 1 == before {
                let card = self.true_state.get_piles()[usize::from(pile)]
                    .last()
                    .copied()
                    .ok_or(BeliefError::InconsistentObservation)?;
                self.belief.note_tableau_reveal(pile, card)?;
            }
        }

        self.belief.history.push(action);
        self.belief.update_snapshot(&self.true_state)?;
        Ok(())
    }

    pub fn validate_particles<R: Rng>(
        &self,
        n: usize,
        rng: &mut R,
    ) -> Result<usize, BeliefError> {
        self.belief.validate_particles(n, rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::SmallRng, SeedableRng};

    use crate::shuffler::default_shuffle;

    #[test]
    fn initial_particles_preserve_public_state() {
        let cards = default_shuffle(42);
        let mut game = ResearchGame::new(&cards, NonZeroU8::new(3).unwrap()).unwrap();
        let mut rng = SmallRng::seed_from_u64(7);
        assert_eq!(game.belief().known_slot_count(), 7);
        assert_eq!(game.validate_particles(64, &mut rng).unwrap(), 64);

        game.step(StandardMove::DRAW_NEXT).unwrap();
        assert!(game.belief().snapshot().waste_top.is_some());
        assert!(game.belief().known_slot_count() >= 8);
        assert_eq!(game.validate_particles(64, &mut rng).unwrap(), 64);
    }
}
