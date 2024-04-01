use core::ops::ControlFlow;

use arrayvec::ArrayVec;
use static_assertions::const_assert;

use crate::card::{Card, N_CARDS};

pub const N_PILES: u8 = 7;
pub const N_HIDDEN_CARDS: u8 = N_PILES * (N_PILES + 1) / 2;
pub const N_FULL_DECK: usize = (N_CARDS - N_HIDDEN_CARDS) as usize;

#[derive(Debug, Clone)]
pub struct Deck {
    deck: [Card; N_FULL_DECK],
    draw_step: u8,
    draw_next: u8, // start position of next pile
    draw_cur: u8,  // size of the previous pile
    mask: u32,
    map: [u8; N_CARDS as usize],
}

#[derive(Debug, PartialEq, Eq)]
pub enum Drawable {
    None,
    Current,
    Next,
}

impl Deck {
    #[must_use]
    pub fn new(deck: &[Card; N_FULL_DECK], draw_step: u8) -> Self {
        let draw_step = core::cmp::min(N_FULL_DECK as u8, draw_step);
        let mut map = [!0u8; N_CARDS as usize];
        for (i, c) in deck.iter().enumerate() {
            map[c.value() as usize] = i as u8;
        }

        Self {
            deck: *deck,
            draw_step,
            draw_next: draw_step,
            draw_cur: draw_step,
            mask: 0,
            map,
        }
    }

    #[must_use]
    pub const fn draw_step(&self) -> u8 {
        self.draw_step
    }

    #[must_use]
    pub const fn len(&self) -> u8 {
        N_FULL_DECK as u8 - self.draw_next + self.draw_cur
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.draw_cur == 0 && self.draw_next == N_FULL_DECK as u8
    }

    #[must_use]
    pub fn find_card(&self, card: Card) -> Option<u8> {
        self.deck[..self.draw_cur as usize]
            .iter()
            .chain(self.deck[self.draw_next as usize..].iter())
            .position(|x| x == &card)
            .map(|x| x as u8)
    }

    #[must_use]
    pub fn get_waste(&self) -> &[Card] {
        &self.deck[..self.draw_cur as usize]
    }

    #[must_use]
    pub fn get_deck(&self) -> &[Card] {
        &self.deck[self.draw_next as usize..]
    }

    #[must_use]
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &Card> {
        self.get_waste().iter().chain(self.get_deck().iter())
    }

    #[must_use]
    pub fn iter_waste(
        &self,
    ) -> impl DoubleEndedIterator<Item = (u8, &Card, Drawable)> + ExactSizeIterator {
        self.get_waste().iter().enumerate().map(|x| {
            let pos = x.0 as u8;
            (
                pos,
                x.1,
                if pos + 1 == self.draw_cur {
                    Drawable::Current
                } else if (pos + 1) % self.draw_step == 0 {
                    Drawable::Next
                } else {
                    Drawable::None
                },
            )
        })
    }

    #[must_use]
    pub fn iter_deck(
        &self,
    ) -> impl DoubleEndedIterator<Item = (u8, &Card, Drawable)> + ExactSizeIterator {
        self.get_deck().iter().enumerate().map(|x| {
            let pos = x.0 as u8;
            (
                self.draw_cur + pos,
                x.1,
                if pos + 1 == N_FULL_DECK as u8 - self.draw_next || (pos + 1) % self.draw_step == 0
                {
                    Drawable::Current
                } else if (self.draw_cur + pos + 1) % self.draw_step == 0 {
                    Drawable::Next
                } else {
                    Drawable::None
                },
            )
        })
    }

    #[must_use]
    pub const fn peek(&self, pos: u8) -> &Card {
        if pos < self.draw_cur {
            &self.deck[pos as usize]
        } else {
            &self.deck[(pos - self.draw_cur + self.draw_next) as usize]
        }
    }

    #[must_use]
    pub fn iter_all(&self) -> impl DoubleEndedIterator<Item = (u8, &Card, Drawable)> {
        self.iter_waste().chain(self.iter_deck())
    }

    #[must_use]
    pub fn offset(&self, n_step: u8) -> u8 {
        let next = self.get_offset();
        let len = self.len();
        let step = self.draw_step();

        let n_step_to_end = (len - next).div_ceil(step);

        core::cmp::min(
            if n_step <= n_step_to_end {
                next + step * n_step
            } else {
                let total_step = len.div_ceil(step) + 1;
                let n_step = (n_step - n_step_to_end - 1) % total_step;
                step * n_step
            },
            len,
        )
    }

    #[must_use]
    pub fn offset_once(&self) -> u8 {
        let next = self.get_offset();
        let len = self.len();
        if next >= len {
            0
        } else {
            core::cmp::min(next + self.draw_step(), len)
        }
    }

    pub fn iter_callback<T>(
        &self,
        filter: bool,
        mut func: impl FnMut(u8, &Card) -> ControlFlow<T>,
    ) -> ControlFlow<T> {
        if !filter {
            let mut i = self.draw_step - 1;
            while i + 1 < self.draw_cur {
                func(i, &self.deck[i as usize])?;
                i += self.draw_step;
            }
        }

        if self.draw_cur > 0 {
            func(self.draw_cur - 1, &self.deck[self.draw_cur as usize - 1])?;
        }

        let gap = self.draw_next - self.draw_cur;

        if self.draw_next < N_FULL_DECK as u8 {
            func(N_FULL_DECK as u8 - 1 - gap, &self.deck[N_FULL_DECK - 1])?;
        }

        {
            let mut i = self.draw_next + self.draw_step - 1;
            while i + 1 < N_FULL_DECK as u8 {
                func(i - gap, &self.deck[i as usize])?;
                i += self.draw_step;
            }
        }

        {
            let offset = self.draw_cur % self.draw_step;
            if !filter && offset != 0 {
                let mut i = self.draw_next + self.draw_step - 1 - offset;

                while i + 1 < N_FULL_DECK as u8 {
                    func(i - gap, &self.deck[i as usize])?;
                    i += self.draw_step;
                }
            }
        }
        ControlFlow::Continue(())
    }

    #[must_use]
    pub const fn peek_last(&self) -> Option<&Card> {
        if self.draw_next < N_FULL_DECK as u8 {
            Some(&self.deck[N_FULL_DECK - 1])
        } else if self.draw_cur > 0 {
            Some(&self.deck[self.draw_cur as usize - 1])
        } else {
            None
        }
    }

    pub fn set_offset(&mut self, id: u8) {
        // after this the deck will have structure
        // [.... id-1 <empty> id....]
        //   draw_cur ^       ^ draw_next

        let step = if id < self.draw_cur {
            let step = self.draw_cur - id;
            // moving stuff
            self.deck.copy_within(
                (self.draw_cur - step) as usize..(self.draw_cur as usize),
                (self.draw_next - step) as usize,
            );
            step.wrapping_neg()
        } else {
            let step = id - self.draw_cur;

            self.deck.copy_within(
                (self.draw_next) as usize..(self.draw_next + step) as usize,
                self.draw_cur as usize,
            );
            step
        };

        self.draw_cur = self.draw_cur.wrapping_add(step);
        self.draw_next = self.draw_next.wrapping_add(step);
    }

    fn pop_next(&mut self) -> Card {
        let card = self.deck[self.draw_next as usize];
        self.mask ^= 1 << self.map[card.value() as usize];
        self.draw_next += 1;
        card
    }

    pub fn push(&mut self, card: Card) {
        // or you can undo
        self.mask ^= 1 << self.map[card.value() as usize];
        self.deck[self.draw_cur as usize] = card;
        self.draw_cur += 1;

        //
        // self.draw_next -= 1;
        // self.deck[self.draw_next as usize] = c;
    }

    pub fn draw(&mut self, id: u8) -> Card {
        debug_assert!(
            self.draw_cur <= self.draw_next
                && (id < N_FULL_DECK as u8 - self.draw_next + self.draw_cur)
        );
        self.set_offset(id);
        self.pop_next()
    }

    #[must_use]
    pub const fn get_offset(&self) -> u8 {
        self.draw_cur
    }

    #[must_use]
    pub const fn is_pure(&self) -> bool {
        // this will return true if the deck is pure (when deal repeated it will loop back to the current state)
        self.draw_cur % self.draw_step == 0 || self.draw_next == N_FULL_DECK as u8
    }

    #[must_use]
    pub const fn normalized_offset(&self) -> u8 {
        // this is the standardized version
        if self.draw_cur % self.draw_step == 0 {
            // matched so offset is free
            debug_assert!(self.len() <= N_FULL_DECK as u8);
            self.len()
        } else {
            self.draw_cur
        }
    }

    #[must_use]
    pub const fn encode(&self) -> u32 {
        const_assert!(((N_FULL_DECK - 1).ilog2() + 1 + N_FULL_DECK as u32) <= 32);
        // assert the number of bits
        // 29 bits
        self.mask | ((self.normalized_offset() as u32) << N_FULL_DECK)
    }

    pub fn decode(&mut self, encode: u32) {
        let mask = encode & ((1 << N_FULL_DECK) - 1);
        let offset = (encode >> N_FULL_DECK) as u8;

        let mut rev_map = [Card::FAKE; N_FULL_DECK];

        for i in 0..N_CARDS {
            let val = self.map[i as usize];
            if val < N_FULL_DECK as u8 && (encode >> val) & 1 == 0 {
                rev_map[val as usize] = Card::from_value(i);
            }
        }

        let mut pos = 0;

        for c in rev_map {
            if c != Card::FAKE {
                self.deck[pos] = c;
                pos += 1;
            }
        }

        self.draw_cur = pos as u8;
        self.draw_next = N_FULL_DECK as u8;

        self.set_offset(offset);
        self.mask = mask;
    }

    #[must_use]
    pub fn equivalent_to(&self, other: &Self) -> bool {
        return self
            .iter_all()
            .zip(other.iter_all())
            .all(|x| x.0 .1 == x.1 .1 && (x.0 .2 == Drawable::None) == (x.1 .2 == Drawable::None));
    }

    pub fn deal_once(&mut self) {
        self.set_offset(self.offset_once());
    }

    #[must_use]
    pub fn peek_waste<const N: usize>(&self) -> ArrayVec<Card, N> {
        let draw_cur = self.get_offset();
        self.get_waste()
            .split_at(draw_cur.saturating_sub(N as u8).into())
            .1
            .iter()
            .copied()
            .collect()
    }

    #[must_use]
    pub const fn peek_current(&self) -> Option<&Card> {
        if self.draw_next == 0 {
            None
        } else {
            Some(&self.deck[self.draw_cur as usize - 1])
        }
    }

    pub fn draw_current(&mut self) -> Option<Card> {
        let offset = self.get_offset();
        if offset == 0 {
            None
        } else {
            Some(self.draw(offset - 1))
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::{rngs::StdRng, Rng, SeedableRng};

    use crate::shuffler::default_shuffle;

    use super::*;

    #[test]
    fn test_draw() {
        let mut rng = StdRng::seed_from_u64(14);

        for i in 0..100 {
            let deck = default_shuffle(12 + i);
            let deck = deck[..N_FULL_DECK].try_into().unwrap();

            let draw_step = rng.gen_range(1..5);
            let mut deck = Deck::new(deck, draw_step);

            while !deck.is_empty() {
                assert_eq!(deck.offset_once(), deck.offset(1));
                let step = rng.gen_range(1..100);
                let offset = deck.offset(step);

                for _ in 0..step {
                    deck.deal_once();
                }

                assert_eq!(offset, deck.get_offset());

                for (pos, card, _) in deck.iter_all() {
                    assert_eq!(deck.peek(pos), card);
                }

                for filter in [false, true] {
                    deck.iter_callback::<()>(filter, |pos, card| {
                        assert_eq!(deck.peek(pos), card);
                        ControlFlow::<()>::Continue(())
                    });
                }

                if deck.get_offset() < deck.len() && rng.gen_bool(0.5) {
                    deck.pop_next();
                }
            }
        }
    }
}
