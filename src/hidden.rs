use rand::seq::SliceRandom;
use rand::RngCore;

use arrayvec::ArrayVec;

use crate::card::{Card, KING_RANK, N_CARDS};
use crate::deck::{N_PILES, N_PILE_CARDS};

use crate::standard::HiddenVec;

#[derive(Debug, Clone)]
pub struct Hidden {
    hidden_piles: [Card; N_PILE_CARDS as usize],
    n_hidden: [u8; N_PILES as usize],
    pile_map: [u8; N_CARDS as usize],
    first_layer_mask: u64,
}

impl Hidden {
    #[must_use]
    pub fn new(hidden_piles: [Card; N_PILE_CARDS as usize]) -> Self {
        let mut pile_map = [0; N_CARDS as usize];

        let mut first_layer_mask: u64 = 0;
        for i in 0..N_PILES {
            let start = i * (i + 1) / 2;
            let end = (i + 2) * (i + 1) / 2;

            let p = &hidden_piles[start as usize..end as usize];
            first_layer_mask |= p[0].mask();

            for c in p {
                pile_map[usize::from(c.value())] = i;
            }
        }

        Self {
            hidden_piles,
            n_hidden: core::array::from_fn(|i| (i + 1) as u8),
            pile_map,
            first_layer_mask,
        }
    }

    #[must_use]
    pub fn from_piles(
        piles: &[HiddenVec; N_PILES as usize],
        top: &[Option<Card>; N_PILES as usize],
    ) -> Self {
        let mut hidden_piles = [Card::FAKE; N_PILE_CARDS as usize];
        let mut pile_map = [0u8; N_CARDS as usize];

        let mut first_layer_mask: u64 = 0;
        for i in 0..N_PILES as usize {
            for (j, c) in piles[i].iter().chain(top[i].iter()).enumerate() {
                if j == 0 {
                    first_layer_mask |= c.mask();
                }

                hidden_piles[(i * (i + 1) / 2) + j] = *c;
                pile_map[c.value() as usize] = i as u8;
            }
        }

        Self {
            hidden_piles,
            pile_map,
            n_hidden: core::array::from_fn(|i| piles[i].len() as u8 + u8::from(top[i].is_some())),
            first_layer_mask,
        }
    }

    #[must_use]
    pub fn to_piles(&self) -> [HiddenVec; N_PILES as usize] {
        let mut hidden_piles: [HiddenVec; N_PILES as usize] = Default::default();

        for i in 0..N_PILES {
            let Some((_, pile_map)) = self.get(i).split_last() else {
                continue;
            };
            for c in pile_map {
                hidden_piles[i as usize].push(*c);
            }
        }
        hidden_piles
    }

    #[must_use]
    pub const fn len(&self, pos: u8) -> u8 {
        self.n_hidden[pos as usize]
    }

    #[must_use]
    const fn get_range(&self, pos: u8) -> core::ops::Range<usize> {
        let start = (pos * (pos + 1) / 2) as usize;
        let end = start + self.n_hidden[pos as usize] as usize;
        start..end
    }

    #[must_use]
    pub fn get(&self, pos: u8) -> &[Card] {
        &self.hidden_piles[self.get_range(pos)]
    }

    #[must_use]
    pub fn get_mut(&mut self, pos: u8) -> &mut [Card] {
        let range = self.get_range(pos);
        &mut self.hidden_piles[range]
    }

    #[must_use]
    pub fn peek(&self, pos: u8) -> Option<&Card> {
        self.get(pos).last()
    }

    pub fn pop(&mut self, pos: u8) -> Option<&Card> {
        self.n_hidden[usize::from(pos)] -= 1;
        self.peek(pos)
    }

    pub fn unpop(&mut self, pos: u8) {
        self.n_hidden[usize::from(pos)] += 1;
    }

    #[must_use]
    pub const fn find(&self, c: &Card) -> u8 {
        self.pile_map[c.value() as usize]
    }

    #[must_use]
    pub fn all_turn_up(&self) -> bool {
        self.n_hidden.iter().all(|x| *x <= 1)
    }

    #[must_use]
    pub fn total_down_cards(&self) -> u8 {
        self.n_hidden
            .iter()
            .map(|x| x.saturating_sub(1))
            .sum::<u8>()
    }

    // can be made const fn
    #[must_use]
    pub fn encode(&self) -> u16 {
        self.n_hidden
            .iter()
            .enumerate()
            .rev()
            .fold(0u16, |res, cur| {
                res * (cur.0 as u16 + 2) + u16::from(*cur.1)
            })
    }

    pub fn decode(&mut self, mut hidden_encode: u16) {
        for i in 0..N_PILES {
            let n_options = u16::from(i) + 2;
            self.n_hidden[i as usize] = (hidden_encode % n_options) as u8;
            hidden_encode /= n_options;
        }
    }

    fn update_map(&mut self) {
        for pos in 0..N_PILES {
            for c in &self.hidden_piles[self.get_range(pos)] {
                self.pile_map[c.value() as usize] = pos;
            }
        }
    }

    #[must_use]
    pub fn mask(&self) -> u64 {
        let mut mask = 0;
        for pos in 0..N_PILES {
            if let Some((_, pile_map)) = self.get(pos).split_last() {
                for c in pile_map {
                    mask |= c.mask();
                }
            }
        }
        mask
    }

    #[must_use]
    pub const fn first_layer_mask(&self) -> u64 {
        self.first_layer_mask
    }

    // reset all hidden cards into lexicographic order
    pub fn clear(&mut self) {
        let mut hidden_cards = self.mask();

        for pos in 0..N_PILES {
            if let Some((_, pile_map)) = self.get_mut(pos).split_last_mut() {
                for h in pile_map {
                    debug_assert_ne!(hidden_cards, 0);
                    *h = Card::from_mask(&hidden_cards);
                    hidden_cards &= hidden_cards.wrapping_sub(1);
                }
            }
        }
        debug_assert_eq!(hidden_cards, 0);
        self.update_map();
    }

    pub fn shuffle<R: RngCore>(&mut self, rng: &mut R) {
        let mut all_stuff = ArrayVec::<Card, { N_PILE_CARDS as usize }>::new();
        for pos in 0..N_PILES {
            if let Some((_, pile_map)) = self.get(pos).split_last() {
                all_stuff.extend(pile_map.iter().copied());
            }
        }
        all_stuff.shuffle(rng);

        let mut start = 0;

        for pos in 0..N_PILES {
            if let Some((_, pile_map)) = self.get_mut(pos).split_last_mut() {
                pile_map.copy_from_slice(&all_stuff[start..start + pile_map.len()]);
                start += pile_map.len();
            }
        }
        self.update_map();
    }

    #[must_use]
    pub fn is_valid(&self) -> bool {
        for pos in 0..N_PILES {
            for c in self.get(pos) {
                if self.pile_map[c.value() as usize] != pos {
                    return false;
                }
            }
        }
        true
    }

    #[must_use]
    pub fn normalize(&self) -> [u8; N_PILES as usize] {
        core::array::from_fn(|pos| {
            let n_hid = self.n_hidden[pos];
            match n_hid {
                2.. => n_hid,
                1 => u8::from(self.get(pos as u8)[0].rank() < KING_RANK),
                0 => 0,
            }
        })
    }
}
