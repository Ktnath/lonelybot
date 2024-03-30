use core::fmt;

pub const N_SUITS: u8 = 4;
pub const N_RANKS: u8 = 13;
pub const N_CARDS: u8 = N_SUITS * N_RANKS;
pub const KING_RANK: u8 = N_RANKS - 1;

pub const SYMBOLS: [&str; N_SUITS as usize] = ["♥", "♦", "♣", "♠"];
pub const NUMBERS: [&str; N_RANKS as usize] = [
    "A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Card(u8);

impl Card {
    // suit = 1 to make sure it turn on the first bit in suit for deck
    pub const FAKE: Self = Self::new(N_RANKS, 0);

    #[must_use]
    pub const fn new(rank: u8, suit: u8) -> Self {
        debug_assert!(rank <= N_RANKS && suit < N_SUITS);
        Self(rank * N_SUITS + suit)
    }

    #[must_use]
    pub const fn from_value(value: u8) -> Self {
        Self(value)
    }

    #[must_use]
    pub const fn rank(&self) -> u8 {
        self.0 / N_SUITS
    }

    #[must_use]
    pub const fn suit(&self) -> u8 {
        self.0 % N_SUITS
    }

    #[must_use]
    pub const fn value(&self) -> u8 {
        self.0
    }

    #[must_use]
    pub const fn split(&self) -> (u8, u8) {
        (self.rank(), self.suit())
    }

    #[must_use]
    pub const fn swap_suit(&self) -> Self {
        // keeping the color of the suit and switch to the other type
        // also keeping the rank
        Self(self.0 ^ 1)
    }

    #[must_use]
    pub const fn swap_color(&self) -> Self {
        Self(self.0 ^ 2)
    }

    #[must_use]
    pub const fn reduce_rank(&self) -> Self {
        Self(self.0.saturating_sub(N_SUITS))
    }

    #[must_use]
    pub const fn go_before(&self, other: &Self) -> bool {
        let card_a = self.split();
        let card_b = other.split();
        card_a.0 == card_b.0 + 1 && ((card_a.1 ^ card_b.1) & 2 == 2 || card_a.0 == N_RANKS)
    }

    pub fn print_solvitaire<const LOWER: bool>(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (rank, suit) = self.split();
        let s = match suit {
            0 => 'H',
            1 => 'D',
            2 => 'C',
            3 => 'S',
            _ => 'x',
        };
        write!(
            f,
            r#""{}{}""#,
            NUMBERS[rank as usize],
            if LOWER { s.to_ascii_lowercase() } else { s }
        )
    }
}

#[must_use]
pub const fn card_mask(c: &Card) -> u64 {
    let v = c.value();
    1u64 << (v ^ ((v >> 1) & 2))
}

#[must_use]
pub const fn from_mask(v: &u64) -> Card {
    let v = v.trailing_zeros() as u8;
    let v = v ^ ((v >> 1) & 2);
    Card::new(v / N_SUITS, v % N_SUITS)
}