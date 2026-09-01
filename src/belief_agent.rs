//! Particle belief agent v1 for imperfect-information Klondike.
//!
//! This is a root-sampling baseline: every public candidate action is evaluated
//! on the same distribution of complete worlds sampled from `BeliefState`.
//! Future rollout choices are made inside each determinized world, so this is
//! intentionally *not* yet a full information-set tree search / POMCP agent.
//! The purpose of v1 is to measure whether increasing the particle budget makes
//! action values and recommendations converge under a rigorously public state.

extern crate alloc;

use alloc::vec::Vec;
use core::cmp::Ordering;

use rand::Rng;

use crate::belief::{BeliefError, BeliefState, PublicSnapshot};
use crate::card::{Card, N_SUITS};
use crate::engine::SolitaireEngine;
use crate::moves::Move;
use crate::pruning::FullPruner;
use crate::standard::{Pos, StandardMove, StandardSolitaire};
use crate::state::Solitaire;

#[derive(Debug, Clone, Copy)]
pub struct ParticleBeliefConfig {
    /// Maximum number of optimized-engine moves simulated after the public root
    /// action inside each sampled world.
    pub rollout_depth: usize,
    /// Small random component preventing the deterministic rollout policy from
    /// getting trapped in one brittle heuristic path.
    pub rollout_exploration: f64,
    /// Number of standard errors subtracted from mean rollout value.
    pub value_lcb_z: f64,
    /// Wilson lower-bound z value for empirical rollout wins.
    pub win_lcb_z: f64,
}

impl Default for ParticleBeliefConfig {
    fn default() -> Self {
        Self {
            rollout_depth: 24,
            rollout_exploration: 0.10,
            value_lcb_z: 1.0,
            win_lcb_z: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ParticleActionStats {
    pub action: StandardMove,
    pub requested_particles: usize,
    pub valid_particles: usize,
    pub invalid_particles: usize,
    pub mean_value: f64,
    pub stderr_value: f64,
    pub value_lcb: f64,
    pub win_rate: f64,
    pub win_lcb: f64,
    pub deadlock_rate: f64,
    /// Cheap public hint: 1 when the action is expected to expose a new tableau
    /// card (or advance stock while some initial slots remain unknown), else 0.
    pub information_gain_hint: u8,
}

#[derive(Debug)]
pub struct ParticleBeliefDecision {
    pub particles: usize,
    pub chosen_action: StandardMove,
    pub chosen_index: usize,
    pub actions: Vec<ParticleActionStats>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParticleBeliefError {
    Belief(BeliefError),
    NoPublicActions,
}

impl From<BeliefError> for ParticleBeliefError {
    fn from(value: BeliefError) -> Self {
        Self::Belief(value)
    }
}

#[derive(Default)]
struct Accumulator {
    n: usize,
    invalid: usize,
    wins: usize,
    deadlocks: usize,
    sum: f64,
    sum_sq: f64,
}

#[derive(Debug, Clone, Copy)]
struct RolloutOutcome {
    value: f64,
    won: bool,
    deadlock: bool,
}

fn copy_action(action: &StandardMove) -> StandardMove {
    StandardMove::new(action.from, action.to, action.card)
}

fn clone_standard(game: &StandardSolitaire) -> StandardSolitaire {
    let compact: Solitaire = game.into();
    StandardSolitaire::from(&compact)
}

fn sqrt_newton(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut g = if x >= 1.0 { x } else { 1.0 };
    for _ in 0..20 {
        g = 0.5 * (g + x / g);
    }
    g
}

fn wilson_lower(successes: usize, n: usize, z: f64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let nf = n as f64;
    let p = successes as f64 / nf;
    let z2 = z * z;
    let denom = 1.0 + z2 / nf;
    let centre = p + z2 / (2.0 * nf);
    let spread = z
        * sqrt_newton((p * (1.0 - p) + z2 / (4.0 * nf)) / nf);
    (centre - spread) / denom
}

fn dense_state_value(engine: &SolitaireEngine<FullPruner>) -> RolloutOutcome {
    if engine.state().is_win() {
        return RolloutOutcome {
            value: 1000.0,
            won: true,
            deadlock: false,
        };
    }

    let legal = engine.list_moves_dom();
    let deadlock = legal.is_empty();
    let foundation = f64::from(engine.state().get_stack().len());
    let hidden = f64::from(engine.state().get_hidden().total_down_cards());
    let deck = f64::from(engine.state().get_deck().len());
    let mobility = legal.len() as f64;

    let mut value = foundation * 18.0 - hidden * 7.0 - deck * 1.25 + mobility * 1.5;
    if deadlock {
        // A late deadlock with substantial progress is still better than an
        // immediate deadlock, unlike the old uniform -250 terminal reward.
        value -= 120.0;
    }
    RolloutOutcome {
        value,
        won: false,
        deadlock,
    }
}

fn cheap_rollout_score(engine: &SolitaireEngine<FullPruner>, mv: Move) -> i32 {
    match mv {
        Move::Reveal(card) => {
            let col = engine.state().get_hidden().find(card);
            45 + i32::from(engine.state().get_hidden().len(col))
        }
        Move::PileStack(card) => {
            // Foundation progress is useful, while very early foundation moves
            // get a small restraint because they can reduce tableau mobility.
            26 + i32::from(card.rank()) / 3
        }
        Move::DeckStack(card) => 22 + i32::from(card.rank()) / 4,
        Move::DeckPile(card) => 12 + if card.is_king() { 5 } else { 0 },
        Move::StackPile(card) => 3 + if card.is_king() { 2 } else { 0 },
    }
}

fn rollout<R: Rng>(
    start: &StandardSolitaire,
    action: &StandardMove,
    cfg: &ParticleBeliefConfig,
    rng: &mut R,
) -> Option<RolloutOutcome> {
    let mut after = clone_standard(start);
    if after.do_move(action).is_err() {
        return None;
    }

    let state: Solitaire = (&after).into();
    let mut engine: SolitaireEngine<FullPruner> = state.into();

    for _ in 0..cfg.rollout_depth {
        if engine.state().is_win() {
            break;
        }
        let legal = engine.list_moves_dom();
        if legal.is_empty() {
            break;
        }

        let chosen = if rng.random::<f64>() < cfg.rollout_exploration {
            legal[rng.random_range(0..legal.len())]
        } else {
            let mut best = legal[0];
            let mut best_score = cheap_rollout_score(&engine, best);
            for &mv in legal.iter().skip(1) {
                let score = cheap_rollout_score(&engine, mv);
                if score > best_score {
                    best = mv;
                    best_score = score;
                }
            }
            best
        };

        if !engine.do_move(chosen) {
            break;
        }
    }

    Some(dense_state_value(&engine))
}

fn public_actions_from_snapshot(snapshot: &PublicSnapshot) -> Vec<StandardMove> {
    let mut out = Vec::new();

    if snapshot.deck_len > 0 {
        out.push(StandardMove::DRAW_NEXT);
    }

    if let Some(card) = snapshot.waste_top {
        for to in 0..snapshot.columns.len() {
            if card.go_after(snapshot.columns[to].visible.last().copied()) {
                out.push(StandardMove::new(Pos::Deck, Pos::Pile(to as u8), card));
            }
        }
        let suit = card.suit() as usize;
        if suit < snapshot.foundation_counts.len()
            && snapshot.foundation_counts[suit] == card.rank()
        {
            out.push(StandardMove::new(Pos::Deck, Pos::Stack(card.suit()), card));
        }
    }

    for from in 0..snapshot.columns.len() {
        let source = &snapshot.columns[from].visible;
        for &card in source {
            for to in 0..snapshot.columns.len() {
                if to == from {
                    continue;
                }
                if card.go_after(snapshot.columns[to].visible.last().copied()) {
                    out.push(StandardMove::new(
                        Pos::Pile(from as u8),
                        Pos::Pile(to as u8),
                        card,
                    ));
                }
            }
        }

        if let Some(&card) = source.last() {
            let suit = card.suit() as usize;
            if suit < snapshot.foundation_counts.len()
                && snapshot.foundation_counts[suit] == card.rank()
            {
                out.push(StandardMove::new(
                    Pos::Pile(from as u8),
                    Pos::Stack(card.suit()),
                    card,
                ));
            }
        }
    }

    for suit in 0..N_SUITS {
        let count = snapshot.foundation_counts[suit as usize];
        if count == 0 {
            continue;
        }
        let card = Card::new(count - 1, suit);
        for to in 0..snapshot.columns.len() {
            if card.go_after(snapshot.columns[to].visible.last().copied()) {
                out.push(StandardMove::new(
                    Pos::Stack(suit),
                    Pos::Pile(to as u8),
                    card,
                ));
            }
        }
    }

    // The construction above can theoretically produce duplicates when public
    // symmetries collapse. Keep a stable order while removing exact duplicates.
    let mut unique = Vec::new();
    for action in out {
        if !unique.iter().any(|x: &StandardMove| x == &action) {
            unique.push(action);
        }
    }
    unique
}

fn information_gain_hint(belief: &BeliefState, action: &StandardMove) -> u8 {
    let snapshot = belief.snapshot();
    match action.from {
        Pos::Pile(from) => {
            let col = &snapshot.columns[usize::from(from)];
            let empties_visible = col.visible.first() == Some(&action.card);
            u8::from(empties_visible && col.hidden_count > 0)
        }
        Pos::Deck if action.to == Pos::Deck => u8::from(belief.unknown_slot_count() > 0),
        _ => 0,
    }
}

fn stats_from(acc: &Accumulator, action: &StandardMove, requested: usize, info: u8, cfg: &ParticleBeliefConfig) -> ParticleActionStats {
    let n = acc.n;
    let mean = if n == 0 { -1000.0 } else { acc.sum / n as f64 };
    let variance = if n <= 1 {
        0.0
    } else {
        let raw = (acc.sum_sq - acc.sum * acc.sum / n as f64) / (n - 1) as f64;
        if raw > 0.0 { raw } else { 0.0 }
    };
    let stderr = if n == 0 { 0.0 } else { sqrt_newton(variance / n as f64) };
    let win_rate = if n == 0 { 0.0 } else { acc.wins as f64 / n as f64 };
    let deadlock_rate = if n == 0 { 1.0 } else { acc.deadlocks as f64 / n as f64 };

    ParticleActionStats {
        action: copy_action(action),
        requested_particles: requested,
        valid_particles: n,
        invalid_particles: acc.invalid,
        mean_value: mean,
        stderr_value: stderr,
        value_lcb: mean - cfg.value_lcb_z * stderr,
        win_rate,
        win_lcb: wilson_lower(acc.wins, n, cfg.win_lcb_z),
        deadlock_rate,
        information_gain_hint: info,
    }
}

fn compare_stats(a: &ParticleActionStats, b: &ParticleActionStats) -> Ordering {
    let a_support = a.valid_particles as f64 / a.requested_particles.max(1) as f64;
    let b_support = b.valid_particles as f64 / b.requested_particles.max(1) as f64;

    a_support
        .partial_cmp(&b_support)
        .unwrap_or(Ordering::Equal)
        .then_with(|| a.win_lcb.partial_cmp(&b.win_lcb).unwrap_or(Ordering::Equal))
        .then_with(|| a.value_lcb.partial_cmp(&b.value_lcb).unwrap_or(Ordering::Equal))
        .then_with(|| a.information_gain_hint.cmp(&b.information_gain_hint))
        .then_with(|| b.deadlock_rate.partial_cmp(&a.deadlock_rate).unwrap_or(Ordering::Equal))
}

impl BeliefState {
    /// Enumerate standard Klondike actions whose legality depends only on the
    /// current public snapshot. Hidden card identities are never consulted.
    #[must_use]
    pub fn public_actions(&self) -> Vec<StandardMove> {
        public_actions_from_snapshot(self.snapshot())
    }

    /// Evaluate all public root actions on `particles` independently sampled
    /// worlds. Reusing the same RNG seed for different particle budgets makes
    /// smaller budgets exact prefixes of larger ones, which is useful for
    /// convergence experiments (64 -> 256 -> 2048).
    pub fn particle_decision<R: Rng>(
        &self,
        particles: usize,
        cfg: &ParticleBeliefConfig,
        rng: &mut R,
    ) -> Result<ParticleBeliefDecision, ParticleBeliefError> {
        let actions = self.public_actions();
        if actions.is_empty() {
            return Err(ParticleBeliefError::NoPublicActions);
        }

        let mut acc: Vec<Accumulator> = (0..actions.len()).map(|_| Accumulator::default()).collect();

        for _ in 0..particles {
            let world = self.sample_particle(rng)?;
            for (idx, action) in actions.iter().enumerate() {
                match rollout(&world, action, cfg, rng) {
                    Some(outcome) => {
                        acc[idx].n += 1;
                        acc[idx].sum += outcome.value;
                        acc[idx].sum_sq += outcome.value * outcome.value;
                        acc[idx].wins += usize::from(outcome.won);
                        acc[idx].deadlocks += usize::from(outcome.deadlock);
                    }
                    None => acc[idx].invalid += 1,
                }
            }
        }

        let stats: Vec<ParticleActionStats> = actions
            .iter()
            .zip(acc.iter())
            .map(|(action, a)| stats_from(a, action, particles, information_gain_hint(self, action), cfg))
            .collect();

        let mut best_idx = 0usize;
        for idx in 1..stats.len() {
            if compare_stats(&stats[idx], &stats[best_idx]) == Ordering::Greater {
                best_idx = idx;
            }
        }

        Ok(ParticleBeliefDecision {
            particles,
            chosen_action: copy_action(&stats[best_idx].action),
            chosen_index: best_idx,
            actions: stats,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::num::NonZeroU8;
    use rand::{rngs::SmallRng, SeedableRng};

    use crate::belief::ResearchGame;
    use crate::shuffler::default_shuffle;

    #[test]
    fn initial_particle_decision_has_full_public_support() {
        let cards = default_shuffle(42);
        let game = ResearchGame::new(&cards, NonZeroU8::new(3).unwrap()).unwrap();
        let mut rng = SmallRng::seed_from_u64(123);
        let cfg = ParticleBeliefConfig {
            rollout_depth: 4,
            ..Default::default()
        };
        let decision = game.belief().particle_decision(16, &cfg, &mut rng).unwrap();
        assert!(!decision.actions.is_empty());
        assert!(decision
            .actions
            .iter()
            .all(|s| s.valid_particles == 16 && s.invalid_particles == 0));
    }
}
