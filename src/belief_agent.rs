//! Particle belief agent for imperfect-information Klondike.
//!
//! Phase 1.2 keeps the validated public BeliefState and improves the world
//! evaluator. Values are bounded and multi-component so more particles reduce
//! uncertainty about a meaningful progress signal instead of merely estimating
//! a coarse deadlock-heavy score. Adaptive sampling can stop at 64 or 256
//! particles when the leading action is statistically separated, otherwise it
//! escalates to 2048.

extern crate alloc;

use alloc::vec::Vec;
use core::cmp::Ordering;

use rand::Rng;

use crate::belief::{BeliefError, BeliefState, PublicSnapshot};
use crate::card::{Card, N_SUITS};
use crate::deck::N_PILES;
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
    /// Optional random rollout exploration. Phase 1.2 defaults to zero so the
    /// measured variance mostly reflects hidden-world uncertainty.
    pub rollout_exploration: f64,
    /// Number of standard errors used for action confidence bounds.
    pub value_lcb_z: f64,
    /// Wilson lower-bound z value for empirical rollout wins.
    pub win_lcb_z: f64,
    /// Minimum confidence gap needed for adaptive early stopping.
    pub adaptive_min_gap: f64,
}

impl Default for ParticleBeliefConfig {
    fn default() -> Self {
        Self {
            rollout_depth: 32,
            rollout_exploration: 0.0,
            value_lcb_z: 1.28,
            win_lcb_z: 1.0,
            adaptive_min_gap: 0.01,
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
    pub value_ucb: f64,
    pub win_rate: f64,
    pub win_lcb: f64,
    pub deadlock_rate: f64,
    pub information_gain_hint: u8,
    pub mean_foundation_progress: f64,
    pub mean_tableau_reveal_progress: f64,
    pub mean_stock_clear_progress: f64,
    pub mean_mobility: f64,
    pub mean_empty_columns: f64,
    pub mean_stock_accessibility: f64,
    pub mean_reveal_options: f64,
}

#[derive(Debug)]
pub struct ParticleBeliefDecision {
    pub particles: usize,
    pub chosen_action: StandardMove,
    pub chosen_index: usize,
    pub actions: Vec<ParticleActionStats>,
}

#[derive(Debug)]
pub struct AdaptiveParticleDecision {
    pub decision: ParticleBeliefDecision,
    pub stopped_early: bool,
    pub confidence_gap: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParticleBeliefError {
    Belief(BeliefError),
    NoPublicActions,
    InvalidBudgets,
}

impl From<BeliefError> for ParticleBeliefError {
    fn from(value: BeliefError) -> Self {
        Self::Belief(value)
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct EvaluatorComponents {
    foundation_progress: f64,
    tableau_reveal_progress: f64,
    stock_clear_progress: f64,
    mobility: f64,
    empty_columns: f64,
    stock_accessibility: f64,
    reveal_options: f64,
}

#[derive(Default)]
struct Accumulator {
    n: usize,
    invalid: usize,
    wins: usize,
    deadlocks: usize,
    sum: f64,
    sum_sq: f64,
    components: EvaluatorComponents,
}

#[derive(Debug, Clone, Copy)]
struct RolloutOutcome {
    value: f64,
    won: bool,
    deadlock: bool,
    components: EvaluatorComponents,
}

fn copy_action(action: &StandardMove) -> StandardMove {
    StandardMove::new(action.from, action.to, action.card)
}

fn clone_standard(game: &StandardSolitaire) -> StandardSolitaire {
    // IMPORTANT: preserve exact public pile identities. Converting through the
    // compact Solitaire representation is only equivalence-preserving: once
    // columns become empty, symmetric piles may be reconstructed under different
    // indices. That is fine for the perfect-information solver but invalid for
    // public actions such as P5->P0, whose source/destination indices are part
    // of the observation and must remain stable across every belief particle.
    game.clone()
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

fn clamp01(x: f64) -> f64 {
    if x < 0.0 {
        0.0
    } else if x > 1.0 {
        1.0
    } else {
        x
    }
}

fn clamp11(x: f64) -> f64 {
    if x < -1.0 {
        -1.0
    } else if x > 1.0 {
        1.0
    } else {
        x
    }
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
    let spread = z * sqrt_newton((p * (1.0 - p) + z2 / (4.0 * nf)) / nf);
    (centre - spread) / denom
}

fn normalized_state_value(engine: &SolitaireEngine<FullPruner>) -> RolloutOutcome {
    if engine.state().is_win() {
        return RolloutOutcome {
            value: 1.0,
            won: true,
            deadlock: false,
            components: EvaluatorComponents {
                foundation_progress: 1.0,
                tableau_reveal_progress: 1.0,
                stock_clear_progress: 1.0,
                mobility: 1.0,
                empty_columns: 1.0,
                stock_accessibility: 1.0,
                reveal_options: 0.0,
            },
        };
    }

    let legal = engine.list_moves_dom();
    let deadlock = legal.is_empty();
    let foundation = f64::from(engine.state().get_stack().len()) / 52.0;
    let hidden_down = f64::from(engine.state().get_hidden().total_down_cards());
    let tableau_reveal = clamp01((21.0 - hidden_down) / 21.0);
    let deck_len = f64::from(engine.state().get_deck().len());
    let stock_clear = clamp01((24.0 - deck_len) / 24.0);
    let mobility = clamp01(legal.len() as f64 / 8.0);

    let piles = engine.state().compute_visible_piles();
    let mut empty = 0usize;
    for i in 0..N_PILES {
        if piles[usize::from(i)].is_empty() && engine.state().get_hidden().len(i) == 0 {
            empty += 1;
        }
    }
    let empty_columns = empty as f64 / f64::from(N_PILES);

    let stock_accessibility = if engine.state().get_deck().len() == 0 {
        1.0
    } else {
        let accessible = engine.state().get_deck().compute_mask(false).count_ones() as f64;
        clamp01(accessible / deck_len)
    };

    let reveal_count = legal.iter().filter(|m| matches!(m, Move::Reveal(_))).count();
    let reveal_options = clamp01(reveal_count as f64 / 3.0);

    let c = EvaluatorComponents {
        foundation_progress: foundation,
        tableau_reveal_progress: tableau_reveal,
        stock_clear_progress: stock_clear,
        mobility,
        empty_columns,
        stock_accessibility,
        reveal_options,
    };

    let progress01 = 0.30 * c.foundation_progress
        + 0.27 * c.tableau_reveal_progress
        + 0.12 * c.stock_clear_progress
        + 0.10 * c.mobility
        + 0.08 * c.empty_columns
        + 0.08 * c.stock_accessibility
        + 0.05 * c.reveal_options;

    let mut value = 2.0 * progress01 - 1.0;
    if deadlock {
        value -= 0.25;
    }

    RolloutOutcome {
        value: clamp11(value),
        won: false,
        deadlock,
        components: c,
    }
}

fn cheap_rollout_score(engine: &SolitaireEngine<FullPruner>, mv: Move) -> i32 {
    match mv {
        Move::Reveal(card) => {
            let col = engine.state().get_hidden().find(card);
            55 + 3 * i32::from(engine.state().get_hidden().len(col))
        }
        Move::PileStack(card) => 28 + i32::from(card.rank()) / 3,
        Move::DeckStack(card) => 24 + i32::from(card.rank()) / 4,
        Move::DeckPile(card) => 14 + if card.is_king() { 6 } else { 0 },
        Move::StackPile(card) => 2 + if card.is_king() { 3 } else { 0 },
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

    // Once the public root action has been applied using the exact Standard
    // layout, the future determinized rollout may safely use the compact solver
    // representation: no subsequent decision is exposed as a public indexed
    // action in this root-sampling baseline.
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

        let chosen = if cfg.rollout_exploration > 0.0
            && rng.random::<f64>() < cfg.rollout_exploration
        {
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

    Some(normalized_state_value(&engine))
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
                out.push(StandardMove::new(Pos::Stack(suit), Pos::Pile(to as u8), card));
            }
        }
    }

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

fn add_components(sum: &mut EvaluatorComponents, c: EvaluatorComponents) {
    sum.foundation_progress += c.foundation_progress;
    sum.tableau_reveal_progress += c.tableau_reveal_progress;
    sum.stock_clear_progress += c.stock_clear_progress;
    sum.mobility += c.mobility;
    sum.empty_columns += c.empty_columns;
    sum.stock_accessibility += c.stock_accessibility;
    sum.reveal_options += c.reveal_options;
}

fn stats_from(
    acc: &Accumulator,
    action: &StandardMove,
    requested: usize,
    info: u8,
    cfg: &ParticleBeliefConfig,
) -> ParticleActionStats {
    let n = acc.n;
    let mean = if n == 0 { -1.0 } else { acc.sum / n as f64 };
    let variance = if n <= 1 {
        0.0
    } else {
        let raw = (acc.sum_sq - acc.sum * acc.sum / n as f64) / (n - 1) as f64;
        if raw > 0.0 { raw } else { 0.0 }
    };
    let stderr = if n == 0 { 0.0 } else { sqrt_newton(variance / n as f64) };
    let win_rate = if n == 0 { 0.0 } else { acc.wins as f64 / n as f64 };
    let deadlock_rate = if n == 0 { 1.0 } else { acc.deadlocks as f64 / n as f64 };
    let denom = if n == 0 { 1.0 } else { n as f64 };

    ParticleActionStats {
        action: copy_action(action),
        requested_particles: requested,
        valid_particles: n,
        invalid_particles: acc.invalid,
        mean_value: mean,
        stderr_value: stderr,
        value_lcb: mean - cfg.value_lcb_z * stderr,
        value_ucb: mean + cfg.value_lcb_z * stderr,
        win_rate,
        win_lcb: wilson_lower(acc.wins, n, cfg.win_lcb_z),
        deadlock_rate,
        information_gain_hint: info,
        mean_foundation_progress: acc.components.foundation_progress / denom,
        mean_tableau_reveal_progress: acc.components.tableau_reveal_progress / denom,
        mean_stock_clear_progress: acc.components.stock_clear_progress / denom,
        mean_mobility: acc.components.mobility / denom,
        mean_empty_columns: acc.components.empty_columns / denom,
        mean_stock_accessibility: acc.components.stock_accessibility / denom,
        mean_reveal_options: acc.components.reveal_options / denom,
    }
}

fn compare_stats(a: &ParticleActionStats, b: &ParticleActionStats) -> Ordering {
    let a_support = a.valid_particles as f64 / a.requested_particles.max(1) as f64;
    let b_support = b.valid_particles as f64 / b.requested_particles.max(1) as f64;

    a_support
        .partial_cmp(&b_support)
        .unwrap_or(Ordering::Equal)
        .then_with(|| a.value_lcb.partial_cmp(&b.value_lcb).unwrap_or(Ordering::Equal))
        .then_with(|| a.win_lcb.partial_cmp(&b.win_lcb).unwrap_or(Ordering::Equal))
        .then_with(|| a.information_gain_hint.cmp(&b.information_gain_hint))
        .then_with(|| b.deadlock_rate.partial_cmp(&a.deadlock_rate).unwrap_or(Ordering::Equal))
        .then_with(|| a.mean_value.partial_cmp(&b.mean_value).unwrap_or(Ordering::Equal))
}

fn build_decision(
    belief: &BeliefState,
    actions: &[StandardMove],
    acc: &[Accumulator],
    requested: usize,
    cfg: &ParticleBeliefConfig,
) -> ParticleBeliefDecision {
    let stats: Vec<ParticleActionStats> = actions
        .iter()
        .zip(acc.iter())
        .map(|(action, a)| {
            stats_from(
                a,
                action,
                requested,
                information_gain_hint(belief, action),
                cfg,
            )
        })
        .collect();

    let mut best_idx = 0usize;
    for idx in 1..stats.len() {
        if compare_stats(&stats[idx], &stats[best_idx]) == Ordering::Greater {
            best_idx = idx;
        }
    }

    ParticleBeliefDecision {
        particles: requested,
        chosen_action: copy_action(&stats[best_idx].action),
        chosen_index: best_idx,
        actions: stats,
    }
}

fn confidence_gap(decision: &ParticleBeliefDecision) -> f64 {
    if decision.actions.len() <= 1 {
        return 2.0;
    }
    let best = &decision.actions[decision.chosen_index];
    let mut strongest_other_ucb = -2.0;
    for (idx, stats) in decision.actions.iter().enumerate() {
        if idx != decision.chosen_index && stats.value_ucb > strongest_other_ucb {
            strongest_other_ucb = stats.value_ucb;
        }
    }
    best.value_lcb - strongest_other_ucb
}

fn sample_more<R: Rng>(
    belief: &BeliefState,
    actions: &[StandardMove],
    acc: &mut [Accumulator],
    additional: usize,
    cfg: &ParticleBeliefConfig,
    rng: &mut R,
) -> Result<(), ParticleBeliefError> {
    for _ in 0..additional {
        let world = belief.sample_particle(rng)?;
        for (idx, action) in actions.iter().enumerate() {
            match rollout(&world, action, cfg, rng) {
                Some(outcome) => {
                    acc[idx].n += 1;
                    acc[idx].sum += outcome.value;
                    acc[idx].sum_sq += outcome.value * outcome.value;
                    acc[idx].wins += usize::from(outcome.won);
                    acc[idx].deadlocks += usize::from(outcome.deadlock);
                    add_components(&mut acc[idx].components, outcome.components);
                }
                None => acc[idx].invalid += 1,
            }
        }
    }
    Ok(())
}

impl BeliefState {
    #[must_use]
    pub fn public_actions(&self) -> Vec<StandardMove> {
        public_actions_from_snapshot(self.snapshot())
    }

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
        sample_more(self, &actions, &mut acc, particles, cfg, rng)?;
        Ok(build_decision(self, &actions, &acc, particles, cfg))
    }

    pub fn adaptive_particle_decision<R: Rng>(
        &self,
        budgets: &[usize],
        cfg: &ParticleBeliefConfig,
        rng: &mut R,
    ) -> Result<AdaptiveParticleDecision, ParticleBeliefError> {
        if budgets.is_empty()
            || budgets[0] == 0
            || budgets.windows(2).any(|w| w[1] <= w[0])
        {
            return Err(ParticleBeliefError::InvalidBudgets);
        }
        let actions = self.public_actions();
        if actions.is_empty() {
            return Err(ParticleBeliefError::NoPublicActions);
        }
        let mut acc: Vec<Accumulator> = (0..actions.len()).map(|_| Accumulator::default()).collect();
        let mut previous = 0usize;

        for (stage, &budget) in budgets.iter().enumerate() {
            sample_more(self, &actions, &mut acc, budget - previous, cfg, rng)?;
            previous = budget;
            let decision = build_decision(self, &actions, &acc, budget, cfg);
            let gap = confidence_gap(&decision);
            let final_stage = stage + 1 == budgets.len();
            if final_stage || gap >= cfg.adaptive_min_gap {
                return Ok(AdaptiveParticleDecision {
                    decision,
                    stopped_early: !final_stage,
                    confidence_gap: gap,
                });
            }
        }

        Err(ParticleBeliefError::InvalidBudgets)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::num::NonZeroU8;
    use rand::{rngs::SmallRng, SeedableRng};

    use crate::belief::ResearchGame;
    use crate::convert::convert_moves;
    use crate::shuffler::default_shuffle;
    use crate::solver::solve;
    use crate::state::Solitaire;

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
        assert!(decision
            .actions
            .iter()
            .all(|s| s.mean_value >= -1.0 && s.mean_value <= 1.0));
    }

    #[test]
    fn adaptive_budget_is_valid_prefix_search() {
        let cards = default_shuffle(17);
        let game = ResearchGame::new(&cards, NonZeroU8::new(3).unwrap()).unwrap();
        let mut rng = SmallRng::seed_from_u64(991);
        let cfg = ParticleBeliefConfig {
            rollout_depth: 4,
            adaptive_min_gap: 10.0,
            ..Default::default()
        };
        let result = game
            .belief()
            .adaptive_particle_decision(&[8, 16, 32], &cfg, &mut rng)
            .unwrap();
        assert_eq!(result.decision.particles, 32);
        assert!(!result.stopped_early);
    }

    #[test]
    fn late_game_public_actions_keep_exact_column_identity() {
        // Regression for Phase 1.2 smoke: seed 2, public step 43 previously
        // produced one candidate that was invalid in every particle because
        // StandardSolitaire was cloned through the compact symmetry-reduced
        // representation, which can renumber empty columns.
        let draw_step = NonZeroU8::new(3).unwrap();
        let cards = default_shuffle(2);
        let mut oracle = Solitaire::new(&cards, draw_step);
        let (_, solution) = solve(&mut oracle);
        let solution = solution.expect("seed 2 should be solvable");

        let mut research = ResearchGame::new(&cards, draw_step).unwrap();
        let mut shadow = StandardSolitaire::new(&cards, draw_step);
        let mut public_step = 0usize;

        for mv in solution {
            let sequence = convert_moves(&mut shadow, &[mv]).unwrap();
            for action in sequence {
                if public_step == 43 {
                    let mut rng = SmallRng::seed_from_u64(0xE1A2_0002_002B);
                    let cfg = ParticleBeliefConfig {
                        rollout_depth: 2,
                        ..Default::default()
                    };
                    let decision = research
                        .belief()
                        .particle_decision(32, &cfg, &mut rng)
                        .unwrap();
                    assert!(decision.actions.len() >= 2);
                    assert!(decision.actions.iter().all(|s| {
                        s.valid_particles == 32 && s.invalid_particles == 0
                    }));
                    return;
                }
                research.step(action).unwrap();
                public_step += 1;
            }
        }

        panic!("seed 2 did not reach public step 43");
    }
}
