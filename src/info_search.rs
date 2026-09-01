//! Information-set tree search for imperfect-information Klondike.
//!
//! Phase 1.3 uses root-sampled particles from the validated `BeliefState`, but
//! unlike the earlier determinization baselines it never chooses actions from
//! hidden identities. Every tree node is a *public observation* and every edge
//! is a `StandardMove` derivable from that observation alone. When the same
//! public action reveals different cards in different sampled worlds, those
//! outcomes become distinct observation children beneath the same action edge.
//!
//! This is deliberately a compact POMCP-style baseline rather than the final
//! production solver: UCB statistics live on public action edges, one complete
//! world is sampled at the root of each simulation, and newly expanded nodes use
//! a public-safe rollout policy plus the bounded Phase-1.2 world evaluator.

extern crate alloc;

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::cmp::Ordering;

use rand::Rng;

use crate::belief::{BeliefError, BeliefState, PublicColumn, PublicSnapshot};
use crate::card::{Card, N_SUITS};
use crate::deck::N_PILES;
use crate::engine::SolitaireEngine;
use crate::moves::Move;
use crate::pruning::FullPruner;
use crate::standard::{Pos, StandardMove, StandardSolitaire};
use crate::state::Solitaire;

#[derive(Debug, Clone, Copy)]
pub struct InformationSetSearchConfig {
    /// Number of root-sampled simulations.
    pub simulations: usize,
    /// Maximum number of information-set tree decisions before leaf rollout.
    pub tree_depth: usize,
    /// Maximum number of public-safe rollout decisions at a newly expanded leaf.
    pub rollout_depth: usize,
    /// UCB exploration strength. Values are on a bounded [-1, 1] scale.
    pub exploration_c: f64,
    /// Optional public-policy rollout exploration probability.
    pub rollout_exploration: f64,
}

impl Default for InformationSetSearchConfig {
    fn default() -> Self {
        Self {
            simulations: 1024,
            tree_depth: 8,
            rollout_depth: 24,
            exploration_c: 0.35,
            rollout_exploration: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct InformationSetRootActionStats {
    pub action: StandardMove,
    pub visits: usize,
    pub mean_value: f64,
    pub observation_children: usize,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct InformationSetDiagnostics {
    pub simulations: usize,
    pub nodes_created: usize,
    pub total_observation_children: usize,
    pub branching_action_edges: usize,
    pub max_observation_children: usize,
    pub max_tree_depth_reached: usize,
    pub invalid_public_transitions: usize,
    pub rollout_repeat_breaks: usize,
}

#[derive(Debug)]
pub struct InformationSetDecision {
    pub chosen_action: StandardMove,
    pub chosen_index: usize,
    pub root_actions: Vec<InformationSetRootActionStats>,
    pub diagnostics: InformationSetDiagnostics,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InformationSetSearchError {
    Belief(BeliefError),
    NoPublicActions,
    InvalidConfig,
    RootProjectionMismatch,
}

impl From<BeliefError> for InformationSetSearchError {
    fn from(value: BeliefError) -> Self {
        Self::Belief(value)
    }
}

#[derive(Debug)]
struct ObservationChild {
    snapshot: PublicSnapshot,
    node: Box<InformationSetNode>,
}

#[derive(Debug)]
struct ActionEdge {
    action: StandardMove,
    visits: usize,
    value_sum: f64,
    children: Vec<ObservationChild>,
}

#[derive(Debug)]
struct InformationSetNode {
    snapshot: PublicSnapshot,
    visits: usize,
    edges: Vec<ActionEdge>,
}

impl InformationSetNode {
    fn new(snapshot: PublicSnapshot) -> Self {
        let edges = public_actions_from_snapshot(&snapshot)
            .into_iter()
            .map(|action| ActionEdge {
                action,
                visits: 0,
                value_sum: 0.0,
                children: Vec::new(),
            })
            .collect();
        Self {
            snapshot,
            visits: 0,
            edges,
        }
    }
}

fn copy_action(action: &StandardMove) -> StandardMove {
    StandardMove::new(action.from, action.to, action.card)
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

/// Public projection of one sampled world. Only fields observable by the agent
/// are copied into the tree key.
fn project_public(game: &StandardSolitaire) -> PublicSnapshot {
    let columns = core::array::from_fn(|i| PublicColumn {
        #[allow(clippy::cast_possible_truncation)]
        hidden_count: game.get_hidden()[i].len() as u8,
        visible: game.get_piles()[i].clone(),
    });
    let foundation_counts = core::array::from_fn(|s| game.get_stack().get(s as u8));
    PublicSnapshot {
        columns,
        foundation_counts,
        deck_len: game.get_deck().len(),
        deck_cursor: game.get_deck().get_offset(),
        waste_top: game.get_deck().peek_current(),
        draw_step: game.get_deck().draw_step().get(),
    }
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

/// Action prior used only for unvisited-edge ordering and public rollouts.
/// Every feature here is derivable from the public snapshot.
fn public_action_score(snapshot: &PublicSnapshot, action: &StandardMove) -> i32 {
    let mut score = 0i32;
    match (action.from, action.to) {
        (Pos::Pile(from), _) => {
            let col = &snapshot.columns[usize::from(from)];
            if col.visible.first() == Some(&action.card) && col.hidden_count > 0 {
                score += 80 + i32::from(col.hidden_count) * 3;
            }
        }
        _ => {}
    }

    match (action.from, action.to) {
        (Pos::Pile(_), Pos::Stack(_)) => score += 32,
        (Pos::Deck, Pos::Stack(_)) => score += 28,
        (Pos::Deck, Pos::Pile(_)) => score += 18,
        (Pos::Pile(_), Pos::Pile(_)) => score += 14,
        (Pos::Deck, Pos::Deck) => score += 6,
        (Pos::Stack(_), Pos::Pile(_)) => score += 1,
        _ => {}
    }
    if action.card.is_king() {
        score += 3;
    }
    score
}

/// Bounded Phase-1.2-style evaluator. It is allowed to inspect the sampled
/// world because this is a simulation return, not an action-selection input.
fn world_value(game: &StandardSolitaire) -> f64 {
    if game.is_win() {
        return 1.0;
    }

    let compact: Solitaire = game.into();
    let engine: SolitaireEngine<FullPruner> = compact.into();
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

    let progress01 = 0.30 * foundation
        + 0.27 * tableau_reveal
        + 0.12 * stock_clear
        + 0.10 * mobility
        + 0.08 * empty_columns
        + 0.08 * stock_accessibility
        + 0.05 * reveal_options;
    let mut value = 2.0 * progress01 - 1.0;
    if deadlock {
        value -= 0.25;
    }
    clamp11(value)
}

fn choose_public_rollout_action<R: Rng>(
    snapshot: &PublicSnapshot,
    actions: &[StandardMove],
    exploration: f64,
    rng: &mut R,
) -> usize {
    if exploration > 0.0 && rng.random::<f64>() < exploration {
        return rng.random_range(0..actions.len());
    }
    let mut best = 0usize;
    let mut best_score = public_action_score(snapshot, &actions[0]);
    for (idx, action) in actions.iter().enumerate().skip(1) {
        let score = public_action_score(snapshot, action);
        if score > best_score {
            best = idx;
            best_score = score;
        }
    }
    best
}

fn public_rollout<R: Rng>(
    world: &mut StandardSolitaire,
    cfg: &InformationSetSearchConfig,
    diagnostics: &mut InformationSetDiagnostics,
    rng: &mut R,
) -> f64 {
    let mut seen: Vec<PublicSnapshot> = Vec::new();
    for _ in 0..cfg.rollout_depth {
        if world.is_win() {
            return 1.0;
        }
        let snapshot = project_public(world);
        if seen.iter().any(|s| s == &snapshot) {
            diagnostics.rollout_repeat_breaks += 1;
            break;
        }
        seen.push(snapshot.clone());
        let actions = public_actions_from_snapshot(&snapshot);
        if actions.is_empty() {
            break;
        }
        let idx = choose_public_rollout_action(&snapshot, &actions, cfg.rollout_exploration, rng);
        if world.do_move(&actions[idx]).is_err() {
            diagnostics.invalid_public_transitions += 1;
            return -1.0;
        }
    }
    world_value(world)
}

fn select_edge(node: &InformationSetNode, exploration_c: f64) -> usize {
    // Expand every public action at least once. Public prior only determines the
    // order; no hidden-world information enters this choice.
    let mut best_unvisited: Option<(usize, i32)> = None;
    for (idx, edge) in node.edges.iter().enumerate() {
        if edge.visits == 0 {
            let prior = public_action_score(&node.snapshot, &edge.action);
            if best_unvisited.map_or(true, |(_, score)| prior > score) {
                best_unvisited = Some((idx, prior));
            }
        }
    }
    if let Some((idx, _)) = best_unvisited {
        return idx;
    }

    let parent_scale = sqrt_newton((node.visits.max(1)) as f64);
    let mut best_idx = 0usize;
    let mut best_score = f64::NEG_INFINITY;
    for (idx, edge) in node.edges.iter().enumerate() {
        let mean = edge.value_sum / edge.visits as f64;
        let bonus = exploration_c * parent_scale / sqrt_newton(edge.visits as f64);
        let score = mean + bonus;
        if score > best_score {
            best_score = score;
            best_idx = idx;
        }
    }
    best_idx
}

fn simulate<R: Rng>(
    node: &mut InformationSetNode,
    world: &mut StandardSolitaire,
    depth: usize,
    cfg: &InformationSetSearchConfig,
    diagnostics: &mut InformationSetDiagnostics,
    rng: &mut R,
) -> f64 {
    if world.is_win() {
        return 1.0;
    }
    if depth >= cfg.tree_depth || node.edges.is_empty() {
        return public_rollout(world, cfg, diagnostics, rng);
    }

    if depth > diagnostics.max_tree_depth_reached {
        diagnostics.max_tree_depth_reached = depth;
    }

    let edge_idx = select_edge(node, cfg.exploration_c);
    let action = copy_action(&node.edges[edge_idx].action);
    if world.do_move(&action).is_err() {
        diagnostics.invalid_public_transitions += 1;
        node.visits += 1;
        node.edges[edge_idx].visits += 1;
        node.edges[edge_idx].value_sum -= 1.0;
        return -1.0;
    }

    let observed = project_public(world);
    let child_pos = node.edges[edge_idx]
        .children
        .iter()
        .position(|child| child.snapshot == observed);

    let value = if let Some(pos) = child_pos {
        simulate(
            &mut node.edges[edge_idx].children[pos].node,
            world,
            depth + 1,
            cfg,
            diagnostics,
            rng,
        )
    } else {
        let child_node = InformationSetNode::new(observed.clone());
        let child = ObservationChild {
            snapshot: observed,
            node: Box::new(child_node),
        };
        node.edges[edge_idx].children.push(child);
        diagnostics.nodes_created += 1;
        diagnostics.total_observation_children += 1;
        let n_children = node.edges[edge_idx].children.len();
        if n_children == 2 {
            diagnostics.branching_action_edges += 1;
        }
        if n_children > diagnostics.max_observation_children {
            diagnostics.max_observation_children = n_children;
        }
        node.edges[edge_idx].children[n_children - 1].node.visits = 1;
        if depth + 1 > diagnostics.max_tree_depth_reached {
            diagnostics.max_tree_depth_reached = depth + 1;
        }
        public_rollout(world, cfg, diagnostics, rng)
    };

    node.visits += 1;
    node.edges[edge_idx].visits += 1;
    node.edges[edge_idx].value_sum += value;
    value
}

fn compare_root_edges(a: &ActionEdge, b: &ActionEdge) -> Ordering {
    a.visits
        .cmp(&b.visits)
        .then_with(|| {
            let av = if a.visits == 0 { -2.0 } else { a.value_sum / a.visits as f64 };
            let bv = if b.visits == 0 { -2.0 } else { b.value_sum / b.visits as f64 };
            av.partial_cmp(&bv).unwrap_or(Ordering::Equal)
        })
}

impl BeliefState {
    /// POMCP-style root-sampled information-set search.
    ///
    /// Each simulation samples one complete current world from this belief.
    /// Tree action choices use only public observation nodes. Hidden identities
    /// affect only state transitions and the returned simulation value.
    pub fn information_set_decision<R: Rng>(
        &self,
        cfg: &InformationSetSearchConfig,
        rng: &mut R,
    ) -> Result<InformationSetDecision, InformationSetSearchError> {
        if cfg.simulations == 0 || cfg.tree_depth == 0 {
            return Err(InformationSetSearchError::InvalidConfig);
        }
        let root_snapshot = self.snapshot().clone();
        let mut root = InformationSetNode::new(root_snapshot.clone());
        if root.edges.is_empty() {
            return Err(InformationSetSearchError::NoPublicActions);
        }

        let mut diagnostics = InformationSetDiagnostics {
            simulations: cfg.simulations,
            nodes_created: 1,
            ..Default::default()
        };

        for _ in 0..cfg.simulations {
            let mut world = self.sample_particle(rng)?;
            if project_public(&world) != root_snapshot {
                return Err(InformationSetSearchError::RootProjectionMismatch);
            }
            let _ = simulate(&mut root, &mut world, 0, cfg, &mut diagnostics, rng);
        }

        let mut best_idx = 0usize;
        for idx in 1..root.edges.len() {
            if compare_root_edges(&root.edges[idx], &root.edges[best_idx]) == Ordering::Greater {
                best_idx = idx;
            }
        }

        let root_actions = root
            .edges
            .iter()
            .map(|edge| InformationSetRootActionStats {
                action: copy_action(&edge.action),
                visits: edge.visits,
                mean_value: if edge.visits == 0 {
                    -1.0
                } else {
                    edge.value_sum / edge.visits as f64
                },
                observation_children: edge.children.len(),
            })
            .collect();

        Ok(InformationSetDecision {
            chosen_action: copy_action(&root.edges[best_idx].action),
            chosen_index: best_idx,
            root_actions,
            diagnostics,
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
    fn information_set_search_preserves_public_legality() {
        let cards = default_shuffle(42);
        let game = ResearchGame::new(&cards, NonZeroU8::new(3).unwrap()).unwrap();
        let mut rng = SmallRng::seed_from_u64(12345);
        let cfg = InformationSetSearchConfig {
            simulations: 128,
            tree_depth: 5,
            rollout_depth: 8,
            ..Default::default()
        };
        let decision = game
            .belief()
            .information_set_decision(&cfg, &mut rng)
            .unwrap();
        assert!(!decision.root_actions.is_empty());
        assert_eq!(decision.diagnostics.invalid_public_transitions, 0);
        assert!(decision.root_actions[decision.chosen_index].visits > 0);
    }

    #[test]
    fn root_draw_can_branch_by_public_observation() {
        let cards = default_shuffle(17);
        let game = ResearchGame::new(&cards, NonZeroU8::new(3).unwrap()).unwrap();
        let mut rng = SmallRng::seed_from_u64(9981);
        let mut observations: Vec<PublicSnapshot> = Vec::new();
        for _ in 0..64 {
            let mut world = game.belief().sample_particle(&mut rng).unwrap();
            world.do_move(&StandardMove::DRAW_NEXT).unwrap();
            let obs = project_public(&world);
            if !observations.iter().any(|x| x == &obs) {
                observations.push(obs);
            }
        }
        assert!(observations.len() > 1);
    }
}
