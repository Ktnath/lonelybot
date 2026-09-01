use core::num::NonZeroU8;
use std::env;

use lonelybot::belief::ResearchGame;
use lonelybot::convert::convert_moves;
use lonelybot::shuffler::default_shuffle;
use lonelybot::solver::{solve, SearchResult};
use lonelybot::standard::{Pos, StandardSolitaire};
use lonelybot::state::Solitaire;
use rand::{rngs::SmallRng, SeedableRng};
use serde_json::json;

#[derive(Default)]
struct ActionCounts {
    draw: usize,
    deck_pile: usize,
    deck_stack: usize,
    pile_pile: usize,
    pile_stack: usize,
    stack_pile: usize,
}

fn main() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    let n_seeds: u64 = args.get(1).map_or(Ok(20), |s| s.parse()).map_err(|e| format!("invalid seeds: {e}"))?;
    let particles: usize = args.get(2).map_or(Ok(8), |s| s.parse()).map_err(|e| format!("invalid particles: {e}"))?;
    let max_actions: usize = args.get(3).map_or(Ok(2000), |s| s.parse()).map_err(|e| format!("invalid max_actions: {e}"))?;
    let draw_step = NonZeroU8::new(3).unwrap();

    let mut rng = SmallRng::seed_from_u64(0xB3113F);
    let mut solved = 0usize;
    let mut unsolved = 0usize;
    let mut actions = 0usize;
    let mut checkpoints = 0usize;
    let mut particles_validated = 0usize;
    let mut max_known_slots = 0usize;
    let mut min_unknown_slots = 52usize;
    let mut action_counts = ActionCounts::default();

    'seeds: for seed in 0..n_seeds {
        let cards = default_shuffle(seed);
        let mut oracle = Solitaire::new(&cards, draw_step);
        let (status, solution) = solve(&mut oracle);
        let Some(solution) = solution else {
            if status == SearchResult::Unsolvable {
                unsolved += 1;
            }
            continue;
        };
        solved += 1;

        let mut research = ResearchGame::new(&cards, draw_step).map_err(|e| format!("seed {seed} init: {e:?}"))?;
        let mut shadow = StandardSolitaire::new(&cards, draw_step);

        particles_validated += research
            .validate_particles(particles, &mut rng)
            .map_err(|e| format!("seed {seed} initial validation: {e:?}"))?;
        checkpoints += 1;

        for mv in solution {
            let sequence = convert_moves(&mut shadow, &[mv])
                .map_err(|_| format!("seed {seed}: optimized-to-standard conversion failed"))?;

            for action in sequence {
                match (action.from, action.to) {
                    (Pos::Deck, Pos::Deck) => action_counts.draw += 1,
                    (Pos::Deck, Pos::Pile(_)) => action_counts.deck_pile += 1,
                    (Pos::Deck, Pos::Stack(_)) => action_counts.deck_stack += 1,
                    (Pos::Pile(_), Pos::Pile(_)) => action_counts.pile_pile += 1,
                    (Pos::Pile(_), Pos::Stack(_)) => action_counts.pile_stack += 1,
                    (Pos::Stack(_), Pos::Pile(_)) => action_counts.stack_pile += 1,
                    _ => {}
                }

                research
                    .step(action)
                    .map_err(|e| format!("seed {seed} action {actions}: {e:?}"))?;
                actions += 1;

                particles_validated += research
                    .validate_particles(particles, &mut rng)
                    .map_err(|e| format!("seed {seed} checkpoint {actions}: {e:?}"))?;
                checkpoints += 1;

                max_known_slots = max_known_slots.max(research.belief().known_slot_count());
                min_unknown_slots = min_unknown_slots.min(research.belief().unknown_slot_count());

                if actions >= max_actions {
                    break 'seeds;
                }
            }
        }
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "seeds_requested": n_seeds,
            "solved_seeds_entered": solved,
            "unsolved_seeds": unsolved,
            "standard_actions_replayed": actions,
            "checkpoints": checkpoints,
            "particles_per_checkpoint": particles,
            "particles_validated": particles_validated,
            "particle_failures": 0,
            "max_known_initial_slots": max_known_slots,
            "min_unknown_initial_slots": min_unknown_slots,
            "action_counts": {
                "draw_stock": action_counts.draw,
                "deck_to_tableau": action_counts.deck_pile,
                "deck_to_foundation": action_counts.deck_stack,
                "tableau_to_tableau": action_counts.pile_pile,
                "tableau_to_foundation": action_counts.pile_stack,
                "foundation_to_tableau": action_counts.stack_pile,
            },
            "invariant": "every sampled particle replays the complete public history and reproduces the exact current public snapshot"
        }))
        .map_err(|e| e.to_string())?
    );
    Ok(())
}
