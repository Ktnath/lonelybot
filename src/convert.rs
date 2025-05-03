use crate::{
    moves::Move,
    standard::{InvalidMove, MoveResult, Pos, StandardHistoryVec, StandardMove, StandardSolitaire},
};

/// # Errors
///
/// Return `InvalidMove` when the move is not valid and not modify anything
pub(crate) fn convert_move(
    game: &StandardSolitaire,
    m: Move,
    move_seq: &mut StandardHistoryVec,
) -> MoveResult<()> {
    match m {
        Move::DeckPile(c) => {
            let cnt = game.find_deck_card(c).ok_or(InvalidMove {})?;
            let pile = game.find_free_pile(c).ok_or(InvalidMove {})?;
            for _ in 0..cnt {
                move_seq.push(StandardMove::DRAW_NEXT);
            }

            move_seq.push(StandardMove::new(Pos::Deck, Pos::Pile(pile), c));
        }
        Move::DeckStack(c) => {
            if c.rank() != game.get_stack().get(c.suit()) {
                return Err(InvalidMove {});
            }

            let cnt = game.find_deck_card(c).ok_or(InvalidMove {})?;
            for _ in 0..cnt {
                move_seq.push(StandardMove::DRAW_NEXT);
            }

            move_seq.push(StandardMove::new(Pos::Deck, Pos::Stack(c.suit()), c));
        }
        Move::StackPile(c) => {
            if c.rank() + 1 != game.get_stack().get(c.suit()) {
                return Err(InvalidMove {});
            }
            let pile = game.find_free_pile(c).ok_or(InvalidMove {})?;
            move_seq.push(StandardMove::new(Pos::Stack(c.suit()), Pos::Pile(pile), c));
        }
        Move::Reveal(c) => {
            let pile_from = game.find_top_card(c).ok_or(InvalidMove {})?;
            let pile_to = game.find_free_pile(c).ok_or(InvalidMove {})?;

            if pile_to == pile_from {
                return Err(InvalidMove {});
            }

            move_seq.push(StandardMove::new(
                Pos::Pile(pile_from),
                Pos::Pile(pile_to),
                c,
            ));
        }
        Move::PileStack(c) => {
            if c.rank() != game.get_stack().get(c.suit()) {
                return Err(InvalidMove {});
            }
            let (pile, cards) = game.find_card(c).ok_or(InvalidMove {})?;
            if let Some(&move_card) = cards.get(1) {
                let pile_other = game.find_free_pile(move_card).ok_or(InvalidMove {})?;

                if pile == pile_other {
                    return Err(InvalidMove {});
                }

                move_seq.push(StandardMove::new(
                    Pos::Pile(pile),
                    Pos::Pile(pile_other),
                    move_card,
                ));
            }
            move_seq.push(StandardMove::new(Pos::Pile(pile), Pos::Stack(c.suit()), c));
        }
    }
    Ok(())
}

/// this will convert and execute the moves
/// # Errors
///
/// Return `InvalidMove` when the one of the move is not valid and the state of the game will stop before making that move
/// # Panics
///
/// Never (unless buggy)
pub fn convert_moves(game: &mut StandardSolitaire, m: &[Move]) -> MoveResult<StandardHistoryVec> {
    let mut move_seq = StandardHistoryVec::new();
    for mm in m {
        let start = move_seq.len();
        convert_move(game, *mm, &mut move_seq)?;

        for m in &move_seq[start..] {
            let valid_move = game.do_move(m).is_ok();
            debug_assert!(valid_move);
        }
    }
    Ok(move_seq)
}

#[cfg(test)]
mod tests {

    use core::num::NonZeroU8;

    use crate::{shuffler::default_shuffle, solver::solve, state::Solitaire};

    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    fn do_test_convert(seed: u64) {
        let draw_step = NonZeroU8::new(3).unwrap();

        let cards = default_shuffle(seed);
        let mut game = StandardSolitaire::new(&cards, draw_step);

        let res = {
            let mut game_1: Solitaire = From::from(&game);
            let mut game_2: Solitaire = Solitaire::new(&cards, draw_step);

            let res1 = solve(&mut game_1);
            let res2 = solve(&mut game_2);

            assert_eq!(res1, res2);
            res1
        };

        let Some(moves) = res.1 else {
            return;
        };

        let mut his = StandardHistoryVec::new();

        let mut game_x: Solitaire = From::from(&game);
        for pos in 0..moves.len() {
            his.clear();
            convert_move(&game, moves[pos], &mut his).unwrap();
            for m in &his {
                assert!(game.do_move(m).is_ok());
            }

            game_x.do_move(moves[pos]);
            let mut game_c: Solitaire = From::from(&game);
            assert!(game_c.is_valid());
            assert!(game_x.equivalent_to(&game_c));

            let mut game_cc: StandardSolitaire = From::from(&game_c);

            for &m in moves[pos + 1..].iter() {
                game_c.do_move(m);
            }
            convert_moves(&mut game_cc, &moves[pos + 1..]).unwrap();
            assert!(game_c.is_win());
            assert!(game_cc.is_win());
        }
    }

    #[test]
    fn test_convert() {
        for seed in 12..20 {
            do_test_convert(seed);
        }
    }
}
