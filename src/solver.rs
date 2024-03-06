use quick_cache::unsync::Cache;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::Arc;
use std::time::Duration;
use std::{collections::HashSet, fmt::Display};

use crate::engine::{Encode, Move, Solitaire, UndoInfo};

use std::thread;

const TRACK_DEPTH: usize = 8;

#[derive(Debug)]
pub struct SearchStats {
    total_visit: AtomicUsize,
    tp_hit: AtomicUsize,
    max_depth: AtomicUsize,
    move_state: [(AtomicU8, AtomicU8); TRACK_DEPTH],
}

#[derive(Debug)]
pub enum SearchResult {
    Terminated,
    Solved,
    Unsolvable,
}

impl SearchStats {
    pub fn new() -> SearchStats {
        SearchStats {
            total_visit: AtomicUsize::new(0),
            tp_hit: AtomicUsize::new(0),
            max_depth: AtomicUsize::new(0),
            move_state: Default::default(),
        }
    }

    pub fn total_visit(&self) -> usize {
        self.total_visit.load(Ordering::Relaxed)
    }

    pub fn tp_hit(&self) -> usize {
        self.tp_hit.load(Ordering::Relaxed)
    }

    pub fn max_depth(&self) -> usize {
        self.max_depth.load(Ordering::Relaxed)
    }
}

impl Display for SearchStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (total, hit, depth) = (self.total_visit(), self.tp_hit(), self.max_depth());
        write!(
            f,
            "Total visit: {}\nTransposition hit: {} (rate {})\nMiss state: {}\nMax depth search: {}\nCurrent progress:",
            total, hit, (hit as f64)/(total as f64), total - hit, depth,
        )?;

        for (cur, total) in &self.move_state {
            write!(
                f,
                " {}/{}",
                cur.load(Ordering::Relaxed),
                total.load(Ordering::Relaxed)
            )?;
        }
        Ok(())
    }
}

const fn get_rev_move(m: &Move, u: &UndoInfo) -> Option<Move> {
    match m {
        Move::DeckStack(_) => None,
        Move::PileStack(c) => {
            if *u == 0 {
                Some(Move::StackPile(*c))
            } else {
                None
            }
        }
        Move::DeckPile(_) => None,
        Move::StackPile(c) => Some(Move::PileStack(*c)),
        Move::Reveal(_) => None,
    }
}

fn solve(
    g: &mut Solitaire,
    rev_move: Option<Move>,
    tp: &mut Cache<Encode, ()>,
    tp_hist: &mut HashSet<Encode>,
    move_list: &mut Vec<Move>,
    history: &mut Vec<Move>,
    stats: &SearchStats,
    terminated: &AtomicBool,
) -> SearchResult {
    if terminated.load(Ordering::Relaxed) {
        return SearchResult::Terminated;
    }

    stats.max_depth.fetch_max(history.len(), Ordering::Relaxed);
    stats.total_visit.fetch_add(1, Ordering::Relaxed);

    if g.is_win() {
        return SearchResult::Solved;
    }
    let encode = g.encode();
    if tp.get(&encode).is_some() || !tp_hist.insert(encode) {
        stats.tp_hit.fetch_add(1, Ordering::Relaxed);
        return SearchResult::Unsolvable;
    } else {
        tp.insert(encode, ());
    }

    let start = move_list.len();
    g.list_moves::<true>(move_list);

    let end = move_list.len();

    let depth = history.len();
    if depth < TRACK_DEPTH {
        stats.move_state[depth].0.store(0, Ordering::Relaxed);
        stats.move_state[depth]
            .1
            .store((end - start) as u8, Ordering::Relaxed);
    }

    for pos in start..end {
        let m = move_list[pos];

        if Some(m) == rev_move {
            continue;
        }

        let undo = g.do_move(&m);
        history.push(m);

        let res = solve(
            g,
            get_rev_move(&m, &undo),
            tp,
            tp_hist,
            move_list,
            history,
            stats,
            terminated,
        );
        if !matches!(res, SearchResult::Unsolvable) {
            return res;
        }
        history.pop();

        g.undo_move(&m, &undo);

        if depth < TRACK_DEPTH {
            stats.move_state[depth]
                .0
                .store((pos - start + 1) as u8, Ordering::Relaxed);
        }
    }

    move_list.truncate(start);
    tp_hist.remove(&encode);

    SearchResult::Unsolvable
}

fn solve_game(
    g: &mut Solitaire,
    stats: &SearchStats,
    terminated: &AtomicBool,
    done: &Sender<()>,
) -> (SearchResult, Option<Vec<Move>>) {
    let mut tp_hist = HashSet::<Encode>::new();
    let mut tp = Cache::<Encode, ()>::new(1024 * 1024 * 256);
    let mut move_list = Vec::<Move>::new();
    let mut history = Vec::<Move>::new();

    let search_res = solve(
        g,
        None,
        &mut tp,
        &mut tp_hist,
        &mut move_list,
        &mut history,
        stats,
        terminated,
    );

    done.send(()).unwrap();

    if let SearchResult::Solved = search_res {
        (search_res, Some(history))
    } else {
        (search_res, None)
    }
}

const STACK_SIZE: usize = 4 * 1024 * 1024;

pub fn run_solve(
    mut g: Solitaire,
    verbose: bool,
    term_signal: &Arc<AtomicBool>,
) -> (SearchResult, SearchStats, Option<Vec<Move>>) {
    let ss = Arc::new(SearchStats::new());

    let (send, recv) = channel::<()>();

    let child = {
        // Spawn thread with explicit stack size
        let ss_clone = ss.clone();
        let term = term_signal.clone();
        thread::Builder::new()
            .stack_size(STACK_SIZE)
            .spawn(move || solve_game(&mut g, ss_clone.as_ref(), term.as_ref(), &send))
            .unwrap()
    };

    if verbose {
        loop {
            match recv.recv_timeout(Duration::from_millis(1000)) {
                Ok(()) => break,
                Err(_) => println!("{}", ss),
            };
        }
    }

    let (res, hist) = child.join().unwrap();

    return (res, Arc::try_unwrap(ss).unwrap(), hist);
}
