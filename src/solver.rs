use crate::engine::{Encode, Move, Solitaire};
use arrayvec::ArrayVec;
use core::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use quick_cache::{unsync::Cache, UnitWeighter};

pub type TpCache = Cache<Encode, (), UnitWeighter, nohash_hasher::BuildNoHashHasher<u64>>;

// before every progress you'd do at most 2*N_RANKS move
// and there would only be N_FULL_DECK + N_HIDDEN progress step
const TP_SIZE: usize = 256 * 1024 * 1024;
const N_PLY_MAX: usize = 1024;
const TRACK_DEPTH: usize = 8;

pub type HistoryVec = ArrayVec<Move, N_PLY_MAX>;

pub trait SearchStatistics {
    fn hit_a_state(&self, depth: usize);
    fn hit_unique_state(&self, depth: usize, n_moves: usize);
    fn finish_move(&self, depth: usize, move_pos: usize);

    fn total_visit(&self) -> usize;
    fn unique_visit(&self) -> usize;
    fn max_depth(&self) -> usize;
}

#[derive(Debug)]
pub struct AtomicSearchStats {
    total_visit: AtomicUsize,
    unique_visit: AtomicUsize,
    max_depth: AtomicUsize,
    move_state: [(AtomicU8, AtomicU8); TRACK_DEPTH],
}
impl AtomicSearchStats {
    pub fn new() -> AtomicSearchStats {
        AtomicSearchStats {
            total_visit: AtomicUsize::new(0),
            unique_visit: AtomicUsize::new(0),
            max_depth: AtomicUsize::new(0),
            move_state: Default::default(),
        }
    }
}

impl SearchStatistics for AtomicSearchStats {
    fn total_visit(&self) -> usize {
        self.total_visit.load(Ordering::Relaxed)
    }

    fn unique_visit(&self) -> usize {
        self.unique_visit.load(Ordering::Relaxed)
    }

    fn max_depth(&self) -> usize {
        self.max_depth.load(Ordering::Relaxed)
    }

    fn hit_a_state(&self, depth: usize) {
        self.max_depth.fetch_max(depth, Ordering::Relaxed);
        self.total_visit.fetch_add(1, Ordering::Relaxed);
    }

    fn hit_unique_state(&self, depth: usize, n_moves: usize) {
        self.unique_visit.fetch_add(1, Ordering::Relaxed);

        if depth < TRACK_DEPTH {
            self.move_state[depth].0.store(0, Ordering::Relaxed);
            self.move_state[depth]
                .1
                .store(n_moves as u8, Ordering::Relaxed);
        }
    }

    fn finish_move(&self, depth: usize, move_pos: usize) {
        if depth < TRACK_DEPTH {
            self.move_state[depth]
                .0
                .store(move_pos as u8, Ordering::Relaxed);
        }
    }
}

impl core::fmt::Display for AtomicSearchStats {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let (total, unique, depth) = (self.total_visit(), self.unique_visit(), self.max_depth());
        let hit = total - unique;
        write!(
            f,
            "Total visit: {}\nTransposition hit: {} (rate {})\nMiss state: {}\nMax depth search: {}\nCurrent progress:",
            total, hit, (hit as f64)/(total as f64), unique, depth,
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

pub trait SearchSignal {
    fn terminate(&self);
    fn is_terminated(&self) -> bool;
    fn search_finish(&self);
}

pub struct DefaultSearchSignal;

impl SearchSignal for DefaultSearchSignal {
    fn terminate(&self) {}

    fn is_terminated(&self) -> bool {
        false
    }

    fn search_finish(&self) {}
}

#[derive(Debug)]
pub enum SearchResult {
    Terminated,
    Solved,
    Unsolvable,
    Crashed,
}

// These are bit-mixers, to creater better hash key for the encoded game
fn _murmur64(mut h: u64) -> u64 {
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccd;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53;
    h ^= h >> 33;
    h
}

// https://zimbry.blogspot.com/2011/09/better-bit-mixing-improving-on.html
// 	31	0x7fb5d329728ea185	27	0x81dadef4bc2dd44d	33
fn murmur64_mix1(mut h: u64) -> u64 {
    h ^= h >> 31;
    h *= 0x7fb5d329728ea185;
    h ^= h >> 27;
    h *= 0x81dadef4bc2dd44d;
    h ^= h >> 33;
    h
}

fn _fast_hash(mut h: u64) -> u64 {
    h ^= h >> 23;
    h *= 0x2127599bf4325c37;
    h ^= h >> 47;
    h
}

fn _rrmxmx(mut v: u64) -> u64 {
    v ^= v.rotate_right(49) ^ v.rotate_right(24);
    v *= 0x9fb21c651e98df25;
    v ^= v >> 28;
    v *= 0x9fb21c651e98df25;
    v ^ (v >> 28)
}

fn solve(
    g: &mut Solitaire,
    rev_move: Option<Move>,
    tp: &mut TpCache,
    history: &mut HistoryVec,
    stats: &impl SearchStatistics,
    sign: &impl SearchSignal,
) -> SearchResult {
    // no need for history caching since the graph is mostly acyclic already, just prevent going to their own parent

    if sign.is_terminated() {
        return SearchResult::Terminated;
    }

    let depth = history.len();
    stats.hit_a_state(depth);

    if g.is_win() {
        return SearchResult::Solved;
    }
    let encode = murmur64_mix1(g.encode());
    if tp.get(&encode).is_some() {
        return SearchResult::Unsolvable;
    }

    tp.insert(encode, ());

    let move_list = g.list_moves::<true>();

    stats.hit_unique_state(depth, move_list.len());

    for (pos, &m) in move_list.iter().enumerate() {
        if Some(m) == rev_move {
            continue;
        }
        let rev_move = g.get_rev_move(&m);

        let undo = g.do_move(&m);
        history.push(m);

        let res = solve(g, rev_move, tp, history, stats, sign);
        if !matches!(res, SearchResult::Unsolvable) {
            return res;
        }
        history.pop();

        g.undo_move(&m, &undo);

        stats.finish_move(depth, pos);
    }

    SearchResult::Unsolvable
}

pub fn solve_game(
    g: &mut Solitaire,
    stats: &impl SearchStatistics,
    sign: &impl SearchSignal,
) -> (SearchResult, Option<HistoryVec>) {
    let mut tp = TpCache::with(
        TP_SIZE,
        TP_SIZE as u64,
        Default::default(),
        Default::default(),
        Default::default(),
    );
    let mut history = HistoryVec::new();

    let search_res = solve(g, None, &mut tp, &mut history, stats, sign);

    sign.search_finish();

    if let SearchResult::Solved = search_res {
        (search_res, Some(history))
    } else {
        (search_res, None)
    }
}
