use core::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

const TRACK_DEPTH: usize = 8;

pub trait SearchStatistics {
    fn hit_a_state(&self, depth: usize);
    fn hit_unique_state(&self, depth: usize, n_moves: usize);
    fn finish_move(&self, depth: usize);

    fn total_visit(&self) -> usize;
    fn unique_visit(&self) -> usize;
    fn max_depth(&self) -> usize;
}

pub struct EmptySearchStats;

impl SearchStatistics for EmptySearchStats {
    fn total_visit(&self) -> usize {
        Default::default()
    }

    fn unique_visit(&self) -> usize {
        Default::default()
    }

    fn max_depth(&self) -> usize {
        Default::default()
    }

    fn hit_a_state(&self, _: usize) {}
    fn hit_unique_state(&self, _: usize, _: usize) {}
    fn finish_move(&self, _: usize) {}
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

    fn finish_move(&self, depth: usize) {
        if depth < TRACK_DEPTH {
            self.move_state[depth].0.fetch_add(1, Ordering::Relaxed);
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
