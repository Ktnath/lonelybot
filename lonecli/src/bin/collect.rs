fn main() {
    if let Err(e) = lonecli::training::collect_training_data(10_000_000) {
        eprintln!("{e}");
    }
}
