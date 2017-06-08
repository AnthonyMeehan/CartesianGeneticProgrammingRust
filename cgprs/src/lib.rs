extern crate rand; // Random number generators

pub mod graph;
pub mod operations;
pub mod dataset;

#[cfg(test)]
mod tests {
    use dataset;
    use graph;

    #[test]
    fn it_works() {
    }

    #[test]
    #[should_panic]
    fn not_working() {
        assert!(false);
    }

}

