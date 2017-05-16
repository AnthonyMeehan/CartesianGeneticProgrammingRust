extern crate rand; // Random number generators




mod graph;
mod nodes;
mod fitness_fns;

#[cfg(test)]
mod tests {
    use fitness_fns;

    #[test]
    fn it_works() {
    }

    #[test]
    #[should_panic]
    fn not_working() {
        assert!(false);
    }

    /*
    #[test]
    fn test_sin() {
        assert!(fitness_fns::sin(0.0) == 0.0);
        let one: f64 = 1.0;
        assert!(fitness_fns::sin(1.0) == one.sin());
        let minus_one: f64 = -1.0;
        assert!(fitness_fns::sin(1.0) == minus_one.sin());
    }*/


}

