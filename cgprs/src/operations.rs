///
/// Operations which can be used by Cartesian-genetic-programming genomes
///
use graph;


fn add(x: f64, y: f64) -> f64 {
	x + y
}

fn subtract(x: f64, y: f64) -> f64 {
	x - y
}

fn multiply(x: f64, y: f64) -> f64 {
	x * y
}

fn protected_divide(x: f64, y: f64) -> f64 {
	if y == 0.0
	{
		return 1_000_000_000_000_000.0
	}
	x / y
}

pub fn get_basic_operations() -> Vec<graph::BiFunction> {
    return vec![add, subtract, multiply, protected_divide];
}