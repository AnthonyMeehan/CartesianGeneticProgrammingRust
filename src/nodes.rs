

/*
	Node functions
*/

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
		return 1000_000_000_000_000.0
	}
	x / y

}

