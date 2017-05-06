

/*
	Node functions
*/

fn add(x: i64, y: i64) -> i64 {

	x + y

}

fn subtract(x: i64, y: i64) -> i64 {

	x - y

}

fn multiply(x: i64, y: i64) -> i64 {

	x * y

}

fn protected_divide(x: i64, y: i64) -> i64 {

	if y == 0
	{
		return 1000_000_000_000_000
	}
	x / y

}