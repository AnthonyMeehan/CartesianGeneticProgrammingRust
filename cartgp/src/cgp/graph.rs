

// The Triple struct representing a gene
// Contains a function, and two input indexes.
struct Triple {

	function: fn(i64,i64),
	input_one: i32,
	input_two: i32,

}

// The Graph struct containing the list of genes
// and the input into the Graph, and a list of 
// functions to be used as nodes
struct Graph {

	input: Vec<i64>,
	genome: Vec<Vec<Triple>>,
	nodes: Vec<fn(i64,i64)>,

}

impl Triple {


}

impl Graph {


}