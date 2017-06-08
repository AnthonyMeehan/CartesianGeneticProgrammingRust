extern crate cgp_rs;
extern crate rand;

fn main() {
    println!("Welcome to cgp_rs!");
	
	/*
	
		The following example shows how to use cgp_rs.
		
		Steps:
			
			1. Build dataset
			2. Setup graph
			3. Run
	
	*/
	
	// Initialize the data set from a csv using dataset::dataset_from_csv()
	
	let data: cgp_rs::dataset::Dataset = cgp_rs::dataset::dataset_from_csv(String::from("src\\sin.csv"));
	
	// Initialize the graph - graph takes a HyperParameters struct as an argument,
	// the functions to use, and a rng. Note it needs to be a mutable binding
	
    let mut new_graph: cgp_rs::graph::Graph = cgp_rs::graph::Graph::new(
	
        cgp_rs::graph::HyperParameters {
            num_genomes: 100,    // set population size
            num_inputs: 1,       // set number of inputs
            num_layers: 5,       // number of hidden layers
            nodes_per_layer: 40, // width of each layer
            num_outputs: 1,      // number of outputs
            layers_back: 2,      // layers back: how far back a node can be connected 
        },
		
        cgp_rs::operations::get_basic_operations(), // set the functions. This is a vector of graph::BiFunction
											// operations::get_basic_operations returns +,-,*,/ (protected)
        &mut rand::thread_rng()				// set the random number generator
		
    );
	
	// With the graph set, now we can run it!
	// Use the arguments
	// num_iterations: usize , the number of iterations to run for
	// display_progress: bool , if true, displays errors as graph is run
	// the_dataset: &dataset::Dataset , a reference to the dataset
	// tournament_size: usize , the size of the tournament selection
	// num_mutations: usize , number of mutations per mutate
	// random_generator: &mut ThreadRng , the rng
	
	new_graph.run(2000, true, &data, 0, 100, &mut rand::thread_rng());
	
}
