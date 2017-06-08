use rand;
use rand::Rng;
use rand::ThreadRng;
use std::cmp;
use dataset;
use operations;
use std::io::stdout;
use std::io::Write;

pub type BiFunction = fn(f64,f64) -> f64;
type FunctionIndex = usize;
type Layer = Vec<Node>;
type PopulationErrors = Vec<f64>;

#[derive(Clone, Copy, Debug)]
pub enum NodeIndex {
    InputIndex(usize),
    GeneIndex(usize, usize),
}

///Gene Nodes Contain a function index, two input indices, and a possible previously-evaluated result
#[derive(Clone, Copy, Debug)]
struct GeneNode {
    function: FunctionIndex, //BiFunction
    input_node_one: NodeIndex, //Node
    input_node_two: NodeIndex, //Node
    //TODO: output: Option<f64>, separate precomputation matrix?
}

///Nodes in the graph can be inputs, or gene nodes. "Output" Nodes just reference other nodes on the graph
#[derive(Clone, Copy, Debug)]
enum Node {
    GeneNode(GeneNode),
    InputNode(f64),
}

impl Node {
    pub fn is_input(&self) -> bool {
        match *self {
            Node::InputNode(_) => true,
            Node::GeneNode(_) => false,
        }
    }

    pub fn is_gene(&self) -> bool {
        return !self.is_input();
    }

    pub fn get_input(self) -> f64 {
        match self {
            Node::InputNode(x) => x,
            Node::GeneNode(_) => panic!("called `Node::get_input()` on a `GeneNode` value"),
        }
    }

    pub fn get_gene(self) -> GeneNode {
        match self {
            Node::GeneNode(x) => x,
            Node::InputNode(_) => panic!("called `Node::get_gene()` on a `InputNode` value"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Genome {
    inner_layers: Vec<Layer>,
    output_layer: Vec<NodeIndex>, //TODO: was just 'Layer' before
}


impl Genome {
    /// Makes a randomized new genome
    pub fn new(genome_parameters: &HyperParameters, num_functions: usize, random_generator: &mut ThreadRng) -> Genome {
        let mut the_inner_layers: Vec<Layer> = Vec::with_capacity(genome_parameters.num_layers);

        //Add inner layers from left to right
        for i in 0..genome_parameters.num_layers {
            let mut inner_layer: Layer = Vec::with_capacity(genome_parameters.nodes_per_layer);
            //Add random valid nodes
            for j in 0..genome_parameters.nodes_per_layer {
                inner_layer.push(Node::GeneNode(GeneNode {
                    function: random_generator.gen_range(0, num_functions),
                    input_node_one: (Genome::get_random_node_from_layers(&the_inner_layers, i, genome_parameters.num_inputs, genome_parameters.layers_back, random_generator)),
                    input_node_two: (Genome::get_random_node_from_layers(&the_inner_layers, i, genome_parameters.num_inputs, genome_parameters.layers_back, random_generator)),
                }));
            }
            the_inner_layers.push(inner_layer);
        }

        //Add valid outputs
        let mut the_output_layer: Vec<NodeIndex> = Vec::new();
        for i in 0..genome_parameters.num_outputs {
            the_output_layer.push(Genome::get_random_node_from_layers(&the_inner_layers, the_inner_layers.len(), genome_parameters.num_inputs, genome_parameters.layers_back, random_generator));
        }

        return Genome {
            inner_layers: the_inner_layers,
            output_layer: the_output_layer,
        }
    }



    /// Pick a random node from some layer N to N-layers_back (not including N).
    /// Since nodes aren't chosen from N, N can be equal to number of function_layers,
    /// i.e. for output layer.
    fn get_random_node(&self, from_layer: usize, num_inputs: usize, layers_back: usize, random_generator: &mut ThreadRng) -> NodeIndex {
        return Genome::get_random_node_from_slice(self.inner_layers.as_slice(), from_layer, num_inputs, layers_back, random_generator);

        /*
        assert!(layers_back > 0, "layers_back must be greater than 0");
        assert!(from_layer >= 0, "from_layer must be greater than or equal to 0");
        assert!(from_layer <= self.inner_layers.len(), "from_layer must be less than or equal to number of function_layers");

        //If layers_back > from_layer, counteract bias towards input nodes
        let to_layer: isize = cmp::max(-1, (from_layer as isize) - (layers_back as isize));

        let layer_index: isize = random_generator.gen_range(to_layer, from_layer as isize);

        if layer_index < 0 {
            //choose a node index from the input layer
            let node_index = random_generator.gen_range(0, input_layer.len());
            return NodeIndex::InputIndex(node_index);
        } else {
            //choose a node index from some internal layer
            let node_index = random_generator.gen_range(0, self.inner_layers[layer_index as usize].len());
            return NodeIndex::GeneIndex(layer_index as usize, node_index);
        }
        */
    }

    fn get_random_node_from_layers(inner_layers: &Vec<Layer>, from_layer: usize, num_inputs: usize, layers_back: usize, random_generator: &mut ThreadRng) -> NodeIndex {
        return Genome::get_random_node_from_slice(inner_layers.as_slice(), from_layer, num_inputs, layers_back, random_generator);
    }

    /// From a slice of inner layers, presumably from some larger genome, pick a random node from
    /// some layer N to N-layers_back (not including N).
    ///
    /// This can be useful when mutating a node, since Rust only permits either 1 mutable reference,
    /// or many immutable references. So one might perform a split_at_mut to get two separately
    /// managed slices from the original layers.
    ///
    /// Since nodes aren't chosen from N, N can be equal to number of function_layers,
    /// i.e. when dealing with the output_layer.
    fn get_random_node_from_slice(inner_layers: &[Layer], from_layer: usize, num_inputs: usize, layers_back: usize, random_generator: &mut ThreadRng) -> NodeIndex {
        assert!(layers_back > 0, "layers_back must be greater than 0");
        assert!(from_layer >= 0, "from_layer must be greater than or equal to 0");
        assert!(from_layer <= inner_layers.len(), "from_layer must be less than or equal to number of function_layers");

        //If layers_back > from_layer, counteract bias towards input nodes
        let to_layer: isize = cmp::max(-1, (from_layer as isize) - (layers_back as isize));

        let layer_index: isize = random_generator.gen_range(to_layer, from_layer as isize);

        if layer_index < 0 {
            //choose a node index from the input layer
            let node_index = random_generator.gen_range(0, num_inputs);
            return NodeIndex::InputIndex(node_index);
        } else {
            //choose a node index from some internal layer
            let node_index = random_generator.gen_range(0, inner_layers[layer_index as usize].len());
            return NodeIndex::GeneIndex(layer_index as usize, node_index);
        }
    }

    fn mutate_node(&mut self, num_inputs: usize, layers_back: usize, output_probability: f64, num_functions: usize, random_generator: &mut ThreadRng) {
        let layer_index: usize = random_generator.gen_range(0, self.inner_layers.len()+1);

        if random_generator.gen_range(0.0, 1.0) < output_probability {
            //println!("Outer!");
            //mutate an output connection
            let output_index: usize = random_generator.gen_range(0, self.output_layer.len());
            self.output_layer[output_index] = self.get_random_node(self.inner_layers.len(), num_inputs, layers_back, random_generator);
        }
        else {
            //println!("Internal!");
            //mutate an internal node
            let layer_index: usize = random_generator.gen_range(0, self.inner_layers.len());

            //need to slice layers in two in order to deal with 1 immutable and 1 mutable reference
            let (below_layer, at_or_above_layer) = self.inner_layers.split_at_mut(layer_index);

            //let node_index: usize = random_generator.gen_range(0, at_or_above_layer[0].len());

            let mut node_to_mutate: &mut Node = random_generator.choose_mut(at_or_above_layer[0].as_mut_slice()).expect("Chose a nonexistent out-of-bounds inner layer.");
            match node_to_mutate {
                &mut Node::GeneNode(ref mut gene) => {
                    //pick between function and two inputs
                    match random_generator.gen_range(0, 3) {
                        //function: pick index from a list of functions //TODO: Use the list directly?
                        0 => gene.function = random_generator.gen_range(0, num_functions),
                        1 => gene.input_node_one = Genome::get_random_node_from_slice(below_layer, layer_index, num_inputs, layers_back,  random_generator),
                        2 => gene.input_node_two = Genome::get_random_node_from_slice(below_layer, layer_index, num_inputs, layers_back,  random_generator),
                        _ => panic!("Generated outside range"),
                    }

                }
                &mut Node::InputNode(_) => panic!("Expected GeneNode in internal layer, got Input instead")
            }
        }
    }

    fn mutate_nodes(&mut self, num_mutations: usize, num_inputs: usize, layers_back: usize, num_functions: usize, random_generator: &mut ThreadRng) {
        let output_probability: f64 = self.get_output_probability();
        //println!("Outprob: {}", output_probability);
        for _ in 0..num_mutations {
            self.mutate_node(num_inputs, layers_back, output_probability, num_functions, random_generator);
        }
    }

    ///
    /// Probability that an output node will be chosen at random
    ///
    fn get_output_probability(&self) -> f64 {
        return self.output_layer.len() as f64 / (self.output_layer.len() as f64 + (self.inner_layers[0].len() as f64 * self.inner_layers.len() as f64));
    }


    /*
    //Gets messy with now adding length of input layer, layers back
    fn mutate_nodes(&self, expected_muts: usize, input: &Layer, layers_back: usize, f: usize) -> Genome {
        let mut hacks = Genome {
            inner_layers: self.inner_layers.clone(),
            output_layer: self.output_layer.clone(),
        };

        let mut nodes: usize = self.output_layer.len();
        for x in &(self.inner_layers) {
            nodes = nodes + x.len();
        }

        let mutation_chance: f64 = (expected_muts as f64) / (nodes as f64);

        for index in 0..hacks.inner_layers.len() {
            let mut x = &mut (hacks.inner_layers[index]);
            for mut node in x {
                if rand::thread_rng().gen::<f64>() < mutation_chance {
                    let value = rand::thread_rng().gen_range(0, 3);
                    match node {
                        &mut Node::GeneNode(ref mut gene) =>
                            match value {
                                0 => gene.input_node_one = self.get_random_node(index, layers_back, input, &mut rand::thread_rng()),
                                1 => gene.input_node_two = self.get_random_node(index, layers_back, input, &mut rand::thread_rng()),
                                2 => gene.function = rand::thread_rng().gen_range(0, f),
                                _ => panic!("WTF"),
                            },
                        _ => panic!("There's an input node in the hidden layers!"),
                    }
                }
            }
        }
        let index = hacks.inner_layers.len();
        for mut node in &mut (hacks.output_layer) {
            if rand::thread_rng().gen::<f64>() < mutation_chance {
                let value = rand::thread_rng().gen_range(0, 3);
                match node {
                    &mut Node::GeneNode(ref mut gene) =>
                        match value {
                            0 => gene.input_node_one = self.get_random_node(index, layers_back, input, &mut rand::thread_rng()),
                            1 => gene.input_node_two = self.get_random_node(index, layers_back, input, &mut rand::thread_rng()),
                            2 => gene.function = rand::thread_rng().gen_range(0, f),
                            _ => panic!("WTF"),
                        },
                    _ => panic!("There's an input node in the hidden layers!"),
                };
            }
        }
        hacks
    }*/


    fn apply_fn(&self, function_layer: &Vec<BiFunction>, gene: &GeneNode, input_layer: &Layer) -> f64 {
        //Evaluate new result recursively
        //let input_one = self.get_node(&gene.input_node_one, input_layer);
        let result_one = self.evaluate_node(&gene.input_node_one, input_layer, function_layer);
        //let input_two = self.get_node(&gene.input_node_two, input_layer);
        let result_two = self.evaluate_node(&gene.input_node_two, input_layer, function_layer);
        return (function_layer[gene.function])(result_one, result_two);
    }

    //TODO: Correct usage of lifetimes?
    fn get_node<'a>(&'a self, i: &NodeIndex, input_layer: &'a Layer) -> &'a Node {
        match i {
            &NodeIndex::InputIndex(j) => &input_layer[j],
            &NodeIndex::GeneIndex(j, k) => &self.inner_layers[j][k],
        }
    }

    fn evaluate_node(&self, the_node_i: &NodeIndex, input_layer: &Layer, function_layer: &Vec<BiFunction>) -> f64 {
        let the_node: &Node = self.get_node(the_node_i, input_layer);
        match the_node {
            //Input is effectively constant
            &Node::InputNode(input) => input,
            &Node::GeneNode(ref gene) => self.apply_fn(function_layer, gene, input_layer),
            //TODO: Save results to some "precomputed" table, would make multi-output faster. Dealing with recursive mutable structures is tricky.
            /*{
                match gene.output {
                    //Return recent result
                    Some(result) => result,
                    //Evaluate new result recursively
                    None => {
                        return self.apply_fn(function_layer, gene, input_layer);

                    }
                }
            }*/
        }
    }

    /// Evaluates all output nodes separately, for fitness evaluation
    fn get_outputs(&self, input_layer: &Layer, function_layer: &Vec<BiFunction>) -> Vec<f64> {
        let mut result: Vec<f64> = Vec::with_capacity(self.output_layer.len());
        for output in self.output_layer.iter() {
            result.push(self.evaluate_node(&output, input_layer, function_layer));
        }
        return result;
    }

    /// Returns the mean-squared error or mean-absolute error for this gene, given an expected output
    fn error_on_example(&self, expected_output: &Vec<f64>, use_mean_squared: bool, input_layer: &Layer, function_layer: &Vec<BiFunction>) -> f64 {
        assert!(expected_output.len() == self.output_layer.len(), "Gene output must have same size as expected output");
        let outputs: Vec<f64> = self.get_outputs(input_layer, function_layer);
        let mut cumulative_error: f64 = 0.0;
        for i in 0..outputs.len() {
            let difference: f64 = outputs[i] - expected_output[i];
            if use_mean_squared {
                cumulative_error += difference * difference;
            }
            else {
                cumulative_error += difference.abs();
            }
        }
        return cumulative_error / outputs.len() as f64;
    }

    /// Returns the average of the mean-squared error or mean-absolute error for this genome,
    /// across a dataset of examples
    fn error_on_dataset(&self, the_dataset: &dataset::Dataset, use_mean_squared: bool, function_layer: &Vec<BiFunction>) -> f64 {
        assert!(the_dataset.input_examples.len() == the_dataset.output_examples.len(), "Mismatch between length of input and output examples.");
        let mut cumulative_error: f64 = 0.0;
        for i in 0..the_dataset.input_examples.len() {
            let the_input = vec![Node::InputNode(the_dataset.input_examples[i])];
            let the_output = vec![the_dataset.output_examples[i]];
            cumulative_error += self.error_on_example(&the_output, use_mean_squared, &the_input, function_layer);
        }
        return cumulative_error / the_dataset.input_examples.len() as f64;
    }
}

#[derive(Clone, Debug)]
pub struct HyperParameters {
    pub num_genomes: usize,
    pub num_inputs: usize,
    pub num_layers: usize,
    pub nodes_per_layer: usize,
    pub num_outputs: usize,
    pub layers_back: usize,
}

/// The Graph struct containing the list of genes, and the inputs to the Graph,
/// as well as a list of functions to be used as nodes
#[derive(Clone, Debug)]
pub struct Graph {
    //dataset: dataset::Dataset,
    //inputs: Layer,
    //TODO: constants: Vec<f64>,
    pub hyperparameters: HyperParameters,
    pub functions: Vec<BiFunction>,
    pub genomes: Vec<Genome>,
    pub errors: Option<PopulationErrors>,
}

impl Graph {
    pub fn new(genome_parameters: HyperParameters, the_functions: Vec<BiFunction>, random_generator: &mut ThreadRng) -> Graph {
        let mut the_genomes: Vec<Genome> = Vec::with_capacity(genome_parameters.num_genomes);

        //Initialise inputs to empty values
        let mut initial_inputs: Layer = Vec::with_capacity(genome_parameters.num_inputs);
        for i in 0..genome_parameters.num_inputs {
            initial_inputs.push(Node::InputNode(0.0));
        }

        for i in 0..genome_parameters.num_genomes {
            the_genomes.push(Genome::new(&genome_parameters, the_functions.len(), random_generator));
        }

        return Graph {
            //inputs: initial_inputs,
            hyperparameters: genome_parameters,
            functions: the_functions,
            genomes: the_genomes,
            errors: Option::None,
        };
    }


    fn get_random_fn(&self, random_generator: &mut ThreadRng) -> FunctionIndex {
        return random_generator.gen_range(0, self.functions.len());
    }

    /// Prints the inputs, genomes and outputs for the graph to stdout
    pub fn print_graph(&self, graph_name: &str) {
        println!("\nGraph: {}", graph_name);
        print!("Num_Inputs: {}, Layers_Back: {}", self.hyperparameters.num_inputs, self.hyperparameters.layers_back);
        print!("\nGenomes:");
        for (i, genome) in self.genomes.iter().enumerate() {
            println!("\n\tGenome {}: ", i);
            for (j, layer) in genome.inner_layers.iter().enumerate() {
                println!("\t\tInner Layer {}: ", j);
                for (k, node) in layer.iter().enumerate() {
                    let gene = node.get_gene();
                    println!("\t\t\tNode{}: (fn:{}, i1:{:?}, i2:{:?}, ",
                             k, gene.function, gene.input_node_one, gene.input_node_two);
                }
            }
            print!("\t\tOutputs: ");
            for (j, output) in genome.output_layer.iter().enumerate() {
                print!("{}: {:?}, ", j, output)
            }
        }
        println!()
    }

    /// Evaluate the average of the mean-squared error or mean-absolute error for all genomes,
    /// across a dataset of examples
    fn evaluate_population_errors(&mut self, the_dataset: &dataset::Dataset, use_mean_squared: bool) -> PopulationErrors {
        let mut population_errors: PopulationErrors = Vec::with_capacity(self.genomes.len());
        for genome in self.genomes.iter() {
            population_errors.push(genome.error_on_dataset(the_dataset, use_mean_squared, &self.functions));
        }
        return population_errors;
    }

    /// Select the genome with the lowest error out of all genomes
    fn top_1_selection(&self) -> (&Genome, f64) {
        let errors = self.errors.as_ref().unwrap();
        //Best solution from the previous generation, if top-1 selection was in effect
        let mut best_genome: &Genome = &self.genomes[0];
        let mut lowest_error: f64 = errors[0];

        for i in 1..self.genomes.len() {
            //Prefer new, equally good solutions over the previous best solution
            if errors[i] <= lowest_error {
                best_genome = &self.genomes[i];
                lowest_error = errors[i];
            }
        }
        return (best_genome, lowest_error);
    }

    /// Select genome with the lowest error out of N random samples from all genomes
    fn tournament_selection(&self, tournament_size: usize, random_generator: &mut ThreadRng) -> &Genome {
        let errors = self.errors.as_ref().unwrap();
        //Winner is the best out of N random samples
        let mut winning_index: usize = random_generator.gen_range(0, self.genomes.len());
        for i in 1..tournament_size {
            let contender_index: usize = random_generator.gen_range(0, self.genomes.len());
            if errors[contender_index] < errors[winning_index] {
                winning_index = contender_index;
            }
        }
        return &self.genomes[winning_index];
    }

    fn new_mutated_genome(&self, original: &Genome, num_mutations: usize, random_generator: &mut ThreadRng) -> Genome {
        let mut mutated_solution: Genome = original.clone();
        mutated_solution.mutate_nodes(
            num_mutations,
            self.hyperparameters.num_inputs,
            self.hyperparameters.layers_back,
            self.functions.len(),
            random_generator
        );
        return mutated_solution;
    }

    /// Apply elitist top-1 selection(when tournament_size = 0), or tournament selection(for any
    /// other tournament_size) to generate the next generation of solutions. Returns the best
    /// solution from this new generation.
    fn next_generation(&mut self, the_dataset: &dataset::Dataset, tournament_size: usize, num_mutations: usize,  random_generator: &mut ThreadRng) -> (&Genome, f64) {
        if self.errors.is_none() {
            self.errors = Some(self.evaluate_population_errors(the_dataset, true));
        }

        let mut new_errors: PopulationErrors = Vec::with_capacity(self.genomes.len());
        let mut new_genomes: Vec<Genome> = Vec::with_capacity(self.genomes.len());

        let mut best_result: (&Genome, f64);

        if tournament_size == 0 {
            //Use elitist top-1 selection
            let (parent, parent_error): (&Genome, f64) = self.top_1_selection();
            //println!("Lowest_error: {}", parent.error_on_dataset(&the_dataset, true, &self.functions));
            new_genomes.push(parent.clone());
            new_errors.push(parent_error);

            for i in 1..self.genomes.len() {
                let child: Genome = self.new_mutated_genome(parent, num_mutations, random_generator);
                let mutated_error: f64 = child.error_on_dataset(the_dataset, true, &self.functions);
                new_genomes.push(child);
                new_errors.push(mutated_error);
            }
        }
        else {
            //Use tournament selection
            let mut intermediate_population: Vec<&Genome> = Vec::with_capacity(self.genomes.len());
            for _ in 0..self.genomes.len() {
                intermediate_population.push(self.tournament_selection(tournament_size, random_generator));
            }
            for parent in intermediate_population {
                // Only point mutation is currently supported
                let child = self.new_mutated_genome(parent, num_mutations, random_generator);
                let child_error = child.error_on_dataset(the_dataset, true, &self.functions);
                new_genomes.push(child);
                new_errors.push(child_error);
            }
        }

        self.genomes = new_genomes;
        self.errors = Some(new_errors);
        // Keep track of the best solution so far
        return self.top_1_selection();
    }


    pub fn run(&mut self, num_iterations: usize, display_progress: bool, the_dataset: &dataset::Dataset, tournament_size: usize, num_mutations: usize, random_generator: &mut ThreadRng) -> (Genome, f64) {
        if self.errors.is_none() {
            self.errors = Some(self.evaluate_population_errors(the_dataset, true));
        }

        let mut best_genome: Genome;
        let mut lowest_error;
        //Isolate borrow to this scope
        {
            let (mut first_genome, mut first_error): (&Genome, f64) = self.top_1_selection();
            best_genome = first_genome.clone();
            lowest_error = first_error;
        }

        //For a nice logarithmically-scaled progress indicator
        let mut last_int_log_result: usize = 0;
        let mut iterations_without_result: usize = 0;

        if display_progress { print!("\n\nInitial, Lowest error: {}", lowest_error); }

        for i in 0..num_iterations {
            let (candidate_genome, candidate_error) = self.next_generation(the_dataset, tournament_size, num_mutations, random_generator);
            if candidate_error < lowest_error {
                if display_progress {
                    print!("\nIteration {}/{}, Lowest error: {}", i, num_iterations, lowest_error);
                    last_int_log_result = 0;
                    iterations_without_result = 0;
                }

                best_genome = candidate_genome.clone();
                lowest_error = candidate_error;
            }
            //Display progress proportional to the log of the lack of progress
            else if display_progress {
                iterations_without_result += 1;
                let log_result = ((iterations_without_result as f64).log2()) as usize;
                if log_result > last_int_log_result {
                    last_int_log_result = log_result;
                    print!(".");
                    stdout().flush();
                }
            }
        }

        if display_progress {
            println!("\nFinal, Lowest error: {}\n", lowest_error);
            //self.print_graph("Final Graph");
        }
        return (best_genome, lowest_error);
    }

	/*
	fn run(&self, expected_outputs: &Vec<f64>, muts: usize, layers_back: usize,rng: &mut ThreadRng) {
		let mut max: f64 = 0.0;
		let mut g_ref: &Genome = &self.genomes[0];
		for genome in self.genomes {
			let x = genome.test(expected_outputs);
			if x > max:
				max = x;
				g_ref = &genome;
		}
		for mut genome in mut self.genomes {
			genome = *g_ref.clone();
		}
		for i in 0..self.genomes.len() {
			let g = &mut self.genomes[i];
			g = g.new_mutate_nodes(muts, self.inputs, layers_back, self.functions.len(), rng);
		}
	}

	*/

    /*
	Won't work yet due to mutable stuff, and there is no genome.error!
	fn update(&mut self,expected_muts: usize, layers_back: usize) {
		let mut max = 0.0;
		let mut max_genome = 0;
		for (index,genome) in self.genomes.iter().enumerate() {
			let result = genome.error();
			if result > max:
				max = result;
				max_genome = index;
		}
		for (index,genome) in self.genomes.iter().enumerate() {
			if index != max_genome {
				genome = self.genomes[max_genome].clone();
				genome.mutate_nodes(expected_muts, self.inputs, layers_back, self.functions.len());
			}
		}
	}
	*/
}


/*
struct GraphBuilder {
    genome_count: i32,
    inputs: Layer,
    functions: Vec<BiFunction>,
    hidden_layers: Vec<Layer>,
    output: Layer,
    levels: usize,
}

/*
/*
	Graph builder
		new(genomes) initializes builder with given genome count.
		methods are:
			add_input(input vector) adds input vector
			add_functions(function vector) adds functions
			add_hidden(size) adds a hidden layer of given size
			add_output(size) adds an output layer of given size
			levels(levels_back) sets levels back
			build() builds graph from GraphBuilder
			
	TODO: Force restraints on configuration (ex. one output layer)
*/
impl GraphBuilder {
    fn new(genomes: i32) -> GraphBuilder {
        GraphBuilder {
            inputs: Layer::new(),
            functions: Vec::new(),
            hidden_layers: Vec::new(),
            output: Layer::new(),
            genome_count: genomes,
            levels: 1,
        }
    }

    fn add_input(&mut self, input: Vec<f64>) {
        for x in input {
            self.inputs.push(Node::InputNode(x));
        }
    }

    fn add_hidden(&mut self, size: i32) {
        self.hidden_layers.push(Layer::new());
        let length: usize = self.hidden_layers.len() - 1;
        for index in 0..size {
            self.hidden_layers[length].push(Node::GeneNode(GeneNode {
                function: 0,
                //BiFunction
                input_node_one: NodeIndex::InputIndex(0),
                //Node
                input_node_two: NodeIndex::InputIndex(0),
                //Node
            }
            ));
        }
    }

    fn add_output(&mut self, size: i32) {
        self.output = Layer::new();
        for index in 0..size {
            self.output.push(Node::GeneNode(GeneNode {
                function: 0,
                //BiFunction
                input_node_one: NodeIndex::InputIndex(0),
                //Node
                input_node_two: NodeIndex::InputIndex(0),
                //Node
            }
            ));
        }
    }

    fn add_functions(&mut self, funcs: Vec<BiFunction>) -> &mut GraphBuilder {
        self.functions = funcs;
        self
    }
    fn levels(&mut self, levels: usize) -> &mut GraphBuilder {
        self.levels = levels;
        self
    }
    fn build(&mut self) -> Graph {
        let mut g = Vec::new();
        for index in 0..self.genome_count {
            let mut gene = Genome {
                inner_layers: Vec::new(),
                output_layer: Layer::new(),

            };
            gene.inner_layers = self.hidden_layers.clone();
            gene.output_layer = self.output.clone();
            g.push(gene);
        }
        let mut graph = Graph {
            inputs: self.inputs.clone(),
            functions: self.functions.clone(),
            genomes: g,
        };

        let function_len = graph.functions.len();
        let input_len = graph.inputs.len();
        let layers_back = self.levels;

        let mut layer_lens = Vec::new();
        let graph = graph;
        for layer in &(graph.genomes[0].inner_layers) {
            layer_lens.push(layer.len());
        }
        let layer_lens = layer_lens;
        let mut graph = graph;

        // TODO: randomize
        for mut genome in &mut graph.genomes {
            for mut gene in &mut genome.inner_layers[0] {
                *gene = Node::GeneNode(GeneNode {
                    input_node_one: NodeIndex::InputIndex(rand::thread_rng().gen_range(0, input_len)),
                    input_node_two: NodeIndex::InputIndex(rand::thread_rng().gen_range(0, input_len)),
                    function: rand::thread_rng().gen_range(0, function_len),
                });
            }

            for index in 1..(genome.inner_layers.len()) {
                for mut gene in &mut genome.inner_layers[index] {
                    // TODO: Allow genes to connect to both an input and gene node
                    let back = if layers_back != 1 {
                        rand::thread_rng().gen_range(1, layers_back)
                    } else {
                        1
                    };
                    let to_layer: isize = cmp::max(-1, (index as isize) - (back as isize));
                    if to_layer < 0 {
                        let to_layer = to_layer as usize;
                        *gene = Node::GeneNode(GeneNode {
                            input_node_one: NodeIndex::InputIndex(rand::thread_rng().gen_range(0, input_len)),
                            input_node_two: NodeIndex::InputIndex(rand::thread_rng().gen_range(0, input_len)),
                            function: rand::thread_rng().gen_range(0, function_len),
                        });
                    } else {
                        let to_layer = to_layer as usize;
                        *gene = Node::GeneNode(GeneNode {
                            input_node_one: NodeIndex::GeneIndex(to_layer, rand::thread_rng().gen_range(0, layer_lens[index - 1])),
                            input_node_two: NodeIndex::GeneIndex(to_layer, rand::thread_rng().gen_range(0, layer_lens[index - 1])),
                            function: rand::thread_rng().gen_range(0, function_len),
                        });
                    }
                }
            }
            for mut gene in &mut genome.output_layer {
                // TODO: Allow genes to connect to both an input and gene node
                let index = layer_lens.len();
                let back = if layers_back != 1 {
                    rand::thread_rng().gen_range(1, layers_back)
                } else {
                    1
                };
                let to_layer: isize = cmp::max(-1, (index as isize) - (back as isize));
                if to_layer < 0 {
                    let to_layer = to_layer as usize;
                    *gene = Node::GeneNode(GeneNode {
                        input_node_one: NodeIndex::InputIndex(rand::thread_rng().gen_range(0, input_len)),
                        input_node_two: NodeIndex::InputIndex(rand::thread_rng().gen_range(0, input_len)),
                        function: rand::thread_rng().gen_range(0, function_len),
                    });
                } else {
                    let to_layer = to_layer as usize;
                    *gene = Node::GeneNode(GeneNode {
                        input_node_one: NodeIndex::GeneIndex(to_layer, rand::thread_rng().gen_range(0, layer_lens[index - 1])),
                        input_node_two: NodeIndex::GeneIndex(to_layer, rand::thread_rng().gen_range(0, layer_lens[index - 1])),
                        function: rand::thread_rng().gen_range(0, function_len),
                    });
                }
            }
        }
        let graph = graph;
        graph
    }
}
*/
*/

#[test]
fn test_graph_on_dataset() {
    let mut new_graph: Graph = Graph::new(
        HyperParameters {
            num_genomes: 100,
            num_inputs: 1,
            num_layers: 5,
            nodes_per_layer: 40,
            num_outputs: 1,
            layers_back: 2,
        },
        operations::get_basic_operations(),
        &mut rand::thread_rng()
    );

    let the_dataset: dataset::Dataset = dataset::dataset_from_csv(String::from("src\\sin.csv"));

    new_graph.print_graph("For sine data");

    for (i, genome) in new_graph.genomes.iter().enumerate() {
        println!("Genome {} MSE: {}", i, genome.error_on_dataset(&the_dataset, true, &new_graph.functions));
    }

    new_graph.run(2000, true, &the_dataset, 0, 100, &mut rand::thread_rng());
}

#[test]
fn test_graph() {
    //Manually construct a graph
    let input1 = Node::InputNode(0.0);
    let input1_i = NodeIndex::InputIndex(0);
    let input2 = Node::InputNode(1.0);
    let input2_i = NodeIndex::InputIndex(1);

    fn op1(x: f64, y: f64) -> f64 { x + y };
    let op1_i = 0;
    fn op2(x: f64, y: f64) -> f64 { x - y };
    let op2_i = 1;

    let gene1 = Node::GeneNode(GeneNode {
        function: op1_i,
        input_node_one: input2_i,
        input_node_two: input2_i,
    });
    let gene1_i = NodeIndex::GeneIndex(0, 0);

    let gene2 = Node::GeneNode(GeneNode {
        function: op2_i,
        input_node_one: input1_i,
        input_node_two: input2_i,
    });
    let gene2_i = NodeIndex::GeneIndex(0, 1);

    let genome = Genome {
        inner_layers: vec![vec![gene1, gene2]],
        output_layer: vec![input1_i, gene1_i, gene2_i],
    };

    let the_inputs = vec![input1, input2];

    let the_graph = Graph {
        //inputs: vec![input1, input2],
        hyperparameters: HyperParameters {
            num_genomes: 1,
            num_inputs: 2,
            num_layers: 1,
            nodes_per_layer: 2,
            num_outputs: 3,
            layers_back: 1,
        },
        functions: vec![op1, op2],
        genomes: vec![genome],
        errors: None,
    };

    let result1 = the_graph.genomes[0].evaluate_node(&the_graph.genomes[0].output_layer[0], &the_inputs, &the_graph.functions);
    let result2 = the_graph.genomes[0].evaluate_node(&the_graph.genomes[0].output_layer[1], &the_inputs, &the_graph.functions);
    let result3 = the_graph.genomes[0].evaluate_node(&the_graph.genomes[0].output_layer[2], &the_inputs, &the_graph.functions);
    assert_eq!(result1, 0.0);
    assert_eq!(result2, 2.0);
    assert_eq!(result3, -1.0);

    the_graph.print_graph("Initial");

    let mut new_graph = the_graph;
    new_graph.genomes[0].mutate_nodes(12, 2, 2, 4, &mut rand::thread_rng());

    new_graph.print_graph("Mutated");


    let gen_graph = Graph::new(
        HyperParameters {
            num_genomes: 2,
            num_inputs: 2,
            num_layers: 2,
            nodes_per_layer: 2,
            num_outputs: 3,
            layers_back: 2,
        },
        vec![op1, op2],
        &mut rand::thread_rng()
    );

    gen_graph.print_graph("Generated");

    for genome in gen_graph.genomes.iter() {
        println!("{:?}", genome.get_outputs(&vec![input1, input2], &vec![op1, op2]));
    }

    let gen_graph2 = gen_graph.clone();
    gen_graph2.print_graph("Cloned");

    println!("Error vs all-1s:");
    for genome in gen_graph2.genomes.iter() {
        println!("{:?}", genome.error_on_example(&vec![1.0, 1.0, 1.0], true, &vec![input1, input2], &vec![op1, op2]));
    }


    /*
    let inp = vec![0.1, 0.2, 0.3, 0.4];
    let fns = [op1, op2].to_vec();
    let new_graph = GraphBuilder::new(1)
        .add_input(inp)
        .add_hidden(8)
        .add_hidden(4)
        .add_hidden(2)
        .add_output(1)
        .add_functions(fns)
        .levels(1)
        .build();

    let mut g = new_graph.genomes[0].mutate_nodes(10, &(new_graph.inputs), 1, new_graph.functions.len());

    let result1 = new_graph.genomes[0].evaluate(&new_graph.genomes[0].output_layer[0], &new_graph.inputs, &new_graph.functions);*/
}