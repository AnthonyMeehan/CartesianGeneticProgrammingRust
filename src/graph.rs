use rand;
use rand::Rng;
use rand::ThreadRng;
use std::cmp;

type BiFunction = fn(f64,f64) -> f64;
type FunctionIndex = usize;

#[derive(Clone, Copy,Debug)]
enum NodeIndex {
    InputIndex(usize),
    GeneIndex(usize, usize),
}

///Gene Nodes Contain a function index, two input indices, and a possible previously-evaluated result
#[derive(Clone, Copy,Debug)]
struct GeneNode {
    function: FunctionIndex, //BiFunction
    input_node_one: NodeIndex, //Node
    input_node_two: NodeIndex, //Node
    //TODO: output: Option<f64>, separate precomputation matrix?
}

///Nodes in the graph can be inputs, or gene nodes. "Output" Nodes just reference other nodes on the graph
#[derive(Clone, Copy,Debug)]
enum Node {
    GeneNode(GeneNode),
    InputNode(f64),
}

impl Node {

    // Constructor
    /*
    fn new(function: fn(f64,f64), input_one: i32, input_two: i32) -> BiFunction {
        BiFunction {
            function: function,
            input_one: input_one,
            input_two: input_two,
        }
    }*/

    /*
    fn random(functions: &Vec<BiFunction>, max_int: i32) -> Node {
        Node {
            function: *rand::thread_rng().choose(functions).unwrap(),
            input_node_one: rand::thread_rng().gen_range(0, max_int), //task_rng
            input_node_two: rand::thread_rng().gen_range(0, max_int),
        }
    }*/


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

    /*
    pub fn get_gene(self) -> GeneNode {
        match self {
            Node::GeneNode(x) => x,
            Node::InputNode(_) => panic!("called `Node::get_gene()` on a `InputNode` value"),
        }
    }*/


}

type Layer = Vec<Node>;
#[derive(Debug)]
struct Genome {
    inner_layers: Vec<Layer>,
    output_layer: Layer,
}


impl Genome {

    /// Pick a random node from some layer N to N-layers_back (not including N).
    /// Since nodes aren't chosen from N, N can be equal to number of function_layers,
    /// i.e. for output layer.
    fn get_random_node(&self, from_layer: usize, layers_back: usize, input_layer: &Layer, random_generator: &mut ThreadRng) -> NodeIndex {
        assert!(layers_back > 0, "layers_back must be greater than 0");
        assert!(from_layer >= 0, "from_layer must be greater than or equal to 0");
        assert!(from_layer <= self.inner_layers.len(), "from_layer must be less than or equal to number of function_layers");

        //If layers_back > from_layer, counteract bias towards input nodes
        let to_layer: isize = cmp::max(-1, (from_layer as isize) - (layers_back as isize));

        let layer_index : isize = random_generator.gen_range(to_layer, from_layer as isize);

        if layer_index < 0 {
            //choose a node index from the input layer
            let node_index = random_generator.gen_range(0, input_layer.len());
            return NodeIndex::InputIndex(node_index);
        }
            else {
                //choose a node index from some internal layer
                let node_index = random_generator.gen_range(0, self.inner_layers[layer_index as usize].len());
                return NodeIndex::GeneIndex(layer_index as usize, node_index);
            }
    }
    
	//Gets messy with now adding length of input layer, layers back 
    fn mutate_nodes(&self, expected_muts: usize, input: &Layer, layers_back: usize, f: usize) -> Genome{
		
		let mut hacks = Genome {
		inner_layers: self.inner_layers.clone(),
		output_layer: self.output_layer.clone(),};
		
		let mut nodes: usize = self.output_layer.len();
		for x in &(self.inner_layers) {
			nodes = nodes + x.len();
		}
		
		let mutation_chance: f64 = (expected_muts as f64) / (nodes as f64);
		
		for index in 0..hacks.inner_layers.len() {
			let mut x = &mut (hacks.inner_layers[index]);
			for mut node in x {
				if rand::thread_rng().gen::<f64>() < mutation_chance {
					let value = rand::thread_rng().gen_range(0,3);
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
				let value = rand::thread_rng().gen_range(0,3);
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
    }
	

    fn apply_fn(&self, function_layer: &Vec<BiFunction>, gene: &GeneNode, input_layer: &Layer) -> f64 {
        //Evaluate new result recursively
        let input_one = self.get_node(&gene.input_node_one, input_layer);
        let result_one = self.evaluate(input_one, input_layer, function_layer);
        let input_two = self.get_node(&gene.input_node_two, input_layer);
        let result_two = self.evaluate(input_two, input_layer, function_layer);
        return (function_layer[gene.function])(result_one, result_two);
    }

    //TODO: Correct usage of lifetimes?
    fn get_node<'a>(&'a self, i: &NodeIndex, input_layer: &'a Layer) -> &'a Node {
        match i {
            & NodeIndex::InputIndex(j) => & input_layer[j],
            & NodeIndex::GeneIndex(j, k) => & self.inner_layers[j][k],
        }
    }

    fn evaluate(&self, the_node: &Node, input_layer: &Layer, function_layer: &Vec<BiFunction>) -> f64 {
        match the_node {
            //Input is effectively constant
            &Node::InputNode(input) => input,
            //TODO: Gene may or may not have been evaluated recently
            &Node::GeneNode(ref gene) => self.apply_fn(function_layer, gene, input_layer),
            /*{
                match gene.output {
                    //Return recent result
                    Some(result) => result,
                    //Evaluate new result recursively
                    None => {
                        return self.apply_fn(function_layer, gene, input_layer);
                        //TODO: Save results to some "precomputed" table
                        //TODO: Report: Tried precomputing inline, dealing with recursive mutable structures is hard...
                    }
                }
            }*/
        }
    }
}

// The Graph struct containing the list of genes
// and the input into the Graph, and a list of
// functions to be used as nodes
#[derive(Debug)]
struct Graph {
    inputs: Layer,
    //TODO: constants: Vec<f64>,
    functions: Vec<BiFunction>,
    genomes: Vec<Genome>,
}

impl Graph {
    fn get_random_fn(&self, random_generator: &mut ThreadRng) -> FunctionIndex {
        return random_generator.gen_range(0, self.functions.len());
    }
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

struct GraphBuilder {
	genome_count: i32,
	inputs: Layer,
	functions: Vec<BiFunction>,
	hidden_layers: Vec<Layer>,
	output: Layer,
	levels: usize,
}

/*
	Graph builder
		new(genomes) initializes builder with given genome count.
		methods are:
			addInput(input vector) adds input vector 
			addFunctions(function vector) adds functions
			addHidden(size) adds a hidden layer of given size
			addOutput(size) adds an output layer of given size
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
	fn addInput(&mut self,input: Vec<f64>) -> &mut GraphBuilder {
		for x in input {
			self.inputs.push(Node::InputNode(x));
		}
		self
	}
	fn addHidden(&mut self,size: i32) -> &mut GraphBuilder {
		self.hidden_layers.push(Layer::new());
		let length: usize = self.hidden_layers.len()-1;
		for index in 0..size {
			self.hidden_layers[length].push(Node::GeneNode(GeneNode {
				function: 0, //BiFunction
				input_node_one: NodeIndex::InputIndex(0), //Node
				input_node_two: NodeIndex::InputIndex(0), //Node
			}
			));
		}
		self
	}
	fn addOutput(&mut self,size: i32) -> &mut GraphBuilder {
		self.output = Layer::new();
		for index in 0..size {
			self.output.push(Node::GeneNode(GeneNode {
				function: 0, //BiFunction
				input_node_one: NodeIndex::InputIndex(0), //Node
				input_node_two: NodeIndex::InputIndex(0), //Node
			}
			));
		}
		self
	}
	fn addFunctions(&mut self, funcs: Vec<BiFunction>) -> &mut GraphBuilder{
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
				*gene = Node::GeneNode( GeneNode {
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
					}else {
						1
					};
					let to_layer: isize = cmp::max(-1, (index as isize) - (back as isize));
					if to_layer < 0 {
						let to_layer = to_layer as usize;
						*gene = Node::GeneNode( GeneNode {
						input_node_one: NodeIndex::InputIndex(rand::thread_rng().gen_range(0, input_len)), 
						input_node_two: NodeIndex::InputIndex(rand::thread_rng().gen_range(0, input_len)),
						function: rand::thread_rng().gen_range(0, function_len),
						});
					}else {
						let to_layer = to_layer as usize;
						*gene = Node::GeneNode( GeneNode {
						input_node_one: NodeIndex::GeneIndex(to_layer,rand::thread_rng().gen_range(0, layer_lens[index-1])), 
						input_node_two: NodeIndex::GeneIndex(to_layer,rand::thread_rng().gen_range(0, layer_lens[index-1])), 		
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
					}else {
						1
					};
				let to_layer: isize = cmp::max(-1, (index as isize) - (back as isize));
				if to_layer < 0 {
					let to_layer = to_layer as usize;
					*gene = Node::GeneNode( GeneNode {
					input_node_one: NodeIndex::InputIndex(rand::thread_rng().gen_range(0, input_len)), 
					input_node_two: NodeIndex::InputIndex(rand::thread_rng().gen_range(0, input_len)),
					function: rand::thread_rng().gen_range(0, function_len),
					});
				}else {
					let to_layer = to_layer as usize;
					*gene = Node::GeneNode( GeneNode {
					input_node_one: NodeIndex::GeneIndex(to_layer,rand::thread_rng().gen_range(0, layer_lens[index-1])), 
					input_node_two: NodeIndex::GeneIndex(to_layer,rand::thread_rng().gen_range(0, layer_lens[index-1])), 		
					function: rand::thread_rng().gen_range(0, function_len),
					});
				}
			}
		}
		let graph = graph;
		graph
	}
}

#[test]
fn test_graph() {
    let input1 = Node::InputNode(0.0);
    let input1_i = NodeIndex::InputIndex(0);
    let input2 = Node::InputNode(1.0);
    let input2_i = NodeIndex::InputIndex(1);

    fn op1 (x: f64, y: f64) -> f64 { x + y };
    let op1_i = 0;
    fn op2 (x: f64, y: f64) -> f64 { x - y };
    let op2_i = 1;

    let gene1 = Node::GeneNode(GeneNode {
        function: op1_i,
        input_node_one: input2_i,
        input_node_two: input2_i,
    });

    let gene2 = Node::GeneNode(GeneNode {
        function: op2_i,
        input_node_one: input1_i,
        input_node_two: input2_i,
    });

    let genome = Genome {
        inner_layers: vec![vec![gene1, gene2]],
        output_layer: vec![input1, gene1, gene2],
    };

    let the_graph = Graph {
        inputs: vec![input1, input2],
        functions: vec![op1, op2],
        genomes: vec![genome],
    };

    let result1 = the_graph.genomes[0].evaluate(&the_graph.genomes[0].output_layer[0], &the_graph.inputs, &the_graph.functions);
    let result2 = the_graph.genomes[0].evaluate(&the_graph.genomes[0].output_layer[1], &the_graph.inputs, &the_graph.functions);
    let result3 = the_graph.genomes[0].evaluate(&the_graph.genomes[0].output_layer[2], &the_graph.inputs, &the_graph.functions);
    assert_eq!(result1, 0.0);
    assert_eq!(result2, 2.0);
    assert_eq!(result3, -1.0);
	
	
	let inp = vec![0.1,0.2,0.3,0.4];
	let fns = [op1,op2].to_vec();
	let new_graph = GraphBuilder::new(1)
	.addInput(inp)
	.addHidden(8)
	.addHidden(4)
	.addHidden(2)
	.addOutput(1)
	.addFunctions(fns)
	.levels(1)
	.build();
	
	let mut g = new_graph.genomes[0].mutate_nodes(10, &(new_graph.inputs), 1, new_graph.functions.len());
	
	let result1 = new_graph.genomes[0].evaluate(&new_graph.genomes[0].output_layer[0], &new_graph.inputs, &new_graph.functions);
	
}











