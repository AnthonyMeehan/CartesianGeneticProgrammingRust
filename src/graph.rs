use rand;
use rand::Rng;
use rand::ThreadRng;
use std::cmp;

type BiFunction = fn(f64,f64) -> f64;
type FunctionIndex = usize;

#[derive(Clone, Copy)]
enum NodeIndex {
    InputIndex(usize),
    GeneIndex(usize, usize),
}

///Gene Nodes Contain a function index, two input indices, and a possible previously-evaluated result
#[derive(Clone, Copy)]
struct GeneNode {
    function: FunctionIndex, //BiFunction
    input_node_one: NodeIndex, //Node
    input_node_two: NodeIndex, //Node
    //TODO: output: Option<f64>, separate precomputation matrix?
}

///Nodes in the graph can be inputs, or gene nodes. "Output" Nodes just reference other nodes on the graph
#[derive(Clone, Copy)]
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



    /*
    fn mutate_nodes(&self) {

    }*/

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
		
		// TODO: randomize
		
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
}