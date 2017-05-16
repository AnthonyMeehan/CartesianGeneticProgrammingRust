use rand;
use rand::Rng;
use rand::ThreadRng;
use std::cmp;

type BiFunction = fn(f64,f64) -> f64;
type FunctionIndex = usize;

enum NodeIndex {
    InputIndex(usize),
    GeneIndex(usize, usize),
}

///Gene Nodes Contain a function index, two input indices, and a possible previously-evaluated result
struct GeneNode {
	function: FunctionIndex, //BiFunction
	input_node_one: NodeIndex, //Node
	input_node_two: NodeIndex, //Node
    output: Option<f64>,
}

///Nodes in the graph can be inputs, or gene nodes. "Output" Nodes just reference other nodes on the graph
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
            &Node::GeneNode(ref gene) => {
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
            }
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


}