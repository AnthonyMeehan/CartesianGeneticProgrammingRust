extern crate csv; // for CSV parsing
use std::fs::File;
use std::io::{BufReader, BufRead};

///A series of input examples matched-up with output examples, for testing CGP genomes
pub struct Dataset {
    pub input_examples: Vec<f64>,
    pub output_examples: Vec<f64>,
}

///Extracts a dataset from a CSV, using a CSV library
pub fn dataset_from_csv(filename: String) -> Dataset {
    //Try and load the file
    let mut reader = csv::Reader::from_file(filename).unwrap().has_headers(false);

    let mut the_dataset = Dataset {
        input_examples: Vec::new(),
        output_examples: Vec::new(),
    };

    //Extract CSV contents into dataset
    for record in reader.decode() {
        let (input_example, output_example): (f64, f64) = record.unwrap(); //Unwrapped replaces wrapped

        the_dataset.input_examples.push(input_example);
        the_dataset.output_examples.push(output_example);
        println!("({}, {})", input_example, output_example);
    }
    return the_dataset;
}

///Extracts a dataset from a CSV, without using any dependencies
pub fn dataset_from_csv_no_dependencies(filename: String) -> Dataset {
    //Try and load the file
    let the_file = match File::open(&filename) {
        Ok(f) => f,
        Err(e) => panic!("Could not find file: {} \n{}", filename, e),
    };
    let the_reader = BufReader::new(the_file);

    let mut the_dataset = Dataset {
        input_examples: Vec::new(),
        output_examples: Vec::new(),
    };

    //Extract CSV contents into dataset
    for line in the_reader.lines() {
        let mut examples = match line {
            Ok(ref l) => l.split(","),
            Err(e) => panic!(e)
        };
        let input_example = examples.next().unwrap().trim().parse().expect("float needed from first CSV column");
        let output_example = examples.next().unwrap().trim().parse().expect("float needed from second CSV column");
        the_dataset.input_examples.push(input_example);
        the_dataset.output_examples.push(output_example);
        println!("({}, {})", input_example, output_example);
    }
    return the_dataset;
}

///Check that the data extractors work correctly on a test dataset
#[test]
fn test_dataset() {
    println!(file!());
    let datasets = vec![
        dataset_from_csv_no_dependencies(String::from("src\\sin.csv")),
        dataset_from_csv(String::from("src\\sin.csv")),
    ];

    for dataset in datasets {
        assert_eq!(dataset.input_examples[0], 0.0);
        assert_eq!(dataset.output_examples[0], 0.0);

        assert_eq!(dataset.input_examples[1], 0.1);
        assert_eq!(dataset.output_examples[1], 0.0998334166468282);

        assert_eq!(*dataset.input_examples.last().unwrap(), 6.2);
        assert_eq!(*dataset.output_examples.last().unwrap(), -0.0830894028174964);
    }
}