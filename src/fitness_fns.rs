extern crate csv; // CSV parsing
use std::fs::File;
use std::io::{BufReader, BufRead};

pub struct Dataset {
    input_examples: Vec<f64>,
    output_examples: Vec<f64>,
}


//TODO: Dataset is simple enough to not need CSV lib?
pub fn dataset_from_csv(filename: String) -> Dataset {
    let mut the_dataset = Dataset {
        input_examples: Vec::new(),
        output_examples: Vec::new(),
    };

    let mut reader = csv::Reader::from_file(filename).unwrap().has_headers(false);
    for record in reader.decode() {
        let (input_example, output_example): (f64, f64) = record.unwrap(); //Unwrapped replaces wrapped

        the_dataset.input_examples.push(input_example);
        the_dataset.output_examples.push(output_example);
        println!("({}, {})", input_example, output_example);
    }
    return the_dataset;
}

//TODO:...
pub fn dataset_from_csv_nodepend(filename: String) -> Dataset {
    let mut the_dataset = Dataset {
        input_examples: Vec::new(),
        output_examples: Vec::new(),
    };

    let the_file = match File::open(&filename) {
        Ok(f) => f,
        Err(e) => panic!("Could not find file: {} \n{}", filename, e),
    };
    let the_reader = BufReader::new(the_file);

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





#[test]
fn test_dataset() {
    println!(file!());
    {
        let the_dataset = dataset_from_csv_nodepend(String::from("src\\sin.csv"));
        assert_eq!(the_dataset.input_examples[0], 0.0);
        assert_eq!(the_dataset.output_examples[0], 0.0);

        assert_eq!(the_dataset.input_examples[1], 0.1);
        assert_eq!(the_dataset.output_examples[1], 0.0998334166468282);

        assert_eq!(*the_dataset.input_examples.last().unwrap(), 6.2);
        assert_eq!(*the_dataset.output_examples.last().unwrap(), -0.0830894028174964);
    }
    {
        let the_dataset = dataset_from_csv(String::from("src\\sin.csv"));
        assert_eq!(the_dataset.input_examples[0], 0.0);
        assert_eq!(the_dataset.output_examples[0], 0.0);

        assert_eq!(the_dataset.input_examples[1], 0.1);
        assert_eq!(the_dataset.output_examples[1], 0.0998334166468282);

        assert_eq!(*the_dataset.input_examples.last().unwrap(), 6.2);
        assert_eq!(*the_dataset.output_examples.last().unwrap(), -0.0830894028174964);
    }
}