extern crate csv; // CSV parsing

pub struct Dataset {
    input_examples: Vec<f64>,
    output_examples: Vec<f64>,
}



pub fn dataset_from_csv(filename: String) -> Dataset {
    let mut the_dataset = Dataset {
        input_examples: Vec::new(),
        output_examples: Vec::new(),
    }; //TODO: mut?

    let mut reader = csv::Reader::from_file(filename).unwrap(); //TODO: mut?
    for record in reader.decode() {
        let (input_example, output_example): (f64, f64) = record.unwrap(); //Unwrapped replaces wrapped

        the_dataset.input_examples.push(input_example);
        the_dataset.output_examples.push(output_example);
        println!("({}, {})", input_example, output_example);
    }
    return the_dataset;
}