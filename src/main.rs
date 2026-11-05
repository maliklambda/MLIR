use crate::neuron::{ActivationFnKind, Neuron, TransferFnKind, WeightedInput};

pub mod neuron;

fn main() {
    let inputs: Vec<WeightedInput> =
        vec![WeightedInput::new(5.0, 2.0), WeightedInput::new(3.5, 1.2)];
    let bias = -1.0;
    let n1 = Neuron::new(
        &inputs,
        &TransferFnKind::WeightedSum,
        &ActivationFnKind::StepFn,
        bias,
    );
    let n2 = Neuron::new(
        &inputs,
        &TransferFnKind::WeightedSum,
        &ActivationFnKind::Sigmoid,
        bias,
    );
    println!("Neuron 1: {:?}\nApplication: {:?}", n1, n1.apply());
    println!("Neuron 2: {:?}\nApplication: {:?}", n2, n2.apply());
}
