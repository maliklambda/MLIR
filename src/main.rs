use crate::{
    layer::Layer,
    network::NeuralNetwork,
    neuron::{ActivationFnKind, Neuron, TransferFnKind},
};

pub mod layer;
pub mod network;
pub mod neuron;

fn main() {
    let inputs: Vec<f32> = vec![1.2, 3.4];
    let bias = -1.0;
    let n1 = Neuron::new(
        &TransferFnKind::WeightedSum,
        &ActivationFnKind::StepFn,
        vec![0.5, -1.3],
        bias,
    );
    let n2 = Neuron::new(
        &TransferFnKind::WeightedSum,
        &ActivationFnKind::Sigmoid,
        vec![1.2, 2.1],
        bias,
    );
    println!("Neuron 1: {:?}\nApplication: {:?}", n1, n1.apply(&inputs));
    println!("Neuron 2: {:?}\nApplication: {:?}", n2, n2.apply(&inputs));
    let l = Layer::new(vec![n1, n2]);
    println!("Layer: {:?}\nApplication: {:?}", l, l.apply(&inputs));
    let nn = NeuralNetwork::new(vec![l]);
    println!(
        "Neural Network: {:?}\nApplication: {:?}",
        nn,
        nn.apply(inputs)
    );
}
