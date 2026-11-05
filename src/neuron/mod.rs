pub use {
    activation::{ActivationFn, ActivationFnKind, Activationalbe},
    input::WeightedInput,
    transfer::{TransferFn, TransferFnKind, Transferable},
};

mod activation;
mod input;
mod transfer;

#[derive(Debug, Clone)]
pub struct Neuron {
    /// Channels the inputs into a single value.
    /// Produces the transfer-value.
    /// Commonly used: Weighted Sum
    pub transfer_fn: TransferFn,

    /// Weights stored by this neuron
    pub weights: Vec<f32>,

    /// Transfer-value is checked based on activation.
    /// Bias shifts result independently of the transfer-value.
    /// Commonly used: Stepfn, Sigmoid, etc.
    pub activation_fn: ActivationFn,

    /// The neuron's bias.
    /// Used in activation_fn.
    pub bias: f32,
}

impl Neuron {
    pub fn new(
        transfer_fn: &dyn Transferable,
        activation_fn: &dyn Activationalbe,
        weights: Vec<f32>,
        bias: f32,
    ) -> Self {
        Self {
            transfer_fn: transfer_fn.to_transfer_fn(),
            activation_fn: activation_fn.to_activation_fn(),
            weights,
            bias,
        }
    }

    /// Applies an input to the neuron and returns the resulting output.
    pub fn apply(&self, inputs: &[f32]) -> f32 {
        assert_eq!(
            inputs.len(),
            self.weights.len(),
            "Length missmatch for inputs & weights in neuron-application."
        );
        let weighted_inputs: Vec<WeightedInput> = self
            .weights
            .iter()
            .zip(inputs)
            .map(|(w, i)| WeightedInput::new(*i, *w))
            .collect();
        let transfer_value = (self.transfer_fn)(&weighted_inputs);
        (self.activation_fn)(transfer_value, self.bias)
    }
}


#[test]
#[should_panic(expected = "Length missmatch for inputs & weights in neuron-application.")]
fn input_application_with_invalid_length_fails() {
    let weights = vec![2.0, 1.0];
    let inputs_invalid_len = vec![1.2, 3.4, 5.4];
    let n = Neuron::new(
        &TransferFnKind::WeightedSum,
        &ActivationFnKind::StepFn,
        weights,
        -1.0,
    );
    n.apply(&inputs_invalid_len);
}
