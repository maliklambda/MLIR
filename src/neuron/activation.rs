/// Adds a bias to the transfer value and checks if the resulting value fires the neuron.
/// Can be of type step-function or sigmoid.
pub type ActivationFn = fn(f32 /*Transfer-value*/, f32 /*Bias*/) -> f32;

/// Allows users to specify their own activation functions.
pub trait Activationalbe {
    fn to_activation_fn(&self) -> ActivationFn;
}

/// Pre-defined activation functions.
#[derive(Debug)]
pub enum ActivationFnKind {
    /// Step Function converts a numeric input into binary output.
    /// Returns 1 for input >= 0.
    /// Returns 0 for input < 0.
    StepFn,

    /// Sigmoid Function creates an S-shape with values ranging between 0 and 1.
    Sigmoid,

    /// Tangens hyperbolicus.
    /// A relationship derived from sinus hyperbolicus (sinh) and cosinus hyperbolicus (cosh).
    /// tanh(x) = sinh(x) / cosh(x)
    /// Returns a value between -1 and 1.
    Tanh,

    /// Rectified Linear Unit
    /// Returns 0 for inputs <= 0.
    /// Returns x for inputs x where x > 0.
    ReLU,
}

/// Activation functions can be turned into an Activationalbe.
impl Activationalbe for ActivationFnKind {
    /// Returns the concrete function associated with the Activationalbe
    fn to_activation_fn(&self) -> ActivationFn {
        match self {
            Self::StepFn => |tv, bias| if (tv + bias) >= 0.0 { 1.0 } else { 0.0 },
            Self::Sigmoid => |tv, bias| {
                let b = tv + bias;
                1.0 / (1.0 + (-b).exp())
            },
            Self::Tanh => |tv, bias| (tv + bias).tanh(), // <- Thanks Rust std-lib
            Self::ReLU => |tv, bias| f32::max(0.0, tv + bias),
        }
    }
}
