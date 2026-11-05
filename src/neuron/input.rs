/// Weighted input to a neuron.
#[derive(Debug, Clone)]
pub struct WeightedInput {
    pub input: f32,
    pub weight: f32,
}

impl WeightedInput {
    pub fn new(input: f32, weight: f32) -> Self {
        Self { input, weight }
    }
}
