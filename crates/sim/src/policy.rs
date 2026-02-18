use dogfight_shared::{Action, Observation};

pub trait Policy: Send {
    fn name(&self) -> &str;
    fn act(&mut self, obs: &Observation) -> Action;
}

/// Policy that does nothing - useful for testing.
pub struct DoNothingPolicy;

impl Policy for DoNothingPolicy {
    fn name(&self) -> &str {
        "do_nothing"
    }

    fn act(&mut self, _obs: &Observation) -> Action {
        Action::none()
    }
}
