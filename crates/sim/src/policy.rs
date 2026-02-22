use dogfight_shared::{Action, Observation};

pub trait Policy: Send {
    fn name(&self) -> &str;
    fn act(&mut self, obs: &Observation) -> Action;
}
