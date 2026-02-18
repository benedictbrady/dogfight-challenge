use glam::Vec2;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FighterState {
    pub position: Vec2,
    pub yaw: f32,
    pub speed: f32,
    pub hp: u8,
    pub gun_cooldown_ticks: u32,
    pub alive: bool,
    pub stall_ticks: u32,
}

impl FighterState {
    pub fn forward(&self) -> Vec2 {
        Vec2::new(self.yaw.cos(), self.yaw.sin())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bullet {
    pub position: Vec2,
    pub velocity: Vec2,
    pub owner: usize,
    pub ticks_remaining: u32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Action {
    pub yaw_input: f32,
    pub throttle: f32,
    pub shoot: bool,
}

impl Action {
    pub fn none() -> Self {
        Self {
            yaw_input: 0.0,
            throttle: 0.0,
            shoot: false,
        }
    }

    pub fn from_raw(raw: [f32; 3]) -> Self {
        Self {
            yaw_input: raw[0].clamp(-1.0, 1.0),
            throttle: raw[1].clamp(0.0, 1.0),
            shoot: raw[2] > 0.0,
        }
    }

    pub fn to_raw(&self) -> [f32; 3] {
        [
            self.yaw_input,
            self.throttle,
            if self.shoot { 1.0 } else { -1.0 },
        ]
    }
}

impl Default for Action {
    fn default() -> Self {
        Self::none()
    }
}

#[derive(Debug, Clone)]
pub struct Observation {
    pub data: [f32; crate::OBS_SIZE],
}

impl serde::Serialize for Observation {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.data.as_slice().serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for Observation {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let v: Vec<f32> = Vec::deserialize(deserializer)?;
        if v.len() != crate::OBS_SIZE {
            return Err(serde::de::Error::custom(format!(
                "expected {} floats, got {}",
                crate::OBS_SIZE,
                v.len()
            )));
        }
        let mut data = [0.0f32; crate::OBS_SIZE];
        data.copy_from_slice(&v);
        Ok(Observation { data })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayFrame {
    pub tick: u32,
    pub fighters: [FighterSnapshot; 2],
    pub bullets: Vec<BulletSnapshot>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FighterSnapshot {
    pub x: f32,
    pub y: f32,
    pub yaw: f32,
    pub speed: f32,
    pub hp: u8,
    pub alive: bool,
    pub stalled: bool,
}

impl From<&FighterState> for FighterSnapshot {
    fn from(s: &FighterState) -> Self {
        Self {
            x: s.position.x,
            y: s.position.y,
            yaw: s.yaw,
            speed: s.speed,
            hp: s.hp,
            alive: s.alive,
            stalled: s.stall_ticks > 0,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BulletSnapshot {
    pub x: f32,
    pub y: f32,
    pub vx: f32,
    pub vy: f32,
    pub owner: usize,
}

impl From<&Bullet> for BulletSnapshot {
    fn from(b: &Bullet) -> Self {
        Self {
            x: b.position.x,
            y: b.position.y,
            vx: b.velocity.x,
            vy: b.velocity.y,
            owner: b.owner,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Replay {
    pub config: MatchConfig,
    pub frames: Vec<ReplayFrame>,
    pub result: MatchResult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchConfig {
    pub seed: u64,
    pub p0_name: String,
    pub p1_name: String,
    pub p0_control_period: u32,
    pub p1_control_period: u32,
    pub max_ticks: u32,
    pub randomize_spawns: bool,
}

impl Default for MatchConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            p0_name: "p0".into(),
            p1_name: "p1".into(),
            p0_control_period: 1,
            p1_control_period: 1,
            max_ticks: crate::MAX_TICKS,
            randomize_spawns: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchResult {
    pub outcome: MatchOutcome,
    pub reason: MatchEndReason,
    pub final_tick: u32,
    pub stats: MatchStats,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MatchOutcome {
    Player0Win,
    Player1Win,
    Draw,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MatchEndReason {
    Elimination,
    Timeout,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MatchStats {
    pub p0_hp: u8,
    pub p1_hp: u8,
    pub p0_hits: u32,
    pub p1_hits: u32,
    pub p0_shots: u32,
    pub p1_shots: u32,
}

impl MatchOutcome {
    pub fn points(&self, player: usize) -> u32 {
        match (self, player) {
            (MatchOutcome::Player0Win, 0) | (MatchOutcome::Player1Win, 1) => {
                crate::WIN_ELIMINATION_POINTS
            }
            (MatchOutcome::Draw, _) => crate::DRAW_POINTS,
            _ => 0,
        }
    }

    pub fn points_with_reason(&self, reason: MatchEndReason, player: usize) -> u32 {
        match (self, reason, player) {
            (MatchOutcome::Player0Win, MatchEndReason::Elimination, 0)
            | (MatchOutcome::Player1Win, MatchEndReason::Elimination, 1) => {
                crate::WIN_ELIMINATION_POINTS
            }
            (MatchOutcome::Player0Win, MatchEndReason::Timeout, 0)
            | (MatchOutcome::Player1Win, MatchEndReason::Timeout, 1) => crate::WIN_HP_POINTS,
            (MatchOutcome::Draw, _, _) => crate::DRAW_POINTS,
            _ => 0,
        }
    }
}
