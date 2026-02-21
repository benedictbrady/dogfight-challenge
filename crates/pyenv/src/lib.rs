use dogfight_shared::*;
use dogfight_sim::opponents::{AcePolicy, BrawlerPolicy, ChaserPolicy, DogfighterPolicy};
use dogfight_sim::physics::SimState;
use dogfight_sim::{DoNothingPolicy, Policy};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rand::SeedableRng;
use rand::Rng;
use rand_pcg::Pcg64;
use rayon::prelude::*;

fn resolve_policy(name: &str) -> PyResult<Box<dyn Policy>> {
    match name {
        "do_nothing" => Ok(Box::new(DoNothingPolicy)),
        "chaser" => Ok(Box::new(ChaserPolicy::new())),
        "dogfighter" => Ok(Box::new(DogfighterPolicy::new())),
        "ace" => Ok(Box::new(AcePolicy::new())),
        "brawler" => Ok(Box::new(BrawlerPolicy::new())),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown policy: {other}"
        ))),
    }
}

/// Create a policy with a specific SimConfig (for domain randomization).
fn make_policy_with_config(name: &str, config: SimConfig) -> Box<dyn Policy> {
    match name {
        "do_nothing" => Box::new(DoNothingPolicy),
        "chaser" => Box::new(ChaserPolicy::with_config(config)),
        "dogfighter" => Box::new(DogfighterPolicy::with_config(config)),
        "ace" => Box::new(AcePolicy::with_config(config)),
        "brawler" => Box::new(BrawlerPolicy::with_config(config)),
        _ => Box::new(DoNothingPolicy),
    }
}

// ---------------------------------------------------------------------------
// SimConfigRanges — domain randomization parameter ranges
// ---------------------------------------------------------------------------

/// Configurable ranges for domain randomization. Each field is `Option<(min, max)>`.
/// `None` = use default value, `Some((min, max))` = sample uniformly per episode.
#[derive(Clone, Default)]
struct SimConfigRanges {
    gravity: Option<(f32, f32)>,
    drag_coeff: Option<(f32, f32)>,
    turn_bleed_coeff: Option<(f32, f32)>,
    max_speed: Option<(f32, f32)>,
    min_speed: Option<(f32, f32)>,
    max_thrust: Option<(f32, f32)>,
    bullet_speed: Option<(f32, f32)>,
    gun_cooldown_ticks: Option<(u32, u32)>,
    bullet_lifetime_ticks: Option<(u32, u32)>,
    max_hp: Option<(u8, u8)>,
    max_turn_rate: Option<(f32, f32)>,
    min_turn_rate: Option<(f32, f32)>,
    rear_aspect_cone: Option<(f32, f32)>,
}

impl SimConfigRanges {
    /// Returns true if any field has a range set.
    fn is_active(&self) -> bool {
        self.gravity.is_some()
            || self.drag_coeff.is_some()
            || self.turn_bleed_coeff.is_some()
            || self.max_speed.is_some()
            || self.min_speed.is_some()
            || self.max_thrust.is_some()
            || self.bullet_speed.is_some()
            || self.gun_cooldown_ticks.is_some()
            || self.bullet_lifetime_ticks.is_some()
            || self.max_hp.is_some()
            || self.max_turn_rate.is_some()
            || self.min_turn_rate.is_some()
            || self.rear_aspect_cone.is_some()
    }

    /// Sample a SimConfig from the ranges. Unset fields use defaults.
    fn sample(&self, rng: &mut Pcg64) -> SimConfig {
        let d = SimConfig::default();
        SimConfig {
            gravity: self.gravity.map_or(d.gravity, |(lo, hi)| rng.gen_range(lo..=hi)),
            drag_coeff: self.drag_coeff.map_or(d.drag_coeff, |(lo, hi)| rng.gen_range(lo..=hi)),
            turn_bleed_coeff: self.turn_bleed_coeff.map_or(d.turn_bleed_coeff, |(lo, hi)| rng.gen_range(lo..=hi)),
            max_speed: self.max_speed.map_or(d.max_speed, |(lo, hi)| rng.gen_range(lo..=hi)),
            min_speed: self.min_speed.map_or(d.min_speed, |(lo, hi)| rng.gen_range(lo..=hi)),
            max_thrust: self.max_thrust.map_or(d.max_thrust, |(lo, hi)| rng.gen_range(lo..=hi)),
            bullet_speed: self.bullet_speed.map_or(d.bullet_speed, |(lo, hi)| rng.gen_range(lo..=hi)),
            gun_cooldown_ticks: self.gun_cooldown_ticks.map_or(d.gun_cooldown_ticks, |(lo, hi)| rng.gen_range(lo..=hi)),
            bullet_lifetime_ticks: self.bullet_lifetime_ticks.map_or(d.bullet_lifetime_ticks, |(lo, hi)| rng.gen_range(lo..=hi)),
            max_hp: self.max_hp.map_or(d.max_hp, |(lo, hi)| rng.gen_range(lo..=hi)),
            max_turn_rate: self.max_turn_rate.map_or(d.max_turn_rate, |(lo, hi)| rng.gen_range(lo..=hi)),
            min_turn_rate: self.min_turn_rate.map_or(d.min_turn_rate, |(lo, hi)| rng.gen_range(lo..=hi)),
            rear_aspect_cone: self.rear_aspect_cone.map_or(d.rear_aspect_cone, |(lo, hi)| rng.gen_range(lo..=hi)),
        }
    }

    /// Parse from a Python dict: {"gravity": (60.0, 100.0), "bullet_speed": (350.0, 450.0), ...}
    fn from_py_dict(dict: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mut ranges = Self::default();
        for (key, value) in dict.iter() {
            let key: String = key.extract()?;
            match key.as_str() {
                "gravity" => ranges.gravity = Some(value.extract()?),
                "drag_coeff" => ranges.drag_coeff = Some(value.extract()?),
                "turn_bleed_coeff" => ranges.turn_bleed_coeff = Some(value.extract()?),
                "max_speed" => ranges.max_speed = Some(value.extract()?),
                "min_speed" => ranges.min_speed = Some(value.extract()?),
                "max_thrust" => ranges.max_thrust = Some(value.extract()?),
                "bullet_speed" => ranges.bullet_speed = Some(value.extract()?),
                "gun_cooldown_ticks" => ranges.gun_cooldown_ticks = Some(value.extract()?),
                "bullet_lifetime_ticks" => ranges.bullet_lifetime_ticks = Some(value.extract()?),
                "max_hp" => ranges.max_hp = Some(value.extract()?),
                "max_turn_rate" => ranges.max_turn_rate = Some(value.extract()?),
                "min_turn_rate" => ranges.min_turn_rate = Some(value.extract()?),
                "rear_aspect_cone" => ranges.rear_aspect_cone = Some(value.extract()?),
                other => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "unknown SimConfig parameter: {other}"
                    )));
                }
            }
        }
        Ok(ranges)
    }
}

// ---------------------------------------------------------------------------
// Single env (unchanged, for backwards compat)
// ---------------------------------------------------------------------------

/// Gym-like environment wrapping the dogfight sim.
///
/// Usage:
///     env = DogfightEnv("chaser", 42, True)
///     obs = env.reset()
///     obs, reward, done, info = env.step([0.0, 1.0, -1.0])
#[pyclass(unsendable)]
struct DogfightEnv {
    opponent_name: String,
    seed: u64,
    randomize_spawns: bool,

    // Reward weights (configurable)
    w_damage_dealt: f32,
    w_damage_taken: f32,
    w_win: f32,
    w_lose: f32,
    w_approach: f32,
    w_alive: f32,
    w_proximity: f32,
    w_facing: f32,

    // Sim state
    state: SimState,
    opponent: Box<dyn Policy>,
    tick: u32,

    // Tracking for delta-based rewards
    prev_my_hp: u8,
    prev_opp_hp: u8,
    prev_distance: f32,
}

#[pymethods]
impl DogfightEnv {
    #[new]
    #[pyo3(signature = (opponent, seed=0, randomize_spawns=false))]
    fn new(opponent: &str, seed: u64, randomize_spawns: bool) -> PyResult<Self> {
        let opp = resolve_policy(opponent)?;
        let state = SimState::new_with_seed(seed, randomize_spawns);
        let dist = (state.fighters[0].position - state.fighters[1].position).length();
        Ok(Self {
            opponent_name: opponent.to_string(),
            seed,
            randomize_spawns,

            w_damage_dealt: 3.0,
            w_damage_taken: -1.0,
            w_win: 5.0,
            w_lose: -5.0,
            w_approach: 0.0001,
            w_alive: 0.0,
            w_proximity: 0.001,
            w_facing: 0.0005,

            state,
            opponent: opp,
            tick: 0,

            prev_my_hp: MAX_HP,
            prev_opp_hp: MAX_HP,
            prev_distance: dist,
        })
    }

    /// Set reward weights. Keyword-only arguments with defaults.
    #[pyo3(signature = (
        damage_dealt=None,
        damage_taken=None,
        win=None,
        lose=None,
        approach=None,
        alive=None,
        proximity=None,
        facing=None,
    ))]
    fn set_rewards(
        &mut self,
        damage_dealt: Option<f32>,
        damage_taken: Option<f32>,
        win: Option<f32>,
        lose: Option<f32>,
        approach: Option<f32>,
        alive: Option<f32>,
        proximity: Option<f32>,
        facing: Option<f32>,
    ) {
        if let Some(v) = damage_dealt {
            self.w_damage_dealt = v;
        }
        if let Some(v) = damage_taken {
            self.w_damage_taken = v;
        }
        if let Some(v) = win {
            self.w_win = v;
        }
        if let Some(v) = lose {
            self.w_lose = v;
        }
        if let Some(v) = approach {
            self.w_approach = v;
        }
        if let Some(v) = alive {
            self.w_alive = v;
        }
        if let Some(v) = proximity {
            self.w_proximity = v;
        }
        if let Some(v) = facing {
            self.w_facing = v;
        }
    }

    /// Reset the environment. Returns observation (list of 46 floats).
    #[pyo3(signature = (seed=None))]
    fn reset(&mut self, seed: Option<u64>) -> Vec<f32> {
        let s = seed.unwrap_or(self.seed);
        self.seed = s;
        self.state = SimState::new_with_seed(s, self.randomize_spawns);
        self.opponent = resolve_policy(&self.opponent_name).unwrap();
        self.tick = 0;
        self.prev_my_hp = MAX_HP;
        self.prev_opp_hp = MAX_HP;
        self.prev_distance =
            (self.state.fighters[0].position - self.state.fighters[1].position).length();

        let obs = self.state.observe(0);
        obs.data.to_vec()
    }

    /// Step the environment with an action [yaw_input, throttle, shoot].
    /// Returns (obs, reward, done, info_dict).
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        action: Vec<f32>,
    ) -> PyResult<(Vec<f32>, f32, bool, Bound<'py, PyDict>)> {
        if action.len() != ACTION_SIZE {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "action must have {} elements, got {}",
                ACTION_SIZE,
                action.len()
            )));
        }

        let my_action = Action::from_raw([action[0], action[1], action[2]]);

        // Get opponent observation and action
        let opp_obs = self.state.observe(1);
        let opp_action = self.opponent.act(&opp_obs);

        // Step physics
        self.state.step(&[my_action, opp_action]);
        self.tick += 1;

        let done = self.state.is_terminal();

        // Compute reward
        let my_hp = self.state.fighters[0].hp;
        let opp_hp = self.state.fighters[1].hp;
        let distance =
            (self.state.fighters[0].position - self.state.fighters[1].position).length();

        let mut reward = 0.0f32;

        // Damage dealt (opponent lost HP)
        let opp_hp_delta = self.prev_opp_hp as f32 - opp_hp as f32;
        reward += opp_hp_delta * self.w_damage_dealt;

        // Damage taken (we lost HP)
        let my_hp_delta = self.prev_my_hp as f32 - my_hp as f32;
        reward += my_hp_delta * self.w_damage_taken;

        // Approach reward (getting closer to opponent)
        let dist_delta = self.prev_distance - distance;
        reward += dist_delta * self.w_approach;

        // Alive bonus
        if self.state.fighters[0].alive {
            reward += self.w_alive;
        }

        // Proximity bonus (within engagement range, ~250m)
        if self.state.fighters[0].alive && distance < 250.0 {
            reward += self.w_proximity;
        }

        // Facing reward (pointing at opponent AND within bullet range)
        if self.state.fighters[0].alive && distance < 300.0 && distance > 1.0 {
            let me = &self.state.fighters[0];
            let rel = self.state.fighters[1].position - me.position;
            let hx = me.yaw.cos();
            let hy = me.yaw.sin();
            let inv_dist = 1.0 / distance;
            let dot = hx * rel.x * inv_dist + hy * rel.y * inv_dist;
            if dot > 0.866 {
                // ~30° cone: cos(30°) ≈ 0.866
                reward += self.w_facing;
            }
        }

        // Terminal rewards
        if done {
            let (outcome, _reason) = self.state.outcome();
            match outcome {
                MatchOutcome::Player0Win => reward += self.w_win,
                MatchOutcome::Player1Win => reward += self.w_lose,
                MatchOutcome::Draw => {} // no bonus
            }
        }

        // Update tracking
        self.prev_my_hp = my_hp;
        self.prev_opp_hp = opp_hp;
        self.prev_distance = distance;

        // Observation
        let obs = self.state.observe(0);

        // Info dict
        let info = PyDict::new_bound(py);
        let (outcome, reason) = self.state.outcome();
        info.set_item("tick", self.tick)?;
        info.set_item("my_hp", my_hp)?;
        info.set_item("opp_hp", opp_hp)?;
        info.set_item("my_alive", self.state.fighters[0].alive)?;
        info.set_item("opp_alive", self.state.fighters[1].alive)?;
        info.set_item("outcome", format!("{:?}", outcome))?;
        info.set_item("reason", format!("{:?}", reason))?;

        Ok((obs.data.to_vec(), reward, done, info))
    }

    /// Return observation size.
    #[getter]
    fn obs_size(&self) -> usize {
        OBS_SIZE
    }

    /// Return action size.
    #[getter]
    fn action_size(&self) -> usize {
        ACTION_SIZE
    }

    /// Return the current tick.
    #[getter]
    fn current_tick(&self) -> u32 {
        self.tick
    }

    /// Return max ticks.
    #[getter]
    fn max_ticks(&self) -> u32 {
        MAX_TICKS
    }
}

// ---------------------------------------------------------------------------
// BatchEnv — Rayon-parallelized vectorized environment
// ---------------------------------------------------------------------------

/// Reward weights shared across all envs in the batch.
#[derive(Clone)]
struct RewardWeights {
    damage_dealt: f32,
    damage_taken: f32,
    win: f32,
    lose: f32,
    approach: f32,
    alive: f32,
    proximity: f32,
    facing: f32,
}

impl Default for RewardWeights {
    fn default() -> Self {
        Self {
            damage_dealt: 1.0,
            damage_taken: -0.5,
            win: 3.0,
            lose: -3.0,
            approach: 0.0,
            alive: 0.0,
            proximity: 0.0,
            facing: 0.0,
        }
    }
}

/// Per-env instance data (all Send-safe).
struct EnvInstance {
    state: SimState,
    opponent: Box<dyn Policy>,
    opponent_name: String,
    config: SimConfig,
    tick: u32,
    prev_my_hp: u8,
    prev_opp_hp: u8,
    prev_distance: f32,
}

/// Result of stepping a single env (plain data, no Python objects).
struct StepResult {
    obs: [f32; OBS_SIZE],
    reward: f32,
    done: bool,
    outcome: String,
    my_hp: u8,
    opp_hp: u8,
}

impl EnvInstance {
    fn new(opponent_name: &str, seed: u64, randomize: bool, config: SimConfig) -> Self {
        let state = SimState::new_with_seed_and_config(seed, randomize, config);
        let dist = (state.fighters[0].position - state.fighters[1].position).length();
        Self {
            state,
            opponent: make_policy_with_config(opponent_name, config),
            opponent_name: opponent_name.to_string(),
            config,
            tick: 0,
            prev_my_hp: config.max_hp,
            prev_opp_hp: config.max_hp,
            prev_distance: dist,
        }
    }

    fn reset(&mut self, opponent_name: &str, seed: u64, randomize: bool, config: SimConfig) -> [f32; OBS_SIZE] {
        self.config = config;
        self.state = SimState::new_with_seed_and_config(seed, randomize, config);
        self.opponent = make_policy_with_config(opponent_name, config);
        self.opponent_name = opponent_name.to_string();
        self.tick = 0;
        self.prev_my_hp = config.max_hp;
        self.prev_opp_hp = config.max_hp;
        self.prev_distance =
            (self.state.fighters[0].position - self.state.fighters[1].position).length();

        self.state.observe(0).data
    }

    /// Run a single physics tick for computing reward, without calling observe(0).
    /// The opponent observes and acts each tick. Returns reward delta for this tick.
    fn step_tick(
        &mut self,
        my_action: &Action,
        weights: &RewardWeights,
    ) -> (f32, bool, String) {
        // Opponent acts (observes each tick at 120Hz)
        let opp_obs = self.state.observe(1);
        let opp_action = self.opponent.act(&opp_obs);

        // Step physics
        self.state.step(&[*my_action, opp_action]);
        self.tick += 1;

        let done = self.state.is_terminal();

        // Compute reward
        let my_hp = self.state.fighters[0].hp;
        let opp_hp = self.state.fighters[1].hp;
        let distance =
            (self.state.fighters[0].position - self.state.fighters[1].position).length();

        let mut reward = 0.0f32;

        let opp_hp_delta = self.prev_opp_hp as f32 - opp_hp as f32;
        reward += opp_hp_delta * weights.damage_dealt;

        let my_hp_delta = self.prev_my_hp as f32 - my_hp as f32;
        reward += my_hp_delta * weights.damage_taken;

        let dist_delta = self.prev_distance - distance;
        reward += dist_delta * weights.approach;

        if self.state.fighters[0].alive {
            reward += weights.alive;
        }

        if self.state.fighters[0].alive && distance < 250.0 {
            reward += weights.proximity;
        }

        if self.state.fighters[0].alive && distance < 300.0 && distance > 1.0 {
            let me = &self.state.fighters[0];
            let rel = self.state.fighters[1].position - me.position;
            let hx = me.yaw.cos();
            let hy = me.yaw.sin();
            let inv_dist = 1.0 / distance;
            let dot = hx * rel.x * inv_dist + hy * rel.y * inv_dist;
            if dot > 0.866 {
                reward += weights.facing;
            }
        }

        let outcome_str = if done {
            let (outcome, _reason) = self.state.outcome();
            match outcome {
                MatchOutcome::Player0Win => {
                    reward += weights.win;
                    "Player0Win"
                }
                MatchOutcome::Player1Win => {
                    reward += weights.lose;
                    "Player1Win"
                }
                MatchOutcome::Draw => "Draw",
            }
        } else {
            ""
        };

        self.prev_my_hp = my_hp;
        self.prev_opp_hp = opp_hp;
        self.prev_distance = distance;

        (reward, done, outcome_str.to_string())
    }

    /// Step with action repeat: run `repeat` physics ticks with the same action,
    /// accumulating reward. RL agent observes ONCE after the loop (12Hz frame history).
    /// Opponent observes and acts each tick (120Hz).
    fn step_repeat(
        &mut self,
        action: [f32; ACTION_SIZE],
        weights: &RewardWeights,
        repeat: u32,
    ) -> StepResult {
        let my_action = Action::from_raw(action);
        let mut total_reward = 0.0f32;
        let mut done = false;
        let mut outcome = String::new();

        for _ in 0..repeat {
            let (reward, tick_done, tick_outcome) = self.step_tick(&my_action, weights);
            total_reward += reward;
            if tick_done {
                done = true;
                outcome = tick_outcome;
                break;
            }
        }

        // RL agent observes ONCE at the decision point (correct 12Hz frame history)
        let obs = self.state.observe(0).data;

        StepResult {
            obs,
            reward: total_reward,
            done,
            outcome,
            my_hp: self.state.fighters[0].hp,
            opp_hp: self.state.fighters[1].hp,
        }
    }
}

/// Vectorized environment that steps all N envs in parallel using Rayon.
///
/// Usage:
///     batch = BatchEnv(64, ["chaser", "dogfighter"], True)
///     obs = batch.reset()                    # flat list: 64 * 46 floats
///     obs, rewards, dones, infos = batch.step(actions)  # actions: 64 * 3 floats
///
/// All env stepping happens in Rust threads — one Python↔Rust call per step.
#[pyclass(unsendable)]
struct BatchEnv {
    envs: Vec<EnvInstance>,
    n_envs: usize,
    opponent_pool: Vec<String>,
    randomize: bool,
    action_repeat: u32,
    weights: RewardWeights,
    rng: Pcg64,
    config_ranges: SimConfigRanges,
}

#[pymethods]
impl BatchEnv {
    #[new]
    #[pyo3(signature = (n_envs, opponent_pool, randomize_spawns=true, seed=0, action_repeat=1))]
    fn new(n_envs: usize, opponent_pool: Vec<String>, randomize_spawns: bool, seed: u64, action_repeat: u32) -> PyResult<Self> {
        if opponent_pool.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "opponent_pool must not be empty",
            ));
        }

        // Validate all policy names
        for name in &opponent_pool {
            resolve_policy(name)?;
        }

        let mut rng = Pcg64::seed_from_u64(seed);
        let default_config = SimConfig::default();
        let envs: Vec<EnvInstance> = (0..n_envs)
            .map(|_| {
                let opp_idx = rng.gen_range(0..opponent_pool.len());
                let env_seed = rng.gen::<u64>();
                EnvInstance::new(&opponent_pool[opp_idx], env_seed, randomize_spawns, default_config)
            })
            .collect();

        Ok(Self {
            envs,
            n_envs,
            opponent_pool,
            randomize: randomize_spawns,
            action_repeat: action_repeat.max(1),
            weights: RewardWeights::default(),
            rng,
            config_ranges: SimConfigRanges::default(),
        })
    }

    /// Update the opponent pool for future resets.
    fn set_opponent_pool(&mut self, pool: Vec<String>) -> PyResult<()> {
        if pool.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "opponent_pool must not be empty",
            ));
        }
        for name in &pool {
            resolve_policy(name)?;
        }
        self.opponent_pool = pool;
        Ok(())
    }

    /// Set reward weights.
    #[pyo3(signature = (
        damage_dealt=None,
        damage_taken=None,
        win=None,
        lose=None,
        approach=None,
        alive=None,
        proximity=None,
        facing=None,
    ))]
    fn set_rewards(
        &mut self,
        damage_dealt: Option<f32>,
        damage_taken: Option<f32>,
        win: Option<f32>,
        lose: Option<f32>,
        approach: Option<f32>,
        alive: Option<f32>,
        proximity: Option<f32>,
        facing: Option<f32>,
    ) {
        if let Some(v) = damage_dealt { self.weights.damage_dealt = v; }
        if let Some(v) = damage_taken { self.weights.damage_taken = v; }
        if let Some(v) = win { self.weights.win = v; }
        if let Some(v) = lose { self.weights.lose = v; }
        if let Some(v) = approach { self.weights.approach = v; }
        if let Some(v) = alive { self.weights.alive = v; }
        if let Some(v) = proximity { self.weights.proximity = v; }
        if let Some(v) = facing { self.weights.facing = v; }
    }

    /// Reset all environments. Returns obs as numpy array (n_envs, OBS_SIZE).
    fn reset<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        // Generate reset params sequentially (RNG is not Send)
        let params: Vec<(String, u64, SimConfig)> = (0..self.n_envs)
            .map(|_| {
                let opp_idx = self.rng.gen_range(0..self.opponent_pool.len());
                let seed = self.rng.gen::<u64>();
                let config = if self.config_ranges.is_active() {
                    self.config_ranges.sample(&mut self.rng)
                } else {
                    SimConfig::default()
                };
                (self.opponent_pool[opp_idx].clone(), seed, config)
            })
            .collect();

        let randomize = self.randomize;

        // Reset in parallel
        let obs_arrays: Vec<[f32; OBS_SIZE]> = self.envs
            .par_iter_mut()
            .zip(params.into_par_iter())
            .map(|(env, (opp_name, seed, config))| {
                env.reset(&opp_name, seed, randomize, config)
            })
            .collect();

        // Write directly into numpy buffer
        let obs_py = PyArray2::<f32>::zeros_bound(py, [self.n_envs, OBS_SIZE], false);
        unsafe {
            let buf = obs_py.as_slice_mut().unwrap();
            for (i, obs) in obs_arrays.iter().enumerate() {
                buf[i * OBS_SIZE..(i + 1) * OBS_SIZE].copy_from_slice(obs);
            }
        }
        obs_py
    }

    /// Step all environments in parallel.
    ///
    /// Args:
    ///     actions: numpy array (n_envs, ACTION_SIZE) float32, C-contiguous
    ///
    /// Returns: (obs, rewards, dones, infos)
    ///     obs:     numpy (n_envs, OBS_SIZE) float32
    ///     rewards: numpy (n_envs,) float32
    ///     dones:   numpy (n_envs,) bool
    ///     infos:   list of n_envs dicts (only populated for done envs)
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: PyReadonlyArray2<f32>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<bool>>,
        Bound<'py, PyList>,
    )> {
        let actions_arr = actions.as_array();
        if actions_arr.shape() != [self.n_envs, ACTION_SIZE] {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected actions shape ({}, {}), got {:?}",
                self.n_envs, ACTION_SIZE, actions_arr.shape()
            )));
        }

        // Read actions from numpy (zero-copy view)
        let action_arrays: Vec<[f32; ACTION_SIZE]> = (0..self.n_envs)
            .map(|i| {
                let row = actions_arr.row(i);
                [row[0], row[1], row[2]]
            })
            .collect();

        let weights = &self.weights;
        let repeat = self.action_repeat;

        // Step all envs in parallel (the hot path — no GIL, no Python objects)
        // Each env runs `action_repeat` physics ticks with the same action.
        let results: Vec<StepResult> = self.envs
            .par_iter_mut()
            .zip(action_arrays.into_par_iter())
            .map(|(env, action)| env.step_repeat(action, weights, repeat))
            .collect();

        // Generate reset params for any done envs (sequential, needs RNG)
        let reset_params: Vec<Option<(String, u64, SimConfig)>> = results
            .iter()
            .map(|r| {
                if r.done {
                    let opp_idx = self.rng.gen_range(0..self.opponent_pool.len());
                    let seed = self.rng.gen::<u64>();
                    let config = if self.config_ranges.is_active() {
                        self.config_ranges.sample(&mut self.rng)
                    } else {
                        SimConfig::default()
                    };
                    Some((self.opponent_pool[opp_idx].clone(), seed, config))
                } else {
                    None
                }
            })
            .collect();

        // Auto-reset done envs in parallel
        let randomize = self.randomize;
        let reset_obs: Vec<Option<[f32; OBS_SIZE]>> = self.envs
            .par_iter_mut()
            .zip(reset_params.into_par_iter())
            .map(|(env, params)| {
                if let Some((opp_name, seed, config)) = params {
                    Some(env.reset(&opp_name, seed, randomize, config))
                } else {
                    None
                }
            })
            .collect();

        // Allocate output numpy arrays and write directly into their buffers
        let obs_py = PyArray2::<f32>::zeros_bound(py, [self.n_envs, OBS_SIZE], false);
        let rew_py = PyArray1::<f32>::zeros_bound(py, self.n_envs, false);
        let done_py = PyArray1::<bool>::zeros_bound(py, self.n_envs, false);

        let info_list = PyList::empty_bound(py);

        unsafe {
            let obs_buf = obs_py.as_slice_mut().unwrap();
            let rew_buf = rew_py.as_slice_mut().unwrap();
            let done_buf = done_py.as_slice_mut().unwrap();

            for (i, result) in results.iter().enumerate() {
                // Use reset obs if env was done, otherwise step obs
                let obs_data = if let Some(new_obs) = &reset_obs[i] {
                    new_obs
                } else {
                    &result.obs
                };
                obs_buf[i * OBS_SIZE..(i + 1) * OBS_SIZE].copy_from_slice(obs_data);
                rew_buf[i] = result.reward;
                done_buf[i] = result.done;
            }
        }

        // Build info list (only create dicts for done envs to minimize overhead)
        for result in results.iter() {
            let info = PyDict::new_bound(py);
            if result.done {
                info.set_item("outcome", &result.outcome)?;
                info.set_item("my_hp", result.my_hp)?;
                info.set_item("opp_hp", result.opp_hp)?;
            }
            info_list.append(info)?;
        }

        Ok((obs_py, rew_py, done_py, info_list))
    }

    /// Number of environments.
    #[getter]
    fn n(&self) -> usize {
        self.n_envs
    }

    #[getter]
    fn obs_size(&self) -> usize {
        OBS_SIZE
    }

    #[getter]
    fn action_size(&self) -> usize {
        ACTION_SIZE
    }

    /// Get/set action repeat (physics ticks per RL step).
    #[getter]
    fn action_repeat(&self) -> u32 {
        self.action_repeat
    }

    #[setter]
    fn set_action_repeat(&mut self, value: u32) {
        self.action_repeat = value.max(1);
    }

    /// Set domain randomization ranges. Pass a dict of parameter name → (min, max) tuples.
    /// Only specified parameters are randomized; unspecified ones use defaults.
    /// Pass an empty dict to disable domain randomization.
    ///
    /// Example:
    ///     env.set_domain_randomization({
    ///         "gravity": (60.0, 100.0),
    ///         "bullet_speed": (350.0, 450.0),
    ///     })
    fn set_domain_randomization(&mut self, ranges: &Bound<'_, PyDict>) -> PyResult<()> {
        self.config_ranges = SimConfigRanges::from_py_dict(ranges)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Self-play helpers — symmetric reward computation
// ---------------------------------------------------------------------------

/// Compute reward from the perspective of `me_idx` (0 or 1).
///
/// This is a standalone helper so both players can reuse the same logic with
/// swapped indices, avoiding code duplication.
fn compute_reward_for_player(
    state: &SimState,
    me_idx: usize,
    prev_my_hp: u8,
    prev_opp_hp: u8,
    prev_distance: f32,
    done: bool,
    weights: &RewardWeights,
) -> f32 {
    let opp_idx = 1 - me_idx;
    let my_hp = state.fighters[me_idx].hp;
    let opp_hp = state.fighters[opp_idx].hp;
    let distance = (state.fighters[me_idx].position - state.fighters[opp_idx].position).length();

    let mut reward = 0.0f32;

    // Damage dealt (opponent lost HP)
    let opp_hp_delta = prev_opp_hp as f32 - opp_hp as f32;
    reward += opp_hp_delta * weights.damage_dealt;

    // Damage taken (I lost HP)
    let my_hp_delta = prev_my_hp as f32 - my_hp as f32;
    reward += my_hp_delta * weights.damage_taken;

    // Approach reward (getting closer)
    let dist_delta = prev_distance - distance;
    reward += dist_delta * weights.approach;

    // Alive bonus
    if state.fighters[me_idx].alive {
        reward += weights.alive;
    }

    // Proximity bonus (within engagement range)
    if state.fighters[me_idx].alive && distance < 250.0 {
        reward += weights.proximity;
    }

    // Facing reward (pointing at opponent AND within bullet range)
    if state.fighters[me_idx].alive && distance < 300.0 && distance > 1.0 {
        let me = &state.fighters[me_idx];
        let rel = state.fighters[opp_idx].position - me.position;
        let hx = me.yaw.cos();
        let hy = me.yaw.sin();
        let inv_dist = 1.0 / distance;
        let dot = hx * rel.x * inv_dist + hy * rel.y * inv_dist;
        if dot > 0.866 {
            reward += weights.facing;
        }
    }

    // Terminal rewards (symmetric: my win → w_win, opponent win → w_lose)
    if done {
        let (outcome, _reason) = state.outcome();
        match (me_idx, outcome) {
            (0, MatchOutcome::Player0Win) | (1, MatchOutcome::Player1Win) => {
                reward += weights.win;
            }
            (0, MatchOutcome::Player1Win) | (1, MatchOutcome::Player0Win) => {
                reward += weights.lose;
            }
            (_, MatchOutcome::Draw) => {} // no bonus
            _ => unreachable!(),
        }
    }

    reward
}

// ---------------------------------------------------------------------------
// SelfPlayBatchEnv — both players controlled by Python (with optional scripted P1)
// ---------------------------------------------------------------------------

/// Whether a self-play env uses a neural P1 (from Python) or a scripted Rust policy.
enum EnvMode {
    /// P1 actions come from Python (neural self-play).
    Neural,
    /// P1 actions are computed internally by a scripted Rust policy.
    Scripted {
        opponent: Box<dyn Policy>,
    },
}

/// Helper: create an EnvMode based on scripted fraction and pool.
fn make_env_mode(rng: &mut Pcg64, scripted_fraction: f64, scripted_pool: &[String], config: SimConfig) -> EnvMode {
    if scripted_pool.is_empty() || scripted_fraction <= 0.0 {
        return EnvMode::Neural;
    }
    if rng.gen::<f64>() < scripted_fraction {
        let idx = rng.gen_range(0..scripted_pool.len());
        EnvMode::Scripted {
            opponent: make_policy_with_config(&scripted_pool[idx], config),
        }
    } else {
        EnvMode::Neural
    }
}

/// Per-env instance for self-play. Tracks reward state for BOTH players.
/// Supports optional scripted P1 opponent (for anchoring against scripted bots).
struct SelfPlayEnvInstance {
    state: SimState,
    config: SimConfig,
    tick: u32,
    mode: EnvMode,
    // Player 0 reward tracking
    p0_prev_hp: u8,
    p0_prev_opp_hp: u8,
    p0_prev_distance: f32,
    // Player 1 reward tracking
    p1_prev_hp: u8,
    p1_prev_opp_hp: u8,
    p1_prev_distance: f32,
}

/// Result of stepping a single self-play env (plain data, no Python objects).
struct SelfPlayStepResult {
    obs_p0: [f32; OBS_SIZE],
    obs_p1: [f32; OBS_SIZE],
    reward_p0: f32,
    reward_p1: f32,
    done: bool,
    outcome: String,
    p0_hp: u8,
    p1_hp: u8,
}

impl SelfPlayEnvInstance {
    fn new(seed: u64, randomize: bool, mode: EnvMode, config: SimConfig) -> Self {
        let state = SimState::new_with_seed_and_config(seed, randomize, config);
        let dist = (state.fighters[0].position - state.fighters[1].position).length();
        Self {
            state,
            config,
            tick: 0,
            mode,
            p0_prev_hp: config.max_hp,
            p0_prev_opp_hp: config.max_hp,
            p0_prev_distance: dist,
            p1_prev_hp: config.max_hp,
            p1_prev_opp_hp: config.max_hp,
            p1_prev_distance: dist,
        }
    }

    fn reset(&mut self, seed: u64, randomize: bool, new_mode: EnvMode, config: SimConfig) -> ([f32; OBS_SIZE], [f32; OBS_SIZE]) {
        self.config = config;
        self.state = SimState::new_with_seed_and_config(seed, randomize, config);
        self.mode = new_mode;
        let dist = (self.state.fighters[0].position - self.state.fighters[1].position).length();
        self.tick = 0;
        self.p0_prev_hp = config.max_hp;
        self.p0_prev_opp_hp = config.max_hp;
        self.p0_prev_distance = dist;
        self.p1_prev_hp = config.max_hp;
        self.p1_prev_opp_hp = config.max_hp;
        self.p1_prev_distance = dist;

        let obs_p0 = self.state.observe(0).data;
        let obs_p1 = self.state.observe(1).data;
        (obs_p0, obs_p1)
    }

    fn is_scripted(&self) -> bool {
        matches!(self.mode, EnvMode::Scripted { .. })
    }

    /// Run a single physics tick without calling observe for RL agents.
    /// Scripted opponent observes each tick. Returns (reward_p0, reward_p1, done, outcome).
    fn step_tick(
        &mut self,
        act0: &Action,
        p1_action: [f32; ACTION_SIZE],
        weights: &RewardWeights,
    ) -> (f32, f32, bool, String) {
        // P1 action: use scripted policy if in scripted mode, otherwise use provided action
        let act1 = match &mut self.mode {
            EnvMode::Neural => Action::from_raw(p1_action),
            EnvMode::Scripted { opponent, .. } => {
                let obs_p1 = self.state.observe(1);
                opponent.act(&obs_p1)
            }
        };

        // Step physics
        self.state.step(&[*act0, act1]);
        self.tick += 1;

        let done = self.state.is_terminal();

        // Compute rewards from both perspectives
        let reward_p0 = compute_reward_for_player(
            &self.state,
            0,
            self.p0_prev_hp,
            self.p0_prev_opp_hp,
            self.p0_prev_distance,
            done,
            weights,
        );
        let reward_p1 = compute_reward_for_player(
            &self.state,
            1,
            self.p1_prev_hp,
            self.p1_prev_opp_hp,
            self.p1_prev_distance,
            done,
            weights,
        );

        // Update tracking for both players
        let distance = (self.state.fighters[0].position - self.state.fighters[1].position).length();
        self.p0_prev_hp = self.state.fighters[0].hp;
        self.p0_prev_opp_hp = self.state.fighters[1].hp;
        self.p0_prev_distance = distance;
        self.p1_prev_hp = self.state.fighters[1].hp;
        self.p1_prev_opp_hp = self.state.fighters[0].hp;
        self.p1_prev_distance = distance;

        let outcome_str = if done {
            let (outcome, _) = self.state.outcome();
            match outcome {
                MatchOutcome::Player0Win => "Player0Win",
                MatchOutcome::Player1Win => "Player1Win",
                MatchOutcome::Draw => "Draw",
            }
        } else {
            ""
        };

        (reward_p0, reward_p1, done, outcome_str.to_string())
    }

    /// Step with action repeat: run `repeat` physics ticks with the same actions,
    /// accumulating rewards for both players. RL agents observe ONCE after the loop.
    fn step_both_repeat(
        &mut self,
        p0_action: [f32; ACTION_SIZE],
        p1_action: [f32; ACTION_SIZE],
        weights: &RewardWeights,
        repeat: u32,
    ) -> SelfPlayStepResult {
        let act0 = Action::from_raw(p0_action);
        let mut total_reward_p0 = 0.0f32;
        let mut total_reward_p1 = 0.0f32;
        let mut done = false;
        let mut outcome = String::new();

        for _ in 0..repeat {
            let (rp0, rp1, tick_done, tick_outcome) =
                self.step_tick(&act0, p1_action, weights);
            total_reward_p0 += rp0;
            total_reward_p1 += rp1;
            if tick_done {
                done = true;
                outcome = tick_outcome;
                break;
            }
        }

        // RL agents observe ONCE at the decision point (correct 12Hz frame history)
        SelfPlayStepResult {
            obs_p0: self.state.observe(0).data,
            obs_p1: self.state.observe(1).data,
            reward_p0: total_reward_p0,
            reward_p1: total_reward_p1,
            done,
            outcome,
            p0_hp: self.state.fighters[0].hp,
            p1_hp: self.state.fighters[1].hp,
        }
    }
}

/// Vectorized self-play environment. Both players are Python-controlled,
/// with optional scripted P1 opponents for a fraction of envs (anchoring).
///
/// Usage:
///     env = SelfPlayBatchEnv(64, True, seed=0, action_repeat=10)
///     env.set_scripted_fraction(0.2)
///     env.set_scripted_pool(["ace", "brawler"])
///     obs_p0, obs_p1 = env.reset()
///     obs_p0, obs_p1, rew_p0, rew_p1, dones, infos = env.step(actions_p0, actions_p1)
///     mask = env.scripted_mask  # which envs use scripted P1
#[pyclass(unsendable)]
struct SelfPlayBatchEnv {
    envs: Vec<SelfPlayEnvInstance>,
    n_envs: usize,
    randomize: bool,
    action_repeat: u32,
    weights: RewardWeights,
    rng: Pcg64,
    scripted_fraction: f64,
    scripted_pool: Vec<String>,
    config_ranges: SimConfigRanges,
}

#[pymethods]
impl SelfPlayBatchEnv {
    #[new]
    #[pyo3(signature = (n_envs, randomize_spawns=true, seed=0, action_repeat=1))]
    fn new(n_envs: usize, randomize_spawns: bool, seed: u64, action_repeat: u32) -> Self {
        let mut rng = Pcg64::seed_from_u64(seed);
        let default_config = SimConfig::default();
        let envs: Vec<SelfPlayEnvInstance> = (0..n_envs)
            .map(|_| {
                let env_seed = rng.gen::<u64>();
                SelfPlayEnvInstance::new(env_seed, randomize_spawns, EnvMode::Neural, default_config)
            })
            .collect();

        Self {
            envs,
            n_envs,
            randomize: randomize_spawns,
            action_repeat: action_repeat.max(1),
            weights: RewardWeights::default(),
            rng,
            scripted_fraction: 0.0,
            scripted_pool: Vec::new(),
            config_ranges: SimConfigRanges::default(),
        }
    }

    /// Set reward weights.
    #[pyo3(signature = (
        damage_dealt=None,
        damage_taken=None,
        win=None,
        lose=None,
        approach=None,
        alive=None,
        proximity=None,
        facing=None,
    ))]
    fn set_rewards(
        &mut self,
        damage_dealt: Option<f32>,
        damage_taken: Option<f32>,
        win: Option<f32>,
        lose: Option<f32>,
        approach: Option<f32>,
        alive: Option<f32>,
        proximity: Option<f32>,
        facing: Option<f32>,
    ) {
        if let Some(v) = damage_dealt { self.weights.damage_dealt = v; }
        if let Some(v) = damage_taken { self.weights.damage_taken = v; }
        if let Some(v) = win { self.weights.win = v; }
        if let Some(v) = lose { self.weights.lose = v; }
        if let Some(v) = approach { self.weights.approach = v; }
        if let Some(v) = alive { self.weights.alive = v; }
        if let Some(v) = proximity { self.weights.proximity = v; }
        if let Some(v) = facing { self.weights.facing = v; }
    }

    /// Reset all environments. Returns (obs_p0, obs_p1) as numpy arrays (n_envs, OBS_SIZE).
    fn reset<'py>(
        &mut self,
        py: Python<'py>,
    ) -> (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>) {
        // Generate reset seeds and modes sequentially (RNG is not Send)
        let params: Vec<(u64, EnvMode, SimConfig)> = (0..self.n_envs)
            .map(|_| {
                let seed = self.rng.gen::<u64>();
                let config = if self.config_ranges.is_active() {
                    self.config_ranges.sample(&mut self.rng)
                } else {
                    SimConfig::default()
                };
                let mode = make_env_mode(&mut self.rng, self.scripted_fraction, &self.scripted_pool, config);
                (seed, mode, config)
            })
            .collect();

        let randomize = self.randomize;

        // Reset in parallel
        let obs_pairs: Vec<([f32; OBS_SIZE], [f32; OBS_SIZE])> = self.envs
            .par_iter_mut()
            .zip(params.into_par_iter())
            .map(|(env, (seed, mode, config))| env.reset(seed, randomize, mode, config))
            .collect();

        // Write directly into numpy buffers
        let obs_p0_py = PyArray2::<f32>::zeros_bound(py, [self.n_envs, OBS_SIZE], false);
        let obs_p1_py = PyArray2::<f32>::zeros_bound(py, [self.n_envs, OBS_SIZE], false);
        unsafe {
            let buf_p0 = obs_p0_py.as_slice_mut().unwrap();
            let buf_p1 = obs_p1_py.as_slice_mut().unwrap();
            for (i, (o0, o1)) in obs_pairs.iter().enumerate() {
                buf_p0[i * OBS_SIZE..(i + 1) * OBS_SIZE].copy_from_slice(o0);
                buf_p1[i * OBS_SIZE..(i + 1) * OBS_SIZE].copy_from_slice(o1);
            }
        }
        (obs_p0_py, obs_p1_py)
    }

    /// Step all environments in parallel.
    ///
    /// Args:
    ///     actions_p0: numpy array (n_envs, ACTION_SIZE) float32
    ///     actions_p1: numpy array (n_envs, ACTION_SIZE) float32
    ///
    /// Returns: (obs_p0, obs_p1, rew_p0, rew_p1, dones, infos)
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions_p0: PyReadonlyArray2<f32>,
        actions_p1: PyReadonlyArray2<f32>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<bool>>,
        Bound<'py, PyList>,
    )> {
        let arr_p0 = actions_p0.as_array();
        let arr_p1 = actions_p1.as_array();
        if arr_p0.shape() != [self.n_envs, ACTION_SIZE] {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected actions_p0 shape ({}, {}), got {:?}",
                self.n_envs, ACTION_SIZE, arr_p0.shape()
            )));
        }
        if arr_p1.shape() != [self.n_envs, ACTION_SIZE] {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected actions_p1 shape ({}, {}), got {:?}",
                self.n_envs, ACTION_SIZE, arr_p1.shape()
            )));
        }

        // Read actions from numpy
        let actions_0: Vec<[f32; ACTION_SIZE]> = (0..self.n_envs)
            .map(|i| {
                let row = arr_p0.row(i);
                [row[0], row[1], row[2]]
            })
            .collect();
        let actions_1: Vec<[f32; ACTION_SIZE]> = (0..self.n_envs)
            .map(|i| {
                let row = arr_p1.row(i);
                [row[0], row[1], row[2]]
            })
            .collect();

        let weights = &self.weights;
        let repeat = self.action_repeat;

        // Step all envs in parallel (hot path — no GIL, no Python objects)
        let results: Vec<SelfPlayStepResult> = self.envs
            .par_iter_mut()
            .zip(actions_0.into_par_iter().zip(actions_1.into_par_iter()))
            .map(|(env, (a0, a1))| env.step_both_repeat(a0, a1, weights, repeat))
            .collect();

        // Generate reset params for done envs (sequential, needs RNG)
        let reset_params: Vec<Option<(u64, EnvMode, SimConfig)>> = results
            .iter()
            .map(|r| {
                if r.done {
                    let seed = self.rng.gen::<u64>();
                    let config = if self.config_ranges.is_active() {
                        self.config_ranges.sample(&mut self.rng)
                    } else {
                        SimConfig::default()
                    };
                    let mode = make_env_mode(&mut self.rng, self.scripted_fraction, &self.scripted_pool, config);
                    Some((seed, mode, config))
                } else {
                    None
                }
            })
            .collect();

        // Auto-reset done envs in parallel
        let randomize = self.randomize;
        let reset_obs: Vec<Option<([f32; OBS_SIZE], [f32; OBS_SIZE])>> = self.envs
            .par_iter_mut()
            .zip(reset_params.into_par_iter())
            .map(|(env, params)| {
                params.map(|(seed, mode, config)| env.reset(seed, randomize, mode, config))
            })
            .collect();

        // Allocate output numpy arrays
        let obs_p0_py = PyArray2::<f32>::zeros_bound(py, [self.n_envs, OBS_SIZE], false);
        let obs_p1_py = PyArray2::<f32>::zeros_bound(py, [self.n_envs, OBS_SIZE], false);
        let rew_p0_py = PyArray1::<f32>::zeros_bound(py, self.n_envs, false);
        let rew_p1_py = PyArray1::<f32>::zeros_bound(py, self.n_envs, false);
        let done_py = PyArray1::<bool>::zeros_bound(py, self.n_envs, false);

        let info_list = PyList::empty_bound(py);

        unsafe {
            let buf_obs_p0 = obs_p0_py.as_slice_mut().unwrap();
            let buf_obs_p1 = obs_p1_py.as_slice_mut().unwrap();
            let buf_rew_p0 = rew_p0_py.as_slice_mut().unwrap();
            let buf_rew_p1 = rew_p1_py.as_slice_mut().unwrap();
            let buf_done = done_py.as_slice_mut().unwrap();

            for (i, result) in results.iter().enumerate() {
                // Use reset obs if env was done, otherwise step obs
                let (o0, o1) = if let Some((new_o0, new_o1)) = &reset_obs[i] {
                    (new_o0, new_o1)
                } else {
                    (&result.obs_p0, &result.obs_p1)
                };
                buf_obs_p0[i * OBS_SIZE..(i + 1) * OBS_SIZE].copy_from_slice(o0);
                buf_obs_p1[i * OBS_SIZE..(i + 1) * OBS_SIZE].copy_from_slice(o1);
                buf_rew_p0[i] = result.reward_p0;
                buf_rew_p1[i] = result.reward_p1;
                buf_done[i] = result.done;
            }
        }

        // Build info list (only create dicts for done envs to minimize overhead)
        for result in results.iter() {
            let info = PyDict::new_bound(py);
            if result.done {
                info.set_item("outcome", &result.outcome)?;
                info.set_item("p0_hp", result.p0_hp)?;
                info.set_item("p1_hp", result.p1_hp)?;
            }
            info_list.append(info)?;
        }

        Ok((obs_p0_py, obs_p1_py, rew_p0_py, rew_p1_py, done_py, info_list))
    }

    /// Number of environments.
    #[getter]
    fn n(&self) -> usize {
        self.n_envs
    }

    #[getter]
    fn obs_size(&self) -> usize {
        OBS_SIZE
    }

    #[getter]
    fn action_size(&self) -> usize {
        ACTION_SIZE
    }

    /// Get/set action repeat (physics ticks per RL step).
    #[getter]
    fn action_repeat(&self) -> u32 {
        self.action_repeat
    }

    #[setter]
    fn set_action_repeat(&mut self, value: u32) {
        self.action_repeat = value.max(1);
    }

    /// Set the fraction of envs that use scripted opponents (0.0 = all neural, 1.0 = all scripted).
    /// Takes effect on next reset or auto-reset of each env.
    fn set_scripted_fraction(&mut self, fraction: f64) {
        self.scripted_fraction = fraction.clamp(0.0, 1.0);
    }

    /// Set the pool of scripted opponent names (e.g. ["ace", "brawler"]).
    fn set_scripted_pool(&mut self, pool: Vec<String>) -> PyResult<()> {
        for name in &pool {
            resolve_policy(name)?;
        }
        self.scripted_pool = pool;
        Ok(())
    }

    /// Get the scripted fraction.
    #[getter]
    fn scripted_fraction(&self) -> f64 {
        self.scripted_fraction
    }

    /// Returns a boolean mask: True for envs using scripted P1, False for neural P1.
    #[getter]
    fn scripted_mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        let mask: Vec<bool> = self.envs.iter().map(|e| e.is_scripted()).collect();
        let arr = PyArray1::<bool>::zeros_bound(py, self.n_envs, false);
        unsafe {
            let buf = arr.as_slice_mut().unwrap();
            buf.copy_from_slice(&mask);
        }
        arr
    }

    /// Returns the number of envs currently using scripted opponents.
    #[getter]
    fn n_scripted(&self) -> usize {
        self.envs.iter().filter(|e| e.is_scripted()).count()
    }

    /// Set domain randomization ranges. Pass a dict of parameter name → (min, max) tuples.
    /// Only specified parameters are randomized; unspecified ones use defaults.
    /// Pass an empty dict to disable domain randomization.
    ///
    /// Example:
    ///     env.set_domain_randomization({
    ///         "gravity": (60.0, 100.0),
    ///         "bullet_speed": (350.0, 450.0),
    ///     })
    fn set_domain_randomization(&mut self, ranges: &Bound<'_, PyDict>) -> PyResult<()> {
        self.config_ranges = SimConfigRanges::from_py_dict(ranges)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// BCDataCollector — behavioral cloning data collection
// ---------------------------------------------------------------------------

/// Collect (observation, action) pairs from scripted policies for behavioral cloning.
///
/// Usage:
///     collector = BCDataCollector("chaser")
///     obs, actions = collector.collect(1000, ["dogfighter", "ace", "brawler"])
#[pyclass(unsendable)]
struct BCDataCollector {
    policy_name: String,
    control_period: u32,
}

#[pymethods]
impl BCDataCollector {
    #[new]
    #[pyo3(signature = (policy_name, control_period=10))]
    fn new(policy_name: String, control_period: u32) -> PyResult<Self> {
        // Validate policy name
        resolve_policy(&policy_name)?;
        Ok(Self {
            policy_name,
            control_period: control_period.max(1),
        })
    }

    /// Collect behavioral cloning data by running the scripted policy vs opponents.
    ///
    /// Args:
    ///     n_episodes: Number of episodes to collect
    ///     opponent_names: List of opponent policy names to play against
    ///
    /// Returns: (obs_array, action_array) as numpy arrays
    ///     obs_array: (N, OBS_SIZE) float32 — stacked observations at decision points
    ///     action_array: (N, ACTION_SIZE) float32 — raw actions taken
    fn collect<'py>(
        &self,
        py: Python<'py>,
        n_episodes: usize,
        opponent_names: Vec<String>,
    ) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>)> {
        // Validate opponent names
        for name in &opponent_names {
            resolve_policy(name)?;
        }

        let mut rng = Pcg64::seed_from_u64(42);
        let mut all_obs: Vec<[f32; OBS_SIZE]> = Vec::new();
        let mut all_actions: Vec<[f32; ACTION_SIZE]> = Vec::new();

        for _ in 0..n_episodes {
            let opp_idx = rng.gen_range(0..opponent_names.len());
            let seed = rng.gen::<u64>();
            let config = SimConfig::default();

            let mut state = SimState::new_with_seed_and_config(seed, true, config);
            let mut policy = make_policy_with_config(&self.policy_name, config);
            let mut opponent = make_policy_with_config(&opponent_names[opp_idx], config);

            let mut tick = 0u32;
            while tick < MAX_TICKS && !state.is_terminal() {
                // At decision points, record observation and action for BOTH players
                if tick % self.control_period == 0 {
                    // Record from our policy's perspective (player 0)
                    let obs0 = state.observe(0);
                    let action0 = policy.act(&obs0);
                    all_obs.push(obs0.data);
                    all_actions.push(action0.to_raw());

                    // Also record from opponent's perspective (player 1) — more data
                    let obs1 = state.observe(1);
                    let action1 = opponent.act(&obs1);
                    all_obs.push(obs1.data);
                    all_actions.push(action1.to_raw());

                    // Run physics for control_period ticks
                    for _ in 0..self.control_period {
                        if state.is_terminal() {
                            break;
                        }
                        // Opponent re-observes each tick for scripted policy
                        let opp_obs = state.observe(1);
                        let opp_act = opponent.act(&opp_obs);
                        state.step(&[action0, opp_act]);
                        tick += 1;
                    }
                } else {
                    // Should not reach here given our loop structure
                    tick += 1;
                }
            }
        }

        let n = all_obs.len();
        let obs_py = PyArray2::<f32>::zeros_bound(py, [n, OBS_SIZE], false);
        let act_py = PyArray2::<f32>::zeros_bound(py, [n, ACTION_SIZE], false);
        unsafe {
            let obs_buf = obs_py.as_slice_mut().unwrap();
            let act_buf = act_py.as_slice_mut().unwrap();
            for (i, obs) in all_obs.iter().enumerate() {
                obs_buf[i * OBS_SIZE..(i + 1) * OBS_SIZE].copy_from_slice(obs);
            }
            for (i, act) in all_actions.iter().enumerate() {
                act_buf[i * ACTION_SIZE..(i + 1) * ACTION_SIZE].copy_from_slice(act);
            }
        }

        Ok((obs_py, act_py))
    }
}

/// Python module definition.
#[pymodule]
fn dogfight_pyenv(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DogfightEnv>()?;
    m.add_class::<BatchEnv>()?;
    m.add_class::<SelfPlayBatchEnv>()?;
    m.add_class::<BCDataCollector>()?;
    m.add("OBS_SIZE", OBS_SIZE)?;
    m.add("ACTION_SIZE", ACTION_SIZE)?;
    m.add("MAX_TICKS", MAX_TICKS)?;
    m.add("SINGLE_FRAME_OBS_SIZE", SINGLE_FRAME_OBS_SIZE)?;
    m.add("FRAME_STACK_SIZE", FRAME_STACK_SIZE)?;
    m.add("CONTROL_PERIOD", CONTROL_PERIOD)?;
    Ok(())
}
