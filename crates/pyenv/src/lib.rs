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

/// Create a policy from name (infallible version for internal use).
fn make_policy(name: &str) -> Box<dyn Policy> {
    match name {
        "do_nothing" => Box::new(DoNothingPolicy),
        "chaser" => Box::new(ChaserPolicy::new()),
        "dogfighter" => Box::new(DogfighterPolicy::new()),
        "ace" => Box::new(AcePolicy::new()),
        "brawler" => Box::new(BrawlerPolicy::new()),
        _ => Box::new(DoNothingPolicy),
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
    fn new(opponent_name: &str, seed: u64, randomize: bool) -> Self {
        let state = SimState::new_with_seed(seed, randomize);
        let dist = (state.fighters[0].position - state.fighters[1].position).length();
        Self {
            state,
            opponent: make_policy(opponent_name),
            opponent_name: opponent_name.to_string(),
            tick: 0,
            prev_my_hp: MAX_HP,
            prev_opp_hp: MAX_HP,
            prev_distance: dist,
        }
    }

    fn reset(&mut self, opponent_name: &str, seed: u64, randomize: bool) -> [f32; OBS_SIZE] {
        self.state = SimState::new_with_seed(seed, randomize);
        self.opponent = make_policy(opponent_name);
        self.opponent_name = opponent_name.to_string();
        self.tick = 0;
        self.prev_my_hp = MAX_HP;
        self.prev_opp_hp = MAX_HP;
        self.prev_distance =
            (self.state.fighters[0].position - self.state.fighters[1].position).length();

        self.state.observe(0).data
    }

    fn step(&mut self, action: [f32; ACTION_SIZE], weights: &RewardWeights) -> StepResult {
        let my_action = Action::from_raw(action);

        // Opponent acts
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

        let obs = self.state.observe(0).data;

        StepResult {
            obs,
            reward,
            done,
            outcome: outcome_str.to_string(),
            my_hp,
            opp_hp,
        }
    }

    /// Step with action repeat: run `repeat` physics ticks with the same action,
    /// accumulating reward. Early-stops on done.
    fn step_repeat(
        &mut self,
        action: [f32; ACTION_SIZE],
        weights: &RewardWeights,
        repeat: u32,
    ) -> StepResult {
        let mut total_reward = 0.0f32;
        let mut result = StepResult {
            obs: [0.0; OBS_SIZE],
            reward: 0.0,
            done: false,
            outcome: String::new(),
            my_hp: MAX_HP,
            opp_hp: MAX_HP,
        };

        for _ in 0..repeat {
            result = self.step(action, weights);
            total_reward += result.reward;
            if result.done {
                break;
            }
        }

        result.reward = total_reward;
        result
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
        let envs: Vec<EnvInstance> = (0..n_envs)
            .map(|_| {
                let opp_idx = rng.gen_range(0..opponent_pool.len());
                let env_seed = rng.gen::<u64>();
                EnvInstance::new(&opponent_pool[opp_idx], env_seed, randomize_spawns)
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
        let params: Vec<(String, u64)> = (0..self.n_envs)
            .map(|_| {
                let opp_idx = self.rng.gen_range(0..self.opponent_pool.len());
                let seed = self.rng.gen::<u64>();
                (self.opponent_pool[opp_idx].clone(), seed)
            })
            .collect();

        let randomize = self.randomize;

        // Reset in parallel
        let obs_arrays: Vec<[f32; OBS_SIZE]> = self.envs
            .par_iter_mut()
            .zip(params.into_par_iter())
            .map(|(env, (opp_name, seed))| {
                env.reset(&opp_name, seed, randomize)
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
        let reset_params: Vec<Option<(String, u64)>> = results
            .iter()
            .map(|r| {
                if r.done {
                    let opp_idx = self.rng.gen_range(0..self.opponent_pool.len());
                    let seed = self.rng.gen::<u64>();
                    Some((self.opponent_pool[opp_idx].clone(), seed))
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
                if let Some((opp_name, seed)) = params {
                    Some(env.reset(&opp_name, seed, randomize))
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
}

/// Python module definition.
#[pymodule]
fn dogfight_pyenv(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DogfightEnv>()?;
    m.add_class::<BatchEnv>()?;
    m.add("OBS_SIZE", OBS_SIZE)?;
    m.add("ACTION_SIZE", ACTION_SIZE)?;
    m.add("MAX_TICKS", MAX_TICKS)?;
    Ok(())
}
