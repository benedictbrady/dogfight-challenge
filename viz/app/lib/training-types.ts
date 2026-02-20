export interface DataPoint {
  step: number;
  wall_time: number;
  value: number;
}

export interface TrainingMetrics {
  run_name: string;
  parsed_at: string;
  losses: Record<string, DataPoint[]>;
  charts: Record<string, DataPoint[]>;
  selfplay: Record<string, DataPoint[]>;
  eval: Record<string, DataPoint[]>;
}

export interface PoolEntry {
  name: string;
  elo: number;
  games: number;
  wins: number;
  losses: number;
  draws: number;
  generation: number; // maps from update_num in raw pool.json
}

export interface PoolData {
  entries: PoolEntry[];
  updated_at?: string;
}

export type RunStatus = "live" | "completed" | "failed";

export interface RunInfo {
  name: string;
  status: RunStatus;
  has_metrics: boolean;
  has_pool: boolean;
  checkpoints: string[];
}

export interface CheckpointInfo {
  name: string;
  path: string;
}
