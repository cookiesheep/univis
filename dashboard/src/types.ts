export interface SessionStart {
  type: 'session_start'
  session_id: string
  model_name: string
  num_layers: number
  layer_names: string[]
  project: string
}

export interface LayerMetrics {
  idx: number
  name: string
  relative_delta: number
  cosine_sim: number
  sparsity: number
}

export interface StepMessage {
  type: 'step'
  session_id: string
  token_idx: number
  timestamp_ms: number
  generated_token: string
  global: {
    vram_total_mb: number
    prediction_entropy: number
  }
  layers: LayerMetrics[]
}

export interface SessionEnd {
  type: 'session_end'
  session_id: string
  total_tokens: number
  total_time_ms: number
  num_layers: number
}

export type ServerMessage = SessionStart | StepMessage | SessionEnd

export interface SessionState {
  sessionStart: SessionStart | null
  steps: StepMessage[]
  sessionEnd: SessionEnd | null
}
