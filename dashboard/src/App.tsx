import { useState, useCallback, useRef } from 'react'
import { SessionState, ServerMessage } from './types'
import HeatmapView from './components/HeatmapView'
import TokenList from './components/TokenList'
import StatsPanel from './components/StatsPanel'

const EMPTY_STATE: SessionState = {
  sessionStart: null,
  steps: [],
  sessionEnd: null,
}

function useWebSocket(url: string, onMessage: (msg: ServerMessage) => void) {
  const wsRef = useRef<WebSocket | null>(null)
  const connect = useCallback(() => {
    if (wsRef.current) wsRef.current.close()
    const ws = new WebSocket(url)
    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data) as ServerMessage
        onMessage(data)
      } catch { /* ignore parse errors */ }
    }
    ws.onerror = () => ws.close()
    wsRef.current = ws
    return ws
  }, [url, onMessage])

  const disconnect = useCallback(() => {
    wsRef.current?.close()
    wsRef.current = null
  }, [])

  return { connect, disconnect, wsRef }
}

export default function App() {
  const [sessionId, setSessionId] = useState('')
  const [connected, setConnected] = useState(false)
  const [state, setState] = useState<SessionState>(EMPTY_STATE)

  const handleMessage = useCallback((msg: ServerMessage) => {
    setState(prev => {
      if (msg.type === 'session_start') return { ...prev, sessionStart: msg }
      if (msg.type === 'session_end') return { ...prev, sessionEnd: msg }
      if (msg.type === 'step') return { ...prev, steps: [...prev.steps, msg] }
      return prev
    })
  }, [])

  const wsUrl = sessionId
    ? `ws://localhost:8765/ws/${sessionId}`
    : ''

  const { connect, disconnect } = useWebSocket(wsUrl, handleMessage)

  const handleConnect = () => {
    if (!sessionId.trim()) return
    setState(EMPTY_STATE)
    const ws = connect()
    ws.onopen = () => setConnected(true)
    ws.onclose = () => setConnected(false)
  }

  const { sessionStart, steps, sessionEnd } = state
  const numLayers = sessionStart?.num_layers ?? 0
  const layerNames = sessionStart?.layer_names ?? []

  return (
    <div style={{ padding: 20, maxWidth: 1200, margin: '0 auto' }}>
      <h1 style={{ color: '#64ffda', marginBottom: 8 }}>UniVis Dashboard</h1>

      {/* Connection Bar */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20, alignItems: 'center' }}>
        <input
          value={sessionId}
          onChange={e => setSessionId(e.target.value)}
          placeholder="Session ID"
          onKeyDown={e => e.key === 'Enter' && handleConnect()}
          style={{
            padding: '8px 12px', borderRadius: 4, border: '1px solid #333',
            background: '#16213e', color: '#e0e0e0', fontSize: 14, width: 240,
          }}
        />
        <button
          onClick={connected ? disconnect : handleConnect}
          style={{
            padding: '8px 16px', borderRadius: 4, border: 'none',
            background: connected ? '#e74c3c' : '#0f3460',
            color: '#fff', cursor: 'pointer', fontSize: 14,
          }}
        >
          {connected ? 'Disconnect' : 'Connect'}
        </button>
        {sessionStart && (
          <span style={{ color: '#8892b0', fontSize: 13 }}>
            {sessionStart.model_name} | {sessionStart.num_layers} layers | {sessionStart.project}
          </span>
        )}
        {sessionEnd && (
          <span style={{ color: '#64ffda', fontSize: 13 }}>
            Done: {sessionEnd.total_tokens} tokens in {(sessionEnd.total_time_ms / 1000).toFixed(1)}s
          </span>
        )}
      </div>

      {/* Stats */}
      {steps.length > 0 && <StatsPanel steps={steps} />}

      {/* Heatmap */}
      {steps.length > 0 && (
        <HeatmapView steps={steps} numLayers={numLayers} layerNames={layerNames} />
      )}

      {/* Token List */}
      {steps.length > 0 && <TokenList steps={steps} />}
    </div>
  )
}
