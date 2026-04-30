import { useState, useCallback, useRef, useMemo } from 'react'
import { SessionState, ServerMessage, StepMessage } from './types'
import HeatmapView from './components/HeatmapView'
import TokenList from './components/TokenList'
import StatsPanel from './components/StatsPanel'
import LayerFilter from './components/LayerFilter'

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

function buildExportHtml(state: SessionState): string {
  const { sessionStart, steps, sessionEnd } = state
  const allDeltas = steps.flatMap(s => s.layers.map(l => l.relative_delta))
  const avgDelta = allDeltas.length > 0
    ? (allDeltas.reduce((a, b) => a + b, 0) / allDeltas.length).toFixed(4)
    : 'N/A'

  const modelName = sessionStart?.model_name ?? 'Unknown'
  const totalTokens = sessionEnd?.total_tokens ?? steps.length
  const totalTime = sessionEnd ? (sessionEnd.total_time_ms / 1000).toFixed(1) + 's' : 'N/A'

  const rows = steps.map(s => {
    const cells = s.layers.map(l =>
      `<td style="padding:4px 8px;border:1px solid #333;text-align:right;font-size:12px;">${l.relative_delta.toFixed(4)}</td>`
    ).join('')
    return `<tr><td style="padding:4px 8px;border:1px solid #333;font-size:12px;font-weight:bold;">${s.generated_token}</td>${cells}</tr>`
  }).join('')

  const layerHeaders = sessionStart?.layer_names?.map(n =>
    `<th style="padding:4px 8px;border:1px solid #333;font-size:11px;">${n}</th>`
  ).join('') ?? ''

  return `<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>UniVis Report</title>
<style>
body{background:#0a192f;color:#ccd6f6;font-family:monospace;padding:24px;margin:0;}
h1{color:#64ffda;margin-bottom:4px;}
.stats{display:flex;gap:24px;margin:16px 0;}
.stat-label{color:#8892b0;font-size:12px;text-transform:uppercase;}
.stat-value{color:#64ffda;font-size:18px;font-weight:600;}
table{border-collapse:collapse;width:100%;}
th{background:#16213e;color:#8892b0;padding:4px 8px;border:1px solid #333;font-size:11px;}
td{padding:4px 8px;border:1px solid #333;}
</style></head>
<body>
<h1>UniVis Report</h1>
<div class="stats">
  <div><div class="stat-label">Model</div><div class="stat-value">${modelName}</div></div>
  <div><div class="stat-label">Tokens</div><div class="stat-value">${totalTokens}</div></div>
  <div><div class="stat-label">Avg RelDelta</div><div class="stat-value">${avgDelta}</div></div>
  <div><div class="stat-label">Time</div><div class="stat-value">${totalTime}</div></div>
</div>
<h2 style="color:#ccd6f6;font-size:14px;">Token-Level Relative Delta</h2>
<table>
<tr><th style="padding:4px 8px;border:1px solid #333;font-size:11px;">Token</th>${layerHeaders}</tr>
${rows}
</table>
</body></html>`
}

export default function App() {
  const [sessionId, setSessionId] = useState('')
  const [connected, setConnected] = useState(false)
  const [state, setState] = useState<SessionState>(EMPTY_STATE)
  const [selectedLayers, setSelectedLayers] = useState<Set<number>>(new Set<number>())

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
    setSelectedLayers(new Set<number>())
    const ws = connect()
    ws.onopen = () => setConnected(true)
    ws.onclose = () => setConnected(false)
  }

  const { sessionStart, steps, sessionEnd } = state
  const numLayers = sessionStart?.num_layers ?? 0
  const layerNames = sessionStart?.layer_names ?? []

  // Initialize selectedLayers when session starts
  const effectiveLayerNames = layerNames.length > 0
    ? layerNames
    : Array.from({ length: numLayers }, (_, i) => `Layer ${i}`)

  // Auto-select all layers when first step arrives and selection is empty
  const effectiveSelectedLayers = useMemo(() => {
    if (effectiveLayerNames.length > 0 && selectedLayers.size === 0) {
      return new Set(effectiveLayerNames.map((_, i) => i))
    }
    return selectedLayers
  }, [selectedLayers, effectiveLayerNames])

  const filteredSteps = useMemo(() => {
    if (effectiveSelectedLayers.size === effectiveLayerNames.length) return steps
    return steps.map(s => ({
      ...s,
      layers: s.layers.filter(l => effectiveSelectedLayers.has(l.idx)),
    }))
  }, [steps, effectiveSelectedLayers, effectiveLayerNames])

  const filteredLayerNames = useMemo(() => {
    return effectiveLayerNames.filter((_, i) => effectiveSelectedLayers.has(i))
  }, [effectiveLayerNames, effectiveSelectedLayers])

  const handleToggleLayer = (idx: number) => {
    setSelectedLayers(prev => {
      const next = new Set(prev)
      if (next.has(idx)) next.delete(idx)
      else next.add(idx)
      return next
    })
  }

  const handleSelectAll = () => {
    setSelectedLayers(new Set(effectiveLayerNames.map((_, i) => i)))
  }

  const handleDeselectAll = () => {
    setSelectedLayers(new Set<number>())
  }

  const handleExport = () => {
    const html = buildExportHtml(state)
    const blob = new Blob([html], { type: 'text/html' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `univis-report-${sessionId || 'session'}.html`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

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
        {steps.length > 0 && (
          <button
            onClick={handleExport}
            style={{
              padding: '8px 16px', borderRadius: 4, border: '1px solid #64ffda',
              background: 'transparent', color: '#64ffda', cursor: 'pointer', fontSize: 14,
            }}
          >
            Export Report
          </button>
        )}
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

      {/* Layer Filter */}
      {steps.length > 0 && (
        <LayerFilter
          layerNames={effectiveLayerNames}
          selectedLayers={effectiveSelectedLayers}
          onToggle={handleToggleLayer}
          onSelectAll={handleSelectAll}
          onDeselectAll={handleDeselectAll}
        />
      )}

      {/* Heatmap */}
      {filteredSteps.length > 0 && filteredLayerNames.length > 0 && (
        <HeatmapView steps={filteredSteps} numLayers={filteredLayerNames.length} layerNames={filteredLayerNames} />
      )}

      {/* Token List */}
      {steps.length > 0 && <TokenList steps={steps} />}
    </div>
  )
}
