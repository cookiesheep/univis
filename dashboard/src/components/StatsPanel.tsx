import { StepMessage } from '../types'

interface Props {
  steps: StepMessage[]
}

export default function StatsPanel({ steps }: Props) {
  const lastStep = steps[steps.length - 1]
  if (!lastStep) return null

  const allDeltas = steps.flatMap(s => s.layers.map(l => l.relative_delta))
  const avgDelta = allDeltas.reduce((a, b) => a + b, 0) / Math.max(allDeltas.length, 1)
  const lastEntropy = lastStep.global.prediction_entropy
  const lastVram = lastStep.global.vram_total_mb

  const stats = [
    { label: 'Tokens', value: steps.length },
    { label: 'Avg RelDelta', value: avgDelta.toFixed(4) },
    { label: 'Entropy', value: lastEntropy.toFixed(2) },
    { label: 'VRAM', value: lastVram > 0 ? `${lastVram.toFixed(0)} MB` : 'N/A' },
  ]

  return (
    <div style={{ display: 'flex', gap: 16, marginBottom: 16 }}>
      {stats.map(s => (
        <div key={s.label} style={{
          background: '#16213e', borderRadius: 6, padding: '10px 16px', minWidth: 100,
        }}>
          <div style={{ color: '#8892b0', fontSize: 11, textTransform: 'uppercase' }}>{s.label}</div>
          <div style={{ color: '#64ffda', fontSize: 20, fontWeight: 600 }}>{s.value}</div>
        </div>
      ))}
    </div>
  )
}
