import { useState, useEffect, useRef } from 'react'
import { StepMessage } from '../types'
import { useAnimatedValue } from '../hooks/useAnimatedValue'

interface Props {
  steps: StepMessage[]
}

type MetricKey = 'tokens' | 'avgRelDelta' | 'entropy' | 'vram'

function deltaColor(v: number): string {
  if (v <= 0.3) return '#64ffda'
  if (v <= 0.6) return '#f48849'
  return '#e53935'
}

function entropyColor(v: number): string {
  if (v <= 3) return '#64ffda'
  if (v <= 6) return '#f48849'
  return '#e53935'
}

function usePulse(value: unknown, duration = 300): boolean {
  const [pulsing, setPulsing] = useState(false)
  const prevRef = useRef(value)

  useEffect(() => {
    if (prevRef.current !== value) {
      prevRef.current = value
      setPulsing(true)
      const timer = setTimeout(() => setPulsing(false), duration)
      return () => clearTimeout(timer)
    }
  }, [value, duration])

  return pulsing
}

export default function StatsPanel({ steps }: Props) {
  const lastStep = steps[steps.length - 1]
  if (!lastStep) return null

  const tokenCount = steps.length

  const allDeltas = steps.flatMap(s => s.layers.map(l => l.relative_delta))
  const avgDelta = allDeltas.reduce((a, b) => a + b, 0) / Math.max(allDeltas.length, 1)
  const lastEntropy = lastStep.global.prediction_entropy
  const lastVram = lastStep.global.vram_total_mb

  const animatedTokens = useAnimatedValue(tokenCount)
  const animatedDelta = useAnimatedValue(avgDelta)
  const animatedEntropy = useAnimatedValue(lastEntropy)
  const animatedVram = useAnimatedValue(lastVram > 0 ? lastVram : 0)

  const tokensPulse = usePulse(tokenCount)
  const deltaPulse = usePulse(avgDelta)
  const entropyPulse = usePulse(lastEntropy)
  const vramPulse = usePulse(lastVram)

  const cardStyle = (color: string, isHero = false, pulsing = false): React.CSSProperties => ({
    background: '#16213e',
    borderRadius: 6,
    padding: '10px 16px',
    minWidth: 100,
    borderLeft: isHero ? `3px solid #64ffda` : '3px solid transparent',
    transition: 'transform 150ms ease-out, opacity 150ms ease-out',
    transform: pulsing ? 'scale(1.02)' : 'scale(1)',
    opacity: pulsing ? 1 : 0.92,
  })

  const stats: { key: MetricKey; label: string; value: string; color: string; hero: boolean; pulse: boolean }[] = [
    {
      key: 'tokens',
      label: 'Tokens',
      value: Math.round(animatedTokens).toString(),
      color: '#64ffda',
      hero: true,
      pulse: tokensPulse,
    },
    {
      key: 'avgRelDelta',
      label: 'Avg RelDelta',
      value: animatedDelta.toFixed(4),
      color: deltaColor(avgDelta),
      hero: false,
      pulse: deltaPulse,
    },
    {
      key: 'entropy',
      label: 'Entropy',
      value: animatedEntropy.toFixed(2),
      color: entropyColor(lastEntropy),
      hero: false,
      pulse: entropyPulse,
    },
    {
      key: 'vram',
      label: 'VRAM',
      value: lastVram > 0 ? `${Math.round(animatedVram)} MB` : 'N/A',
      color: '#5c6bc0',
      hero: false,
      pulse: vramPulse,
    },
  ]

  return (
    <div style={{ display: 'flex', gap: 16, marginBottom: 16 }}>
      {stats.map(s => (
        <div key={s.key} style={cardStyle(s.color, s.hero, s.pulse)}>
          <div style={{ color: '#8892b0', fontSize: 11, textTransform: 'uppercase' }}>{s.label}</div>
          <div style={{ color: s.color, fontSize: s.hero ? 24 : 20, fontWeight: 600 }}>
            {s.value}
          </div>
        </div>
      ))}
    </div>
  )
}
