import { useMemo } from 'react'
import { StepMessage } from '../types'

interface Props {
  steps: StepMessage[]
  highlightedTokenIndex: number | null
  onTokenClick: (tokenIndex: number) => void
}

function entropyToColor(entropy: number, maxEntropy: number): string {
  const ratio = Math.min(entropy / Math.max(maxEntropy, 0.001), 1)
  if (ratio < 0.33) {
    const t = ratio / 0.33
    return `rgb(${Math.round(13 + t * 61)}, ${Math.round(71 + t * (20 - 71))}, ${Math.round(161 + t * (140 - 161))})`
  } else if (ratio < 0.66) {
    const t = (ratio - 0.33) / 0.33
    return `rgb(${Math.round(74 + t * (136 - 74))}, ${Math.round(20 + t * (14 - 20))}, ${Math.round(140 + t * (79 - 140))})`
  } else {
    const t = (ratio - 0.66) / 0.34
    return `rgb(${Math.round(136 + t * (191 - 136))}, ${Math.round(14 + t * (81 - 14))}, ${Math.round(79 + t * (12 - 79))})`
  }
}

export default function TokenList({ steps, highlightedTokenIndex, onTokenClick }: Props) {
  const maxEntropy = useMemo(() => {
    let max = 0
    for (const s of steps) {
      if (s.global.prediction_entropy > max) max = s.global.prediction_entropy
    }
    return max
  }, [steps])

  return (
    <div style={{ marginTop: 16 }}>
      <h3 style={{ color: '#ccd6f6', marginBottom: 8 }}>Generated Tokens</h3>

      {/* Color legend */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6, fontSize: 12,
      }}>
        <span style={{ color: '#8892b0' }}>Low entropy (confident)</span>
        <div style={{
          width: 120, height: 8, borderRadius: 4,
          background: `linear-gradient(to right, ${entropyToColor(0, 1)}, ${entropyToColor(0.5, 1)}, ${entropyToColor(1, 1)})`,
        }} />
        <span style={{ color: '#8892b0' }}>High entropy (uncertain)</span>
      </div>

      <div style={{
        background: '#16213e', borderRadius: 6, padding: 12,
        fontFamily: 'monospace', fontSize: 14, lineHeight: 1.8,
        maxHeight: 120, overflowY: 'auto',
      }}>
        {steps.map(s => {
          const isHighlighted = highlightedTokenIndex === s.token_idx
          return (
            <span
              key={s.token_idx}
              title={`token ${s.token_idx}, entropy=${s.global.prediction_entropy.toFixed(2)}`}
              onClick={() => onTokenClick(s.token_idx)}
              style={{
                display: 'inline-block', margin: '2px 3px', padding: '2px 6px',
                borderRadius: 3, cursor: 'pointer', color: '#e0e0e0',
                background: entropyToColor(s.global.prediction_entropy, maxEntropy),
                border: isHighlighted ? '2px solid #64ffda' : '2px solid transparent',
                transition: 'transform 0.1s, border-color 0.1s',
              }}
              onMouseEnter={e => {
                (e.currentTarget as HTMLElement).style.transform = 'scale(1.05)'
              }}
              onMouseLeave={e => {
                const el = e.currentTarget as HTMLElement
                el.style.transform = 'scale(1)'
                if (highlightedTokenIndex !== s.token_idx) {
                  el.style.borderColor = 'transparent'
                }
              }}
            >
              {s.generated_token}
            </span>
          )
        })}
      </div>
    </div>
  )
}
