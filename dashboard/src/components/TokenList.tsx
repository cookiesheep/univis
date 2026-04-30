import { StepMessage } from '../types'

interface Props {
  steps: StepMessage[]
}

export default function TokenList({ steps }: Props) {
  return (
    <div style={{ marginTop: 16 }}>
      <h3 style={{ color: '#ccd6f6', marginBottom: 8 }}>Generated Tokens</h3>
      <div style={{
        background: '#16213e', borderRadius: 6, padding: 12,
        fontFamily: 'monospace', fontSize: 14, lineHeight: 1.8,
        maxHeight: 120, overflowY: 'auto',
      }}>
        {steps.map(s => (
          <span key={s.token_idx} title={`token ${s.token_idx}, entropy=${s.global.prediction_entropy.toFixed(2)}`}
            style={{
              display: 'inline-block', margin: '2px 3px', padding: '2px 6px',
              borderRadius: 3, background: '#0f3460', color: '#64ffda', cursor: 'default',
            }}>
            {s.generated_token}
          </span>
        ))}
      </div>
    </div>
  )
}
