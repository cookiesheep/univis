import { useState } from 'react'

interface Props {
  layerNames: string[]
  selectedLayers: Set<number>
  onToggle: (idx: number) => void
  onSelectAll: () => void
  onDeselectAll: () => void
}

export default function LayerFilter({ layerNames, selectedLayers, onToggle, onSelectAll, onDeselectAll }: Props) {
  const [open, setOpen] = useState(false)

  const allSelected = selectedLayers.size === layerNames.length
  const noneSelected = selectedLayers.size === 0

  return (
    <div style={{ marginBottom: 12 }}>
      <div
        onClick={() => setOpen(!open)}
        style={{
          display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', userSelect: 'none',
          color: '#ccd6f6', fontSize: 14, fontWeight: 600,
        }}
      >
        <span style={{ color: '#64ffda', fontSize: 11 }}>{open ? '▼' : '▶'}</span>
        Layer Filter
        {!allSelected && (
          <span style={{ color: '#8892b0', fontWeight: 'normal', fontSize: 12 }}>
            ({selectedLayers.size}/{layerNames.length} selected)
          </span>
        )}
      </div>

      {open && (
        <div style={{
          marginTop: 6, background: '#16213e', borderRadius: 6, padding: '8px 12px',
          maxHeight: 200, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 2,
        }}>
          <div style={{ display: 'flex', gap: 8, marginBottom: 4 }}>
            <button
              onClick={onSelectAll}
              disabled={allSelected}
              style={{
                padding: '2px 8px', borderRadius: 3, border: '1px solid #333',
                background: '#0f3460', color: allSelected ? '#555' : '#ccd6f6',
                cursor: allSelected ? 'default' : 'pointer', fontSize: 12,
              }}
            >All</button>
            <button
              onClick={onDeselectAll}
              disabled={noneSelected}
              style={{
                padding: '2px 8px', borderRadius: 3, border: '1px solid #333',
                background: '#0f3460', color: noneSelected ? '#555' : '#ccd6f6',
                cursor: noneSelected ? 'default' : 'pointer', fontSize: 12,
              }}
            >None</button>
          </div>
          {layerNames.map((name, idx) => (
            <label key={idx} style={{
              display: 'flex', alignItems: 'center', gap: 6,
              color: selectedLayers.has(idx) ? '#ccd6f6' : '#555',
              fontSize: 12, cursor: 'pointer',
            }}>
              <input
                type="checkbox"
                checked={selectedLayers.has(idx)}
                onChange={() => onToggle(idx)}
                style={{ accentColor: '#64ffda', cursor: 'pointer' }}
              />
              {name}
            </label>
          ))}
        </div>
      )}
    </div>
  )
}
