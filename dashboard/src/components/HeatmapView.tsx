import { useMemo } from 'react'
import ReactECharts from 'echarts-for-react'
import { StepMessage } from '../types'

interface Props {
  steps: StepMessage[]
  numLayers: number
  layerNames: string[]
}

export default function HeatmapView({ steps, numLayers, layerNames }: Props) {
  const option = useMemo(() => {
    // Build heatmap data: [tokenIdx, layerIdx, relative_delta]
    const data: [number, number, number][] = []
    let maxDelta = 0

    for (const step of steps) {
      for (const layer of step.layers) {
        const val = layer.relative_delta
        data.push([step.token_idx, layer.idx, val])
        if (val > maxDelta) maxDelta = val
      }
    }

    const tokenLabels = steps.map(s => `${s.token_idx}`)
    const layerLabels = layerNames.length > 0
      ? layerNames
      : Array.from({ length: numLayers }, (_, i) => `Layer ${i}`)

    return {
      tooltip: {
        position: 'top' as const,
        formatter: (params: any) => {
          const d = params.data as [number, number, number]
          return `Token ${d[0]}, ${layerLabels[d[1]]}<br/>RelDelta: ${d[2].toFixed(4)}`
        },
      },
      grid: { height: '70%', top: '12%', left: '12%' },
      xAxis: {
        type: 'category' as const,
        data: tokenLabels,
        name: 'Token',
        splitArea: { show: true },
        axisLabel: { color: '#8892b0' },
        nameTextStyle: { color: '#8892b0' },
      },
      yAxis: {
        type: 'category' as const,
        data: layerLabels,
        name: 'Layer',
        axisLabel: { color: '#8892b0', fontSize: 11 },
        nameTextStyle: { color: '#8892b0' },
      },
      visualMap: {
        min: 0,
        max: Math.max(maxDelta * 1.1, 0.01),
        calculable: true,
        orient: 'horizontal' as const,
        left: 'center',
        bottom: '2%',
        inRange: {
          color: ['#e8e8e8', '#ffe0b2', '#ffb74d', '#ff7043', '#e53935', '#b71c1c'],
        },
        textStyle: { color: '#8892b0' },
      },
      series: [{
        type: 'heatmap',
        data,
        progressive: 500,
        animation: false,
        itemStyle: { borderColor: '#1a1a2e', borderWidth: 1 },
        emphasis: { itemStyle: { borderColor: '#64ffda', borderWidth: 2 } },
      }],
    }
  }, [steps, numLayers, layerNames])

  return (
    <div>
      <h3 style={{ color: '#ccd6f6', marginBottom: 8 }}>
        Layer Redundancy Heatmap
        <span style={{ color: '#8892b0', fontWeight: 'normal', fontSize: 13, marginLeft: 8 }}>
          (white = redundant, red = active)
        </span>
      </h3>
      <ReactECharts option={option} style={{ height: 400, width: '100%' }} />
    </div>
  )
}
