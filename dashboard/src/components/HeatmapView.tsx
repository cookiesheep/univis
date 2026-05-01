import { useMemo, useRef, useEffect } from 'react'
import ReactECharts from 'echarts-for-react'
import type { EChartsOption } from 'echarts'
import { StepMessage } from '../types'

interface Props {
  steps: StepMessage[]
  numLayers: number
  layerNames: string[]
  highlightedTokenIndex: number | null
}

export default function HeatmapView({ steps, numLayers, layerNames, highlightedTokenIndex }: Props) {
  const chartRef = useRef<ReactECharts>(null)

  const option = useMemo(() => {
    // Build heatmap data: [tokenIdx, layerIdx, relative_delta]
    const data: [number, number, number][] = []
    let maxDelta = 0

    // Lookup map for tooltip: "tokenIdx,layerIdx" -> full metrics
    const metricsMap = new Map<string, {
      generatedToken: string
      relativeDelta: number
      cosineSim: number
      sparsity: number
    }>()

    for (const step of steps) {
      for (const layer of step.layers) {
        const val = layer.relative_delta
        data.push([step.token_idx, layer.idx, val])
        if (val > maxDelta) maxDelta = val
        metricsMap.set(`${step.token_idx},${layer.idx}`, {
          generatedToken: step.generated_token,
          relativeDelta: layer.relative_delta,
          cosineSim: layer.cosine_sim,
          sparsity: layer.sparsity,
        })
      }
    }

    const tokenLabels = steps.map(s => `${s.token_idx}`)
    const layerLabels = layerNames.length > 0
      ? layerNames
      : Array.from({ length: numLayers }, (_, i) => `Layer ${i}`)

    // Build markLine for highlighted token column
    const markLine = highlightedTokenIndex !== null
      ? {
          silent: true,
          symbol: 'none',
          label: { show: false },
          lineStyle: { color: '#64ffda', width: 2, type: 'solid' as const },
          data: [{ xAxis: highlightedTokenIndex.toString() }],
        }
      : undefined

    return {
      tooltip: {
        position: 'top' as const,
        backgroundColor: 'rgba(26,26,46,0.95)',
        borderColor: '#64ffda',
        borderWidth: 1,
        textStyle: { color: '#ccd6f6', fontFamily: 'DM Sans, system-ui, sans-serif' },
        formatter: (params: any) => {
          const d = params.data as [number, number, number]
          const key = `${d[0]},${d[1]}`
          const m = metricsMap.get(key)
          if (!m) {
            return `<b>${layerLabels[d[1]]}</b> — Token ${d[0]}<br/>RelDelta: ${d[2].toFixed(4)}`
          }
          return [
            `<b>${layerLabels[d[1]]}</b>`,
            `<span style="color:#8892b0">Token index:</span> ${d[0]}`,
            `<span style="color:#8892b0">Generated:</span> "${m.generatedToken}"`,
            `<table style="margin-top:4px;border-collapse:collapse;font-size:12px">`,
            `<tr><td style="color:#8892b0;padding-right:12px">RelDelta</td><td style="text-align:right;font-family:'JetBrains Mono',monospace">${m.relativeDelta.toFixed(4)}</td></tr>`,
            `<tr><td style="color:#8892b0;padding-right:12px">Cosine Sim</td><td style="text-align:right;font-family:'JetBrains Mono',monospace">${m.cosineSim.toFixed(4)}</td></tr>`,
            `<tr><td style="color:#8892b0;padding-right:12px">Sparsity</td><td style="text-align:right;font-family:'JetBrains Mono',monospace">${m.sparsity.toFixed(4)}</td></tr>`,
            `</table>`,
          ].join('')
        },
      },
      grid: { height: '60%', top: '12%', left: '12%', bottom: '12%' },
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
        bottom: '8%',
        inRange: {
          color: ['#0d0887', '#4903a0', '#7d03a8', '#b93289', '#db5c68', '#f48849', '#febd2a', '#f0f921'],
        },
        textStyle: { color: '#8892b0' },
      },
      dataZoom: [
        { type: 'inside', xAxisIndex: 0, filterMode: 'none' },
        {
          type: 'slider',
          xAxisIndex: 0,
          bottom: '0%',
          height: 20,
          borderColor: 'transparent',
          backgroundColor: '#1a1a2e',
          fillerColor: 'rgba(100,255,218,0.15)',
          handleStyle: { color: '#64ffda' },
          textStyle: { color: '#8892b0' },
        },
      ],
      series: [{
        type: 'heatmap',
        data,
        progressive: 500,
        animation: true,
        animationDuration: 300,
        animationEasing: 'cubicOut',
        itemStyle: { borderColor: '#1a1a2e', borderWidth: 1 },
        emphasis: { itemStyle: { borderColor: '#64ffda', borderWidth: 2 } },
        markLine,
      }],
    }
  }, [steps, numLayers, layerNames, highlightedTokenIndex])

  // Auto-scroll to latest token when new data arrives
  useEffect(() => {
    if (steps.length === 0) return
    const chartInstance = chartRef.current?.getEchartsInstance()
    if (!chartInstance) return
    const tokenLabels = steps.map(s => `${s.token_idx}`)
    chartInstance.dispatchAction({
      type: 'dataZoom',
      dataZoomIndex: 0,
      start: Math.max(0, 100 - (100 / tokenLabels.length)),
      end: 100,
    })
  }, [steps])

  return (
    <div>
      <h3 style={{ color: '#ccd6f6', marginBottom: 8 }}>
        Layer Redundancy Heatmap
        <span style={{ color: '#8892b0', fontWeight: 'normal', fontSize: 13, marginLeft: 8 }}>
          (plasma colormap — dark = redundant, bright = active)
        </span>
      </h3>
      <ReactECharts ref={chartRef} option={option} style={{ height: 400, width: '100%' }} />
    </div>
  )
}
