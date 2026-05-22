import ModelSelector from './ModelSelector'

export default function SystemLogCard({
  logText, logColor, sensorStatus,
  selectedModel, onModelSelect,
  queueLength, onProcess
}) {
  return (
    <div className="nc-card fade-in fade-in-d2" style={{ justifyContent: 'space-between' }}>
      <div>
        <div style={{
          fontFamily: 'Orbitron', fontSize: 9, letterSpacing: '0.2em',
          color: 'rgba(0,242,255,0.55)', marginBottom: 18,
          display: 'flex', alignItems: 'center', justifyContent: 'space-between'
        }}>
          <span>02_SYSTEM_LOG</span>
          <span className="status-pill" style={{
            background: sensorStatus === 'ONLINE'
              ? 'rgba(74,222,128,0.12)' : sensorStatus === '2-HAND'
              ? 'rgba(245,158,11,0.12)' : 'rgba(239,68,68,0.12)',
            color: sensorStatus === 'ONLINE'
              ? '#4ade80' : sensorStatus === '2-HAND'
              ? '#fbbf24' : '#f87171',
            border: `1px solid ${sensorStatus === 'ONLINE'
              ? 'rgba(74,222,128,0.3)' : sensorStatus === '2-HAND'
              ? 'rgba(245,158,11,0.3)' : 'rgba(239,68,68,0.3)'}`
          }}>
            {sensorStatus}
          </span>
        </div>

        {/* Log box */}
        <div className="log-box">
          {[
            { k: 'mode',   v: 'STERILE_GESTURE',       cls: 'v-cyan'  },
            { k: 'reset',  v: 'GRAB×3 IN AIR = OFF',   cls: 'v-amber' },
            { k: 'select', v: 'PINCH or HOVER 1.4s',   cls: 'v-green' },
            { k: 'scroll', v: 'TWO HANDS · MODAL',     cls: 'v-green' },
            { k: 'zoom',   v: 'SCROLL IN VIEWER',      cls: 'v-cyan'  },
            { k: 'xai',    v: 'GRAD-CAM ENABLED',      cls: 'v-green' },
          ].map(row => (
            <div key={row.k}>
              <span style={{ color: 'rgba(0,242,255,0.7)' }}>&gt; {row.k}:</span>{' '}
              <span style={{
                color: row.cls === 'v-cyan'  ? 'var(--cyan)'
                     : row.cls === 'v-amber' ? '#fbbf24'
                     : '#4ade80'
              }}>{row.v}</span>
            </div>
          ))}
          <div>
            <span style={{ color: 'rgba(0,242,255,0.7)' }}>&gt; state:</span>{' '}
            <span style={{ color: logColor || 'rgba(100,140,180,0.55)' }}>
              {logText || 'IDLE'}
            </span>
          </div>
        </div>

        {/* Model selector */}
        <ModelSelector selected={selectedModel} onSelect={onModelSelect} />
      </div>

      {/* Process button */}
      {queueLength > 0 && (
        <button className="process-btn" onClick={onProcess}>
          ⬡ INITIALIZE NEURAL SWEEP
        </button>
      )}
    </div>
  )
}