import { useState, useEffect } from 'react'

const MODEL_INFO = {
  cnn: {
    label   : '⬡ CUSTOM CNN — 92.00% TEST ACC',
    arch    : '4-block CNN + BatchNorm',
    params  : '~8.8M trainable',
    val_acc : '96.70%',
  },
  resnet18: {
    label   : '⬡ RESNET18 PRETRAINED — 94.81% TEST ACC',
    arch    : 'ResNet18 pretrained backbone',
    params  : '~11.7M trainable',
    val_acc : '97.32%',
  }
}

export default function ModelSelector({ selected, onSelect }) {
  const [open, setOpen] = useState(false)
  const info = MODEL_INFO[selected]

  useEffect(() => {
    const close = e => {
      if (!e.target.closest('#model-dropdown-wrap')) setOpen(false)
    }
    document.addEventListener('click', close)
    return () => document.removeEventListener('click', close)
  }, [])

  const handleSelect = (key) => {
    onSelect(key)
    setOpen(false)
  }

  return (
    <div style={{ marginTop: 16 }}>
      <div style={{
        fontFamily: 'Orbitron', fontSize: 9, letterSpacing: '0.18em',
        color: 'rgba(0,242,255,0.55)', marginBottom: 8
      }}>
        04_MODEL_SELECT
      </div>

      <div id="model-dropdown-wrap" style={{ position: 'relative' }}>
        {/* Trigger */}
        <button
          className="g-clickable"
          onClick={() => setOpen(o => !o)}
          style={{
            width: '100%', padding: '10px 36px 10px 14px',
            background: 'rgba(6,14,32,0.9)',
            border: open ? '1px solid var(--cyan)' : '1px solid rgba(0,242,255,0.28)',
            borderRadius: 10,
            color: 'rgba(0,242,255,0.85)',
            fontFamily: 'Orbitron', fontSize: 9, letterSpacing: '0.12em',
            cursor: 'pointer', outline: 'none', textAlign: 'left',
            transition: 'all 0.2s', position: 'relative',
            boxShadow: open ? '0 0 16px rgba(0,242,255,0.2)' : 'none'
          }}
        >
          <span id="model-trigger-label">{info.label}</span>
          <span style={{
            position: 'absolute', right: 12, top: '50%',
            transform: 'translateY(-50%)',
            color: 'rgba(0,242,255,0.5)', fontSize: 10,
            transition: 'transform 0.2s',
            display: 'inline-block',
            rotate: open ? '180deg' : '0deg'
          }}>▾</span>
        </button>

        {/* Dropdown */}
        {open && (
          <div style={{
            position: 'absolute', top: 'calc(100% + 6px)',
            left: 0, right: 0,
            background: 'rgba(3,7,18,0.97)',
            border: '1px solid rgba(0,242,255,0.35)',
            borderRadius: 10, overflow: 'hidden',
            zIndex: 600,
            boxShadow: '0 8px 32px rgba(0,0,0,0.6)',
            animation: 'fadeIn 0.18s ease'
          }}>
            {Object.entries(MODEL_INFO).map(([key, m], idx) => (
              <button
                key={key}
                className={`model-opt g-clickable${selected === key ? ' is-active' : ''}`}
                data-key={key}
                onClick={() => handleSelect(key)}
                style={{
                  borderBottom: idx === 0 ? '1px solid rgba(0,242,255,0.1)' : 'none',
                  background: selected === key ? 'rgba(0,242,255,0.1)' : 'transparent'
                }}
              >
                <div>{m.label.split('—')[0].trim()}</div>
                <div style={{
                  fontSize: 8, color: 'rgba(0,242,255,0.45)',
                  marginTop: 3, fontFamily: 'Space Mono'
                }}>
                  Test: {key === 'cnn' ? '92.00%' : '94.81%'} · Val: {m.val_acc} · {m.params}
                </div>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Info badge */}
      <div id="model-badge" style={{
        marginTop: 8, padding: '7px 12px',
        background: 'rgba(0,242,255,0.05)',
        border: '1px solid rgba(0,242,255,0.15)',
        borderRadius: 8,
        fontFamily: 'Space Mono', fontSize: 9,
        color: 'rgba(100,140,180,0.65)', lineHeight: 1.7
      }}>
        <span style={{ color: 'rgba(0,242,255,0.5)' }}>&gt; arch:</span> {info.arch}<br />
        <span style={{ color: 'rgba(0,242,255,0.5)' }}>&gt; params:</span> {info.params}<br />
        <span style={{ color: 'rgba(0,242,255,0.5)' }}>&gt; val_acc:</span>{' '}
        <span style={{ color: '#4ade80' }}>{info.val_acc}</span>
      </div>
    </div>
  )
}