import { useEffect, useRef, useState } from 'react'

export default function ViewerOverlay({ src, onClose }) {
  const canvasRef = useRef(null)
  const imgRef    = useRef(null)
  const zoomRef   = useRef(1)
  const rafRef    = useRef(null)
  const timeRef   = useRef(0)
  const [zoom, setZoom] = useState(1)

  useEffect(() => {
    if (!src || !canvasRef.current) return
    const canvas = canvasRef.current
    const ctx    = canvas.getContext('2d')

    const img = new Image()
    img.onload = () => {
      imgRef.current = img
      const maxW = Math.min(innerWidth * 0.85, 900)
      const maxH = Math.min(innerHeight * 0.82, 700)
      const ar   = img.naturalWidth / img.naturalHeight
      let cw, ch
      if (ar > maxW / maxH) { cw = maxW; ch = maxW / ar }
      else { ch = maxH; cw = maxH * ar }
      canvas.width  = Math.floor(cw)
      canvas.height = Math.floor(ch)
      startRender(ctx, img, canvas)
    }
    img.src = src

    return () => cancelAnimationFrame(rafRef.current)
  }, [src])

  const startRender = (ctx, img, canvas) => {
    const W = canvas.width, H = canvas.height

    const render = () => {
      timeRef.current += 0.016
      const t = timeRef.current
      const z = zoomRef.current

      ctx.clearRect(0, 0, W, H)

      // ── Draw image at zoom ──
      const sw = W / z, sh = H / z
      const sx = (img.naturalWidth  / 2) - sw / 2
      const sy = (img.naturalHeight / 2) - sh / 2
      ctx.drawImage(img,
        Math.max(0, sx), Math.max(0, sy),
        Math.min(sw, img.naturalWidth), Math.min(sh, img.naturalHeight),
        0, 0, W, H
      )

      // ── Holographic scan line ──
      const scanY = ((Math.sin(t * 0.8) + 1) / 2) * H
      const grad  = ctx.createLinearGradient(0, scanY - 20, 0, scanY + 20)
      grad.addColorStop(0,   'rgba(0,242,255,0)')
      grad.addColorStop(0.5, 'rgba(0,242,255,0.12)')
      grad.addColorStop(1,   'rgba(0,242,255,0)')
      ctx.fillStyle = grad
      ctx.fillRect(0, scanY - 20, W, 40)

      // ── Corner brackets ──
      const bSize = 24, bWidth = 2
      ctx.strokeStyle = 'rgba(0,242,255,0.7)'
      ctx.lineWidth   = bWidth
      ;[
        [0, 0, bSize, 0, 0, bSize],
        [W, 0, W - bSize, 0, W, bSize],
        [0, H, bSize, H, 0, H - bSize],
        [W, H, W - bSize, H, W, H - bSize],
      ].forEach(([x1,y1,x2,y2,x3,y3]) => {
        ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke()
        ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x3,y3); ctx.stroke()
      })

      // ── Pulsing grid overlay ──
      const gridAlpha = 0.03 + Math.sin(t * 2) * 0.01
      ctx.strokeStyle = `rgba(0,242,255,${gridAlpha})`
      ctx.lineWidth   = 0.5
      const gStep = 32
      for (let x = 0; x < W; x += gStep) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke()
      }
      for (let y = 0; y < H; y += gStep) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke()
      }

      // ── Vignette ──
      const vig = ctx.createRadialGradient(W/2, H/2, W*0.28, W/2, H/2, W*0.7)
      vig.addColorStop(0, 'rgba(0,0,0,0)')
      vig.addColorStop(1, 'rgba(0,10,30,0.55)')
      ctx.fillStyle = vig
      ctx.fillRect(0, 0, W, H)

      rafRef.current = requestAnimationFrame(render)
    }

    rafRef.current = requestAnimationFrame(render)
  }

  const handleWheel = (e) => {
    e.preventDefault()
    const delta = e.deltaY > 0 ? -0.1 : 0.1
    zoomRef.current = Math.max(0.5, Math.min(6, zoomRef.current + delta))
    setZoom(zoomRef.current)
  }

  const handleClick = (e) => {
    // Ripple effect on click
    const r = canvasRef.current?.getBoundingClientRect()
    if (!r) return
  }

  if (!src) return null

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 1200,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      background: 'rgba(2,4,12,0.97)',
      backdropFilter: 'blur(40px)',
      animation: 'fadeIn 0.3s ease'
    }}>
      <div style={{ position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center', width: '100%', height: '100%' }}>

        {/* Canvas */}
        <canvas
          ref={canvasRef}
          onWheel={handleWheel}
          onClick={handleClick}
          style={{ display: 'block', borderRadius: 18, cursor: 'crosshair' }}
        />

        {/* Close button */}
        <button onClick={onClose} style={{
          position: 'absolute', top: 22, right: 28,
          width: 44, height: 44, borderRadius: '50%',
          background: 'rgba(0,242,255,0.08)',
          border: '1px solid rgba(0,242,255,0.3)',
          color: 'rgba(0,242,255,0.8)', fontSize: 16,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          cursor: 'pointer', zIndex: 1201, transition: 'all 0.2s'
        }}
          onMouseEnter={e => e.currentTarget.style.background = 'rgba(0,242,255,0.18)'}
          onMouseLeave={e => e.currentTarget.style.background = 'rgba(0,242,255,0.08)'}
        >
          <i className="fas fa-times" />
        </button>

        {/* Zoom badge */}
        <div style={{
          position: 'absolute', top: 22, left: '50%', transform: 'translateX(-50%)',
          fontFamily: 'Orbitron', fontSize: 11, letterSpacing: '0.15em',
          color: 'rgba(0,242,255,0.6)', background: 'rgba(3,7,18,0.8)',
          border: '1px solid rgba(0,242,255,0.2)', borderRadius: 20,
          padding: '5px 18px', pointerEvents: 'none'
        }}>
          ZOOM {zoom.toFixed(1)}×
        </div>

        {/* Hints */}
        <div style={{
          position: 'absolute', bottom: 32,
          left: '50%', transform: 'translateX(-50%)',
          fontFamily: 'Orbitron', fontSize: 9, letterSpacing: '0.2em',
          color: 'rgba(0,242,255,0.45)', textAlign: 'center',
          pointerEvents: 'none', display: 'flex', gap: 28
        }}>
          {[
            { icon: '🖱️', label: 'SCROLL = ZOOM' },
            { icon: '🤏', label: 'PINCH = ZOOM (GESTURE)' },
            { icon: '✕',  label: 'CLICK × = CLOSE' },
          ].map(h => (
            <div key={h.label} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
              <div style={{ fontSize: 18, color: 'rgba(0,242,255,0.6)' }}>{h.icon}</div>
              <div>{h.label}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}