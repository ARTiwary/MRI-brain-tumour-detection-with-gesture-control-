import { useEffect, useRef } from 'react'

export default function ParticleBackground({ visible }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    const c   = canvasRef.current
    const ctx = c.getContext('2d')
    let W, H, pts = [], raf

    const resize = () => {
      W = c.width  = innerWidth
      H = c.height = innerHeight
      pts = []
      const cols = Math.ceil(W / 90), rows = Math.ceil(H / 90)
      for (let r = 0; r <= rows; r++)
        for (let col = 0; col <= cols; col++)
          pts.push({
            x:  col * (W/cols) + (Math.random()-0.5)*18,
            y:  r   * (H/rows) + (Math.random()-0.5)*18,
            vx: (Math.random()-0.5)*0.15,
            vy: (Math.random()-0.5)*0.15,
            a:  Math.random()*0.25 + 0.05
          })
    }

    const draw = () => {
      ctx.clearRect(0, 0, W, H)
      for (let i = 0; i < pts.length; i++) {
        const p = pts[i]
        p.x += p.vx; p.y += p.vy
        if (p.x < -20 || p.x > W+20) p.vx *= -1
        if (p.y < -20 || p.y > H+20) p.vy *= -1
        for (let j = i+1; j < pts.length; j++) {
          const q = pts[j]
          const d = Math.hypot(p.x-q.x, p.y-q.y)
          if (d < 120) {
            ctx.strokeStyle = `rgba(0,180,216,${(1-d/120)*0.07})`
            ctx.lineWidth   = 0.5
            ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); ctx.stroke()
          }
        }
        ctx.beginPath(); ctx.arc(p.x, p.y, 1.2, 0, Math.PI*2)
        ctx.fillStyle = `rgba(0,180,216,${p.a})`; ctx.fill()
      }
      raf = requestAnimationFrame(draw)
    }

    window.addEventListener('resize', resize)
    resize(); draw()
    return () => { window.removeEventListener('resize', resize); cancelAnimationFrame(raf) }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'fixed', inset: 0, zIndex: 0,
        pointerEvents: 'none',
        opacity: visible ? 1 : 0.2,
        transition: 'opacity 0.8s ease'
      }}
    />
  )
}