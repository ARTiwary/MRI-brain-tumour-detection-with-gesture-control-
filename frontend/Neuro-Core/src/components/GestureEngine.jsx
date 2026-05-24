import { useEffect, useRef, useCallback } from 'react'

const ALPHA       = 0.16
const PINCH_CLOSE = 0.047
const PINCH_OPEN  = 0.085
const DWELL       = 1400
const SCROLL_MULT = 5.5
const FRICTION    = 0.88

function ema(prev, next, a) { return prev < 0 ? next : prev + a * (next - prev) }

const CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20]
]

export default function GestureEngine({
  gestureOn,
  onHandMove,
  onSensorStatus,
  onLogUpdate,
  onAirGrab,
  modalOpen,
  onModalScroll,
  cursorRef,
  cursor2Ref,
  tooltipRef,
}) {
  const videoRef   = useRef(null)
  const skelRef    = useRef(null)
  const stateRef   = useRef({
    smX: -1, smY: -1, sm2X: -1, sm2Y: -1,
    isPinching: false, pinchConfirm: 0, lastAction: 0,
    dwellEl: null, dwellStart: 0,
    airGrabCount: 0, grabWasOnFile: false, grabResetTimer: null,
    lastScrollY: null, scrollVel: 0, scrollRafId: null
  })
  const inited = useRef(false)

  // Draw skeleton
  const drawSkeleton = useCallback((landmarks) => {
    const canvas = skelRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    if (!landmarks) return

    CONNECTIONS.forEach(([a, b]) => {
      const pa = landmarks[a], pb = landmarks[b]
      ctx.strokeStyle = 'rgba(0,242,255,0.18)'
      ctx.lineWidth   = 1
      ctx.beginPath()
      ctx.moveTo((1 - pa.x) * innerWidth, pa.y * innerHeight)
      ctx.lineTo((1 - pb.x) * innerWidth, pb.y * innerHeight)
      ctx.stroke()
    })
    landmarks.forEach((lm, i) => {
      const x = (1 - lm.x) * innerWidth
      const y = lm.y * innerHeight
      ctx.beginPath()
      ctx.arc(x, y, i === 8 || i === 4 ? 4 : 2, 0, Math.PI * 2)
      ctx.fillStyle = i === 8 || i === 4 ? 'rgba(0,242,255,0.7)' : 'rgba(0,242,255,0.3)'
      ctx.fill()
    })
  }, [])

  // Momentum scroll
  const momentumScroll = useCallback(() => {
    const s = stateRef.current
    if (Math.abs(s.scrollVel) < 0.4) { s.scrollVel = 0; return }
    const modal = document.querySelector('[data-result-modal]')
    if (modal) modal.scrollTop += s.scrollVel
    s.scrollVel *= FRICTION
    s.scrollRafId = requestAnimationFrame(momentumScroll)
  }, [])

  useEffect(() => {
    // Resize skeleton canvas
    const resizeSkel = () => {
      if (skelRef.current) {
        skelRef.current.width  = innerWidth
        skelRef.current.height = innerHeight
      }
    }
    resizeSkel()
    window.addEventListener('resize', resizeSkel)
    return () => window.removeEventListener('resize', resizeSkel)
  }, [])

  useEffect(() => {
    if (!gestureOn || inited.current) return
    if (!window.Hands || !window.Camera) return
    inited.current = true

    const hands = new window.Hands({
      locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}`
    })
    hands.setOptions({
      maxNumHands: 2,
      modelComplexity: 1,
      minDetectionConfidence: 0.75,
      minTrackingConfidence: 0.75
    })

    hands.onResults(results => {
      if (!gestureOn) {
        if (skelRef.current) {
          skelRef.current.getContext('2d').clearRect(0, 0, innerWidth, innerHeight)
        }
        return
      }

      const allHands = results.multiHandLandmarks || []
      const s        = stateRef.current

      // Update sphere
      if (allHands.length > 0 && onHandMove) {
        onHandMove(1 - allHands[0][8].x, allHands[0][8].y)
      }

      drawSkeleton(allHands[0] || null)

      // Sensor status
      if (onSensorStatus) {
        onSensorStatus(allHands.length === 0 ? 'OFFLINE'
          : allHands.length === 2 ? '2-HAND' : 'ONLINE')
      }

      // Two-finger scroll in modal
      if (allHands.length === 2 && modalOpen) {
        cancelAnimationFrame(s.scrollRafId)
        const h0  = allHands[0], h1 = allHands[1]
        const r0y = h0[8].y * innerHeight
        const r1y = h1[8].y * innerHeight
        s.smY  = ema(s.smY,  r0y, ALPHA)
        s.sm2Y = ema(s.sm2Y, r1y, ALPHA)

        if (cursor2Ref?.current) {
          const r0x = (1 - h0[8].x) * innerWidth
          const r1x = (1 - h1[8].x) * innerWidth
          s.smX  = ema(s.smX,  r0x, ALPHA)
          s.sm2X = ema(s.sm2X, r1x, ALPHA)
          cursor2Ref.current.style.opacity    = '1'
          cursor2Ref.current.style.transform  = `translate3d(${s.sm2X - 10}px,${s.sm2Y - 10}px,0)`
          if (cursorRef?.current) {
            cursorRef.current.style.opacity   = '1'
            cursorRef.current.style.transform = `translate3d(${s.smX - 14}px,${s.smY - 14}px,0)`
          }
        }

        const avgY = (s.smY + s.sm2Y) / 2
        if (s.lastScrollY !== null) {
          const delta = (avgY - s.lastScrollY) * SCROLL_MULT
          if (Math.abs(delta) > 0.8) {
            s.scrollVel = delta
            if (onModalScroll) onModalScroll(delta)
            if (onLogUpdate) onLogUpdate(delta > 0 ? 'SCROLL ↓' : 'SCROLL ↑', '#fbbf24')
          }
        }
        s.lastScrollY = avgY
        return
      } else {
        if (s.lastScrollY !== null && Math.abs(s.scrollVel) > 1) {
          s.scrollRafId = requestAnimationFrame(momentumScroll)
        }
        s.lastScrollY = null
        s.sm2X = -1; s.sm2Y = -1
        if (cursor2Ref?.current) cursor2Ref.current.style.opacity = '0'
      }

      // No hands
      if (!allHands.length) {
        if (cursorRef?.current) cursorRef.current.style.opacity = '0'
        if (onLogUpdate) onLogUpdate('NO HAND', 'rgba(100,140,180,0.55)')
        s.dwellEl = null
        s.smX = -1; s.smY = -1
        return
      }

      const hand = allHands[0]
      const rawX = (1 - hand[8].x) * innerWidth
      const rawY = hand[8].y * innerHeight
      s.smX = ema(s.smX, rawX, ALPHA)
      s.smY = ema(s.smY, rawY, ALPHA)
      const x = s.smX, y = s.smY

      if (cursorRef?.current) {
        cursorRef.current.style.opacity   = '1'
        cursorRef.current.style.transform = `translate3d(${x - 14}px,${y - 14}px,0)`
      }
      if (tooltipRef?.current) {
        tooltipRef.current.style.left = `${x + 22}px`
        tooltipRef.current.style.top  = `${y - 14}px`
      }

      const pinch        = Math.hypot(hand[8].x - hand[4].x, hand[8].y - hand[4].y)
      const isPinchClosed = pinch < PINCH_CLOSE
      const isPinchOpen   = pinch > PINCH_OPEN

      // Palm flash
      const palm = hand[8].y < hand[6].y && hand[12].y < hand[10].y &&
                   hand[16].y < hand[14].y && hand[20].y < hand[18].y
      if (palm && Date.now() - s.lastAction > 3000) {
        s.lastAction = Date.now()
        if (onLogUpdate) onLogUpdate('PALM FLASH', '#fbbf24')
        if (cursorRef?.current) {
          cursorRef.current.style.boxShadow = '0 0 40px #ffcc00'
          setTimeout(() => { if (cursorRef.current) cursorRef.current.style.boxShadow = '0 0 12px var(--cyan)' }, 600)
        }
      }

      // Pinch closed
      if (isPinchClosed) {
        if (cursorRef?.current) {
          cursorRef.current.style.borderColor = '#bc13fe'
          cursorRef.current.style.boxShadow   = '0 0 22px #bc13fe'
        }
        if (onLogUpdate) onLogUpdate('PINCH', '#bc13fe')
        s.pinchConfirm++

        if (s.pinchConfirm >= 2 && !s.isPinching && Date.now() - s.lastAction > 640) {
          s.isPinching  = true
          s.lastAction  = Date.now()
          s.dwellEl     = null

          const el = document.elementFromPoint(x, y)
          if (el) {
            const cl = el.closest('.g-clickable, button, [onclick]')
            if (cl) {
              cl.click()
              if (onLogUpdate) onLogUpdate('PINCH SELECT', '#4ade80')
              s.grabWasOnFile = true
            } else {
              s.grabWasOnFile = false
            }
          } else {
            s.grabWasOnFile = false
          }
        }
      } else {
        s.pinchConfirm = 0
        if (isPinchOpen) {
          // Air grab counter
          if (s.isPinching && !s.grabWasOnFile) {
            s.airGrabCount++
            if (onAirGrab) onAirGrab(s.airGrabCount)
            if (onLogUpdate) onLogUpdate(`AIR GRAB ${s.airGrabCount}/3`, '#f59e0b')

            if (s.airGrabCount >= 3) {
              s.airGrabCount = 0
              if (onAirGrab) onAirGrab(0)
              if (onLogUpdate) onLogUpdate('GESTURE OFF', 'rgba(100,140,180,0.55)')
              setTimeout(() => {
                const btn = document.getElementById('gesture-toggle-btn')
                if (btn) btn.click()
              }, 400)
            }
          }
          s.isPinching = false
          if (cursorRef?.current) {
            cursorRef.current.style.borderColor = 'var(--cyan)'
            cursorRef.current.style.boxShadow   = '0 0 12px var(--cyan)'
          }
          clearTimeout(s.grabResetTimer)
          if (s.airGrabCount > 0 && s.airGrabCount < 3) {
            s.grabResetTimer = setTimeout(() => {
              s.airGrabCount = 0
              if (onAirGrab) onAirGrab(0)
            }, 2000)
          }
        }
      }

      // Dwell hover
      if (isPinchOpen) {
        const el     = document.elementFromPoint(x, y)
        const target = el?.closest('.g-clickable')
        document.querySelectorAll('.g-hover').forEach(e => e.classList.remove('g-hover'))

        if (target) {
          target.classList.add('g-hover')
          if (onLogUpdate) onLogUpdate('HOVER', 'var(--cyan)')
          if (s.dwellEl !== target) { s.dwellEl = target; s.dwellStart = Date.now() }
          else {
            const elapsed = Date.now() - s.dwellStart
            const prog    = Math.min(elapsed / DWELL, 1)
            const ring    = target.querySelector('.d-ring')
            if (ring) {
              ring.classList.add('active')
              const c = ring.querySelector('circle')
              if (c) c.style.strokeDashoffset = `${44 - 44 * prog}`
            }
            if (elapsed >= DWELL && Date.now() - s.lastAction > DWELL + 50) {
              s.lastAction = Date.now()
              s.dwellEl    = null
              target.click()
              if (onLogUpdate) onLogUpdate('DWELL CLICK', '#4ade80')
            }
          }
        } else {
          s.dwellEl = null
          document.querySelectorAll('.d-ring').forEach(r => {
            r.classList.remove('active')
            const c = r.querySelector('circle')
            if (c) c.style.strokeDashoffset = '44'
          })
          if (onLogUpdate) onLogUpdate('IDLE', 'rgba(100,140,180,0.55)')
        }
      }
    })

    const video = videoRef.current
    if (video && window.Camera) {
      new window.Camera(video, {
        onFrame: async () => await hands.send({ image: video }),
        width: 640, height: 480
      }).start()
    }
  }, [gestureOn, modalOpen, drawSkeleton, momentumScroll,
      onHandMove, onSensorStatus, onLogUpdate, onAirGrab,
      onModalScroll, cursorRef, cursor2Ref, tooltipRef])

  return (
    <>
      <video id="gesture-video" ref={videoRef}
        style={{ display: 'none' }} autoPlay playsInline />
      <canvas ref={skelRef}
        style={{
          position: 'fixed', inset: 0, zIndex: 2,
          pointerEvents: 'none',
          opacity: gestureOn ? 1 : 0,
          transition: 'opacity 0.5s'
        }}
      />
    </>
  )
}