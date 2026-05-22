import { useRef, useCallback } from 'react'

const ALPHA      = 0.16
const ALPHA_FAST = 0.32
const PINCH_CLOSE = 0.047
const PINCH_OPEN  = 0.085
const DWELL       = 1400

function ema(prev, next, a) { return prev < 0 ? next : prev + a * (next - prev) }

export function useGesture({
  onHandMove,
  onPinch,
  onRelease,
  onTwoHand,
  gestureOn
}) {
  const state = useRef({
    smX: -1, smY: -1, sm2X: -1, sm2Y: -1,
    isPinching: false, pinchConfirm: 0, lastAction: 0,
    dwellEl: null, dwellStart: 0,
    airGrabCount: 0, grabWasOnFile: false, grabResetTimer: null,
    lastScrollY: null
  })

  const initEngine = useCallback(() => {
    if (!window.Hands) return
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
      if (!gestureOn.current) return
      const allHands = results.multiHandLandmarks || []
      const s = state.current

      // Update sphere hand position
      if (allHands.length > 0 && onHandMove) {
        onHandMove(1 - allHands[0][8].x, allHands[0][8].y)
      }

      // Two hand scroll
      if (allHands.length === 2 && onTwoHand) {
        const h0 = allHands[0], h1 = allHands[1]
        s.sm2X = ema(s.sm2X, (1 - h1[8].x) * innerWidth, ALPHA)
        s.sm2Y = ema(s.sm2Y, h1[8].y * innerHeight, ALPHA)
        const avgY = (s.smY + s.sm2Y) / 2
        if (s.lastScrollY !== null) {
          const delta = (avgY - s.lastScrollY) * 5.5
          if (Math.abs(delta) > 0.8) onTwoHand(delta)
        }
        s.lastScrollY = avgY
        return
      } else {
        s.lastScrollY = null
        s.sm2X = -1; s.sm2Y = -1
      }

      if (!allHands.length) {
        s.smX = -1; s.smY = -1
        return
      }

      const hand = allHands[0]
      const rawX = (1 - hand[8].x) * innerWidth
      const rawY = hand[8].y * innerHeight
      s.smX = ema(s.smX, rawX, ALPHA)
      s.smY = ema(s.smY, rawY, ALPHA)

      const pinch = Math.hypot(hand[8].x - hand[4].x, hand[8].y - hand[4].y)
      const isPinchClosed = pinch < PINCH_CLOSE
      const isPinchOpen   = pinch > PINCH_OPEN

      if (onPinch) onPinch({ x: s.smX, y: s.smY, isPinchClosed, isPinchOpen, state: s })
      if (isPinchOpen && onRelease) onRelease({ x: s.smX, y: s.smY, state: s })
    })

    const video = document.getElementById('gesture-video')
    if (video && window.Camera) {
      new window.Camera(video, {
        onFrame: async () => await hands.send({ image: video }),
        width: 640, height: 480
      }).start()
    }
  }, [onHandMove, onPinch, onRelease, onTwoHand, gestureOn])

  return { initEngine, state }
}