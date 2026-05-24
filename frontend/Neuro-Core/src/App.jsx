import { useState, useRef, useCallback } from 'react'

import ParticleBackground from './components/ParticleBackground'
import SphereBackground from './components/SphereBackground'
import FolderPanel from './components/FolderPanel'
import UploadCard from './components/UploadCard'
import SystemLogCard from './components/SystemLogCard'
import ResultModal from './components/ResultModal'
import GestureEngine from './components/GestureEngine'

import { useQueue } from './hooks/useQueue'
import { usePredict } from './hooks/usePredict'

export default function App() {

  // ─────────────────────────────────────────────
  // Gesture State
  // ─────────────────────────────────────────────
  const [gestureOn, setGestureOn] = useState(false)
  const [handX, setHandX] = useState(0.5)
  const [handY, setHandY] = useState(0.5)

  const [sensorStatus, setSensorStatus] = useState('OFFLINE')

  const [logText, setLogText] = useState('IDLE')

  const [logColor, setLogColor] =
    useState('rgba(100,140,180,0.55)')

  const [airGrabs, setAirGrabs] = useState(0)

  // ─────────────────────────────────────────────
  // Model
  // ─────────────────────────────────────────────
  const [selectedModel, setSelectedModel] =
    useState('resnet18')

  // ─────────────────────────────────────────────
  // Queue + Predict
  // ─────────────────────────────────────────────
  const {
    queue,
    addFiles,
    clearQueue
  } = useQueue()

  const {
    results,
    loading,
    modalOpen,
    predict,
    closeModal
  } = usePredict()

  // ─────────────────────────────────────────────
  // Cursor Refs
  // ─────────────────────────────────────────────
  const cursorRef = useRef(null)
  const cursor2Ref = useRef(null)
  const tooltipRef = useRef(null)

  // ─────────────────────────────────────────────
  // Handlers
  // ─────────────────────────────────────────────
  const handleHandMove = useCallback((x, y) => {
    setHandX(x)
    setHandY(y)
  }, [])

  const handleLogUpdate = useCallback((text, color) => {
    setLogText(text)
    setLogColor(color)
  }, [])

  const handleAirGrab = useCallback((count) => {
    setAirGrabs(count)
  }, [])

  const handleModalScroll = useCallback((delta) => {

    if (typeof document === 'undefined') return

    const modal =
      document.querySelector('[data-result-modal]')

    if (modal) {
      modal.scrollTop += delta
    }

  }, [])

  const handleFilesQueued = useCallback((files) => {
    addFiles(files)
  }, [addFiles])

  const handleProcess = useCallback(() => {

    if (!queue.length) return

    predict(queue, selectedModel)

  }, [predict, queue, selectedModel])

  // ─────────────────────────────────────────────
  // Toggle Gesture
  // ─────────────────────────────────────────────
  const toggleGesture = () => {

    setGestureOn(prev => !prev)

    if (gestureOn) {

      setAirGrabs(0)

      setLogText('IDLE')

      setLogColor('rgba(100,140,180,0.55)')

      setSensorStatus('OFFLINE')
    }
  }

  // ─────────────────────────────────────────────
  // Render
  // ─────────────────────────────────────────────
  return (

    <div
      style={{
        width: '100%',
        minHeight: '100vh',
        overflow: 'hidden',
        position: 'relative',
        background: '#03060f'
      }}
    >

      {/* Background Layer */}
      <div
        style={{
          position: 'fixed',
          inset: 0,
          zIndex: 0,
          opacity: gestureOn ? 0 : 1,
          transition: 'opacity 0.8s ease',
          background: `
            radial-gradient(
              ellipse 80% 60% at 50% 0%,
              rgba(0,180,216,0.08) 0%,
              transparent 70%
            ),
            #03060f
          `
        }}
      />

      {/* Particle Background */}
      <ParticleBackground visible={!gestureOn} />

      {/* Sphere Background */}
      <SphereBackground
        active={gestureOn}
        handX={handX}
        handY={handY}
      />

      {/* Gesture Cursor */}
      <div
        ref={cursorRef}
        className="gesture-cursor"
      />

      <div
        ref={cursor2Ref}
        className="gesture-cursor-2"
      />

      {/* Tooltip */}
      <div
        ref={tooltipRef}
        id="g-tooltip"
        style={{
          position: 'fixed',
          zIndex: 9998,
          pointerEvents: 'none',
          opacity: 0
        }}
      />

      {/* Toggle Button */}
      <button
        onClick={toggleGesture}
        style={{
          position: 'fixed',
          top: 20,
          right: 20,
          zIndex: 1000,

          padding: '12px 20px',

          background: 'rgba(6,14,32,0.9)',

          border: gestureOn
            ? '1px solid #00f2ff'
            : '1px solid rgba(0,242,255,0.25)',

          borderRadius: 999,

          color: '#00f2ff',

          cursor: 'pointer',

          fontFamily: 'Orbitron'
        }}
      >
        {gestureOn
          ? 'GESTURE ON'
          : 'GESTURE OFF'}
      </button>

      {/* Air Grab Badge */}
      {airGrabs > 0 && (
        <div
          style={{
            position: 'fixed',
            top: 70,
            right: 20,
            zIndex: 1001,

            padding: '8px 14px',

            background: 'rgba(3,7,18,0.9)',

            border:
              '1px solid rgba(245,158,11,0.4)',

            borderRadius: 20,

            color: '#f59e0b',

            fontFamily: 'Orbitron',

            fontSize: 10
          }}
        >
          AIR GRABS: {airGrabs}
        </div>
      )}

      {/* Gesture Engine */}
      <GestureEngine
        gestureOn={gestureOn}
        onHandMove={handleHandMove}
        onSensorStatus={setSensorStatus}
        onLogUpdate={handleLogUpdate}
        onAirGrab={handleAirGrab}
        modalOpen={modalOpen}
        onModalScroll={handleModalScroll}
        cursorRef={cursorRef}
        cursor2Ref={cursor2Ref}
        tooltipRef={tooltipRef}
      />

      {/* Sidebar */}
      <FolderPanel
        onFilesQueued={handleFilesQueued}
      />

      {/* Main Content */}
      <div
        style={{
          position: 'relative',
          zIndex: 10,

          marginLeft: 258,

          display: 'flex',
          flexDirection: 'column',

          alignItems: 'center',
          justifyContent: 'center',

          minHeight: '100vh',

          padding: '32px 40px'
        }}
      >

        {/* Logo */}
        <div
          style={{
            textAlign: 'center',
            marginBottom: 48
          }}
        >
          <div
            style={{
              fontFamily: 'Orbitron',
              fontSize: 72,
              fontWeight: 900,
              color: '#00f2ff'
            }}
          >
            NEUROCORE
          </div>

          <div
            style={{
              fontSize: 11,
              letterSpacing: '0.3em',
              color: 'rgba(100,140,180,0.7)',
              marginTop: 10
            }}
          >
            STERILE NEURAL DIAGNOSTIC INTERFACE
          </div>
        </div>

        {/* Cards */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: 20,
            width: '100%',
            maxWidth: 860
          }}
        >
          <UploadCard
            queue={queue}
            onAddFiles={addFiles}
          />

          <SystemLogCard
            logText={logText}
            logColor={logColor}
            sensorStatus={sensorStatus}
            selectedModel={selectedModel}
            onModelSelect={setSelectedModel}
            queueLength={queue.length}
            onProcess={handleProcess}
          />
        </div>
      </div>

      {/* Result Modal */}
      <ResultModal
        open={modalOpen}
        results={results}
        loading={loading}
        selectedModel={selectedModel}
        onClose={closeModal}
      />

    </div>
  )
}