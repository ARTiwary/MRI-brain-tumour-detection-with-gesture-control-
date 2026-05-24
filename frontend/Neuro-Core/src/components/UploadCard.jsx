import { useRef, useState } from 'react'

export default function UploadCard({ queue, onAddFiles }) {
  const fileRef  = useRef(null)
  const [dragOver, setDragOver] = useState(false)

  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    const files = Array.from(e.dataTransfer.files).filter(f =>
      f.type === 'image/jpeg' || f.type === 'image/png'
    )
    if (files.length) onAddFiles(files)
  }

  return (
    <div className="nc-card fade-in fade-in-d1">
      <div style={{
        fontFamily: 'Orbitron', fontSize: 9, letterSpacing: '0.2em',
        color: 'rgba(0,242,255,0.55)', marginBottom: 18,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between'
      }}>
        <span>01_INPUT_STREAM</span>
        {queue.length > 0 && (
          <span style={{
            fontFamily: 'Orbitron', fontSize: 9, padding: '2px 9px',
            borderRadius: 4, fontWeight: 700,
            background: 'rgba(74,222,128,0.12)',
            color: '#4ade80',
            border: '1px solid rgba(74,222,128,0.3)'
          }}>
            {queue.length} QUEUED
          </span>
        )}
      </div>

      {/* Drop zone */}
      <div
        className={`drop-zone${dragOver ? ' drag-over' : ''}`}
        onClick={() => fileRef.current?.click()}
        onDragOver={e => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
      >
        <input
          ref={fileRef}
          type="file"
          multiple
          accept="image/*"
          style={{ display: 'none' }}
          onChange={e => onAddFiles(Array.from(e.target.files))}
        />
        <div style={{ fontSize: 28, marginBottom: 10, color: 'rgba(0,242,255,0.4)' }}>
          <i className="fas fa-microchip" />
        </div>
        <div style={{ fontFamily: 'Orbitron', fontSize: 10, letterSpacing: '0.15em' }}>
          {dragOver ? 'DROP TO QUEUE' : 'CLICK OR DROP TO UPLOAD'}
        </div>
        <div style={{ fontSize: 10, color: 'rgba(100,140,180,0.5)', marginTop: 4 }}>
          or select from folder panel ←
        </div>
      </div>

      {/* Queue list */}
      <div style={{ overflowY: 'auto', maxHeight: 140, minHeight: 32 }}>
        {queue.length === 0 ? (
          <p style={{
            color: 'rgba(100,140,180,0.45)', fontSize: 10,
            fontStyle: 'italic', padding: '4px 0'
          }}>No scans loaded...</p>
        ) : (
          queue.map((item, i) => (
            <div key={item.id} style={{
              display: 'flex', alignItems: 'center',
              justifyContent: 'space-between',
              padding: '6px 8px',
              background: 'rgba(0,242,255,0.05)',
              borderLeft: '2px solid rgba(0,242,255,0.5)',
              borderRadius: '0 6px 6px 0',
              marginBottom: 3, fontSize: 9
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 7, overflow: 'hidden' }}>
                <img src={item.url} alt={item.file.name}
                  style={{ width: 16, height: 16, borderRadius: 3, objectFit: 'cover', flexShrink: 0 }} />
                <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {item.file.name}
                </span>
              </div>
              <span style={{
                color: 'rgba(0,242,255,0.45)', flexShrink: 0,
                marginLeft: 4, fontFamily: 'Space Mono'
              }}>#{i + 1}</span>
            </div>
          ))
        )}
      </div>
    </div>
  )
}