import { useState, useCallback } from 'react'

export function useQueue() {
  const [queue, setQueue] = useState([])

  const addFiles = useCallback((files) => {
    const newFiles = Array.from(files).map(f => ({
      file: f,
      url: URL.createObjectURL(f),
      id: `${f.name}-${f.size}-${Date.now()}`
    }))
    setQueue(prev => {
      const existing = new Set(prev.map(q => `${q.file.name}-${q.file.size}`))
      const unique = newFiles.filter(f => !existing.has(`${f.file.name}-${f.file.size}`))
      return [...prev, ...unique]
    })
  }, [])

  const clearQueue = useCallback(() => {
    setQueue(prev => { prev.forEach(q => URL.revokeObjectURL(q.url)); return [] })
  }, [])

  return { queue, addFiles, clearQueue }
}