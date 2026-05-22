import { useState, useCallback } from 'react'
import { quickValidate } from '../utils/validation'

const API = 'http://127.0.0.1:8000'

export function usePredict() {
  const [results,   setResults]   = useState([])
  const [loading,   setLoading]   = useState(false)
  const [modalOpen, setModalOpen] = useState(false)

  const predict = useCallback(async (queue, selectedModel) => {
    if (!queue.length) return
    setLoading(true)
    setModalOpen(true)
    setResults([])

    for (const item of queue) {
      // Frontend validation
      const check = await quickValidate(item.file)
      if (!check.valid) {
        setResults(prev => [...prev, {
          type: 'rejected',
          url: item.url,
          name: item.file.name,
          reason: check.reason,
          hint: 'Upload a grayscale brain MRI scan (T1/T2/FLAIR).'
        }])
        continue
      }

      const fd = new FormData()
      fd.append('file', item.file)

      try {
        const r = await fetch(`${API}/predict?model_key=${selectedModel}`, {
          method: 'POST',
          body: fd
        })

        if (r.status === 422) {
          const err = await r.json()
          const detail = err.detail || {}
          setResults(prev => [...prev, {
            type: 'rejected',
            url: item.url,
            name: item.file.name,
            reason: detail.message || 'Image rejected by server.',
            hint: detail.hint || ''
          }])
          continue
        }

        if (!r.ok) {
          const err = await r.json()
          throw new Error(err.detail || `HTTP ${r.status}`)
        }

        const d = await r.json()
        setResults(prev => [...prev, {
          type: 'success',
          url: item.url,
          name: item.file.name,
          ...d
        }])
      } catch (e) {
        setResults(prev => [...prev, {
          type: 'error',
          url: item.url,
          name: item.file.name,
          message: e.message || 'No response from 127.0.0.1:8000'
        }])
      }
    }

    setLoading(false)
  }, [])

  const closeModal = useCallback(() => setModalOpen(false), [])

  return { results, loading, modalOpen, predict, closeModal }
}