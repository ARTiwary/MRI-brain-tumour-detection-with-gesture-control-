/**
 * Frontend quick validation before sending to API
 * Catches obvious non-MRI files instantly
 */
export async function quickValidate(file) {
  // File size check
  if (file.size > 10 * 1024 * 1024) {
    return { valid: false, reason: 'File too large (max 10 MB).' }
  }

  return new Promise((resolve) => {
    const img = new Image()
    img.onload = () => {
      const w = img.naturalWidth
      const h = img.naturalHeight

      // Aspect ratio
      const aspect = w / h
      if (aspect < 0.5 || aspect > 2.0) {
        resolve({
          valid: false,
          reason: `Aspect ratio ${aspect.toFixed(2)} is unusual for brain MRI (expected 0.5–2.0).`
        })
        return
      }

      // Color saturation check
      const c   = document.createElement('canvas')
      c.width   = Math.min(w, 128)
      c.height  = Math.min(h, 128)
      const ctx = c.getContext('2d')
      ctx.drawImage(img, 0, 0, c.width, c.height)
      const data = ctx.getImageData(0, 0, c.width, c.height).data

      let totalDiff = 0, count = 0
      for (let i = 0; i < data.length; i += 4) {
        const r = data[i], g = data[i + 1], b = data[i + 2]
        totalDiff += Math.abs(r - g) + Math.abs(r - b) + Math.abs(g - b)
        count++
      }
      const avgDiff = totalDiff / (count * 3)

      if (avgDiff > 40) {
        resolve({
          valid: false,
          reason: `Image appears to be a color photo (color variance: ${avgDiff.toFixed(1)}). Brain MRI scans are grayscale.`
        })
        return
      }

      resolve({ valid: true })
    }
    img.onerror = () => resolve({ valid: false, reason: 'Could not read image file.' })
    img.src = URL.createObjectURL(file)
  })
}