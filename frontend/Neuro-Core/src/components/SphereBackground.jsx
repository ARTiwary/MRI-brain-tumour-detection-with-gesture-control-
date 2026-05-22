import { useEffect, useRef } from 'react'
import * as THREE from 'three'

export default function SphereBackground({ active, handX, handY }) {
  const canvasRef  = useRef(null)
  const sceneRef   = useRef(null)
  const rendRef    = useRef(null)
  const meshRef    = useRef(null)
  const pSysRef    = useRef(null)
  const frameRef   = useRef(null)

  useEffect(() => {
    if (!canvasRef.current) return
    if (sceneRef.current) return

    const canvas = canvasRef.current
    const scene  = new THREE.Scene()
    sceneRef.current = scene

    const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.1, 1000)
    camera.position.z = 5

    const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true })
    renderer.setSize(innerWidth, innerHeight)
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2))
    rendRef.current = renderer

    // Morphing sphere
    const geo = new THREE.IcosahedronGeometry(1.6, 5)
    const mat = new THREE.MeshPhongMaterial({
      color: 0x00b4d8, wireframe: true, transparent: true, opacity: 0.3
    })
    const mesh = new THREE.Mesh(geo, mat)
    scene.add(mesh)
    meshRef.current = mesh
    geo.userData.orig = new Float32Array(geo.attributes.position.array)

    // Inner glow
    const innerGeo = new THREE.SphereGeometry(1.4, 32, 32)
    const innerMat = new THREE.MeshPhongMaterial({ color: 0x001a2e, transparent: true, opacity: 0.85 })
    scene.add(new THREE.Mesh(innerGeo, innerMat))

    // Particles
    const pGeo   = new THREE.BufferGeometry()
    const pCount = 1800
    const pPos   = new Float32Array(pCount * 3)
    const pVel   = new Float32Array(pCount * 3)
    for (let i = 0; i < pCount; i++) {
      const r     = 2.5 + Math.random() * 3
      const theta = Math.random() * Math.PI * 2
      const phi   = Math.acos(2 * Math.random() - 1)
      pPos[i*3]   = r * Math.sin(phi) * Math.cos(theta)
      pPos[i*3+1] = r * Math.sin(phi) * Math.sin(theta)
      pPos[i*3+2] = r * Math.cos(phi)
      pVel[i*3]   = (Math.random() - 0.5) * 0.004
      pVel[i*3+1] = (Math.random() - 0.5) * 0.004
      pVel[i*3+2] = (Math.random() - 0.5) * 0.004
    }
    pGeo.setAttribute('position', new THREE.BufferAttribute(pPos, 3))
    pGeo.userData.vel = pVel
    const pSys = new THREE.Points(pGeo, new THREE.PointsMaterial({
      color: 0x00f2ff, size: 0.04, transparent: true, opacity: 0.6
    }))
    scene.add(pSys)
    pSysRef.current = pSys

    // Lights
    scene.add(new THREE.AmbientLight(0x001a2e, 2))
    const l1 = new THREE.PointLight(0x00f2ff, 2, 20); l1.position.set(3, 3, 3); scene.add(l1)
    const l2 = new THREE.PointLight(0x7c3aed, 1.5, 20); l2.position.set(-3, -2, 2); scene.add(l2)

    const onResize = () => {
      camera.aspect = innerWidth / innerHeight
      camera.updateProjectionMatrix()
      renderer.setSize(innerWidth, innerHeight)
    }
    window.addEventListener('resize', onResize)

    let handNX = 0.5, handNY = 0.5

    const animate = () => {
      frameRef.current = requestAnimationFrame(animate)

      const t       = Date.now() * 0.001
      const offsetX = (handNX - 0.5) * 2.2
      const offsetY = (handNY - 0.5) * -2.2
      const dist    = Math.hypot(offsetX, offsetY)

      const pos  = mesh.geometry.attributes.position.array
      const orig = mesh.geometry.userData.orig
      for (let i = 0; i < pos.length; i += 3) {
        const ox = orig[i], oy = orig[i+1], oz = orig[i+2]
        const nx = ox/1.6, ny = oy/1.6, nz = oz/1.6
        const dot  = nx*offsetX + ny*offsetY
        const warp = Math.sin(t*1.2 + i*0.15)*0.12 + (dot*0.35)*Math.max(0, 1 - dist*0.5)
        pos[i]   = ox + nx*warp
        pos[i+1] = oy + ny*warp
        pos[i+2] = oz + nz*Math.sin(t + i*0.08)*0.1
      }
      mesh.geometry.attributes.position.needsUpdate = true
      mesh.rotation.y = t*0.25 + offsetX*0.3
      mesh.rotation.x = t*0.12 + offsetY*0.2

      const pp  = pSys.geometry.attributes.position.array
      const pv  = pSys.geometry.userData.vel
      const att = new THREE.Vector3(offsetX*0.8, offsetY*0.8, 0)
      for (let i = 0; i < pp.length; i += 3) {
        pp[i]   += pv[i]   + (att.x - pp[i])   * 0.0008
        pp[i+1] += pv[i+1] + (att.y - pp[i+1]) * 0.0008
        pp[i+2] += pv[i+2] + (att.z - pp[i+2]) * 0.0004
        const r  = Math.hypot(pp[i], pp[i+1], pp[i+2])
        if (r > 6 || r < 1.8) { pv[i]*=-1; pv[i+1]*=-1; pv[i+2]*=-1 }
      }
      pSys.geometry.attributes.position.needsUpdate = true
      pSys.rotation.y = t * 0.05

      renderer.render(scene, camera)
    }

    // Expose hand update
    canvas._updateHand = (x, y) => { handNX = x; handNY = y }

    animate()
    return () => {
      window.removeEventListener('resize', onResize)
      cancelAnimationFrame(frameRef.current)
      renderer.dispose()
      sceneRef.current = null
    }
  }, [])

  // Update hand position
  useEffect(() => {
    if (canvasRef.current?._updateHand) {
      canvasRef.current._updateHand(handX, handY)
    }
  }, [handX, handY])

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'fixed', inset: 0, zIndex: 1,
        pointerEvents: 'none',
        opacity: active ? 1 : 0,
        transition: 'opacity 1.2s ease'
      }}
    />
  )
}