const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

async function post(path, body) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`${path} failed: ${res.status} ${text}`.trim())
  }
  return res.json()
}

export function queryDirect(question) {
  return post('/query', { question })
}

export function queryAgent(question) {
  return post('/agent/query', { question })
}

export async function fetchEvalResults() {
  const res = await fetch(`${API_BASE}/eval`)
  if (!res.ok) throw new Error(`/eval failed: ${res.status}`)
  return res.json()
}

export { API_BASE }
