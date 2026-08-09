import { useState, useRef, useEffect } from 'react'
import { queryDirect, queryAgent } from '../api'
import TraceViewer from './TraceViewer'
import { renderBold } from '../markdown'

const EXAMPLE_QUESTIONS = [
  'How many days after a premium due date must a policyholder request a paid-up nonforfeiture benefit under the NAIC Standard Nonforfeiture Law?',
  'Compare how NAIC model laws 785 and 786 each treat when a commissioner must allow credit for reinsurance ceded to an assuming insurer.',
  "What was California's total Principle-Based Reserving assessment for life insurers for Fiscal Year 2025-26?",
]

function Citations({ citations }) {
  if (!citations || citations.length === 0) return null
  return (
    <div className="citations">
      {citations.map((c) => (
        <a
          key={c.chunk_id}
          className="citation-chip"
          href={c.url}
          target="_blank"
          rel="noopener noreferrer"
          title={c.section_heading || c.title}
        >
          {c.chunk_id}
        </a>
      ))}
    </div>
  )
}

export default function ChatView() {
  const [mode, setMode] = useState('agent')
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  async function send(question) {
    if (!question.trim() || loading) return
    setError(null)
    setMessages((prev) => [...prev, { role: 'user', content: question }])
    setInput('')
    setLoading(true)
    try {
      if (mode === 'agent') {
        const result = await queryAgent(question)
        setMessages((prev) => [
          ...prev,
          { role: 'assistant', content: result.answer, trace: result.trace, toolCallCount: result.tool_call_count },
        ])
      } else {
        const result = await queryDirect(question)
        setMessages((prev) => [
          ...prev,
          {
            role: 'assistant',
            content: result.answer,
            citations: result.citations,
            answerable: result.answerable,
            costUsd: result.cost_usd,
          },
        ])
      }
    } catch (err) {
      setError(err.message || 'Something went wrong')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="chat-view">
      <div className="chat-controls">
        <div className="mode-toggle" role="radiogroup" aria-label="Query mode">
          <button
            type="button"
            className={mode === 'agent' ? 'mode-btn mode-btn--active' : 'mode-btn'}
            onClick={() => setMode('agent')}
          >
            Agent (multi-step)
          </button>
          <button
            type="button"
            className={mode === 'direct' ? 'mode-btn mode-btn--active' : 'mode-btn'}
            onClick={() => setMode('direct')}
          >
            Direct RAG (single-shot)
          </button>
        </div>
        <p className="mode-note">
          {mode === 'agent'
            ? 'Plans tool calls (search, fetch section, compare, extract) and shows its trace.'
            : 'One retrieval pass into generation, no planning — the S2 hybrid pipeline the eval numbers measure.'}
        </p>
      </div>

      <div className="chat-log">
        {messages.length === 0 && (
          <div className="chat-empty">
            <p>Try one of these, or ask your own question about the corpus:</p>
            <ul className="example-questions">
              {EXAMPLE_QUESTIONS.map((q) => (
                <li key={q}>
                  <button type="button" onClick={() => send(q)}>
                    {q}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`chat-message chat-message--${m.role}`}>
            <div className="chat-message-role">{m.role === 'user' ? 'You' : 'PolicyLens'}</div>
            <div className="chat-message-content">{m.role === 'assistant' ? renderBold(m.content) : m.content}</div>
            {m.role === 'assistant' && m.answerable === false && (
              <div className="refusal-flag">refused to answer — not in corpus</div>
            )}
            {m.role === 'assistant' && <Citations citations={m.citations} />}
            {m.role === 'assistant' && <TraceViewer trace={m.trace} />}
            {m.role === 'assistant' && m.costUsd != null && (
              <div className="chat-cost">${m.costUsd.toFixed(5)}</div>
            )}
          </div>
        ))}
        {loading && (
          <div className="chat-message chat-message--assistant chat-message--loading">
            <div className="chat-message-role">PolicyLens</div>
            <div className="chat-message-content">
              {mode === 'agent' ? 'Planning and calling tools…' : 'Retrieving and generating…'}
            </div>
          </div>
        )}
        {error && <div className="chat-error">Error: {error}</div>}
        <div ref={bottomRef} />
      </div>

      <form
        className="chat-input-row"
        onSubmit={(e) => {
          e.preventDefault()
          send(input)
        }}
      >
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question about the corpus…"
          disabled={loading}
        />
        <button type="submit" disabled={loading || !input.trim()}>
          Send
        </button>
      </form>
    </div>
  )
}
