import { useState } from 'react'
import ChatView from './components/ChatView'
import EvalView from './components/EvalView'
import './App.css'

export default function App() {
  const [tab, setTab] = useState('chat')

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>PolicyLens</h1>
          <p className="app-subtitle">
            Agentic RAG over insurance &amp; regulatory documents —{' '}
            <a href="https://github.com/VedaantAgrawal/policylens" target="_blank" rel="noopener noreferrer">
              source on GitHub
            </a>
          </p>
        </div>
        <nav className="tab-nav">
          <button type="button" className={tab === 'chat' ? 'tab-btn tab-btn--active' : 'tab-btn'} onClick={() => setTab('chat')}>
            Chat
          </button>
          <button type="button" className={tab === 'eval' ? 'tab-btn tab-btn--active' : 'tab-btn'} onClick={() => setTab('eval')}>
            Eval Results
          </button>
        </nav>
      </header>

      <main className="app-main">{tab === 'chat' ? <ChatView /> : <EvalView />}</main>
    </div>
  )
}
