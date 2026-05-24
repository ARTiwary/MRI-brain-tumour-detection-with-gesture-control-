import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css'

// Catch render errors
class ErrorBoundary extends React.Component {
  constructor(props) { super(props); this.state = { error: null } }
  static getDerivedStateFromError(error) { return { error } }
  render() {
    if (this.state.error) {
      return (
        <div style={{
          color: '#f87171', fontFamily: 'monospace',
          padding: 40, background: '#03060f', minHeight: '100vh'
        }}>
          <h2>⚠ Render Error</h2>
          <pre style={{ marginTop: 16, fontSize: 12, whiteSpace: 'pre-wrap' }}>
            {this.state.error.toString()}
            {'\n\n'}
            {this.state.error.stack}
          </pre>
        </div>
      )
    }
    return this.props.children
  }
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <ErrorBoundary>
    <App />
  </ErrorBoundary>
)