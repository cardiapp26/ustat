import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import UpdatePrompt from './components/UpdatePrompt'
import SessionRecoveredNotice from './components/SessionRecoveredNotice'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
    {/* Floating toast — visible on every screen (upload zone, power panel,
        main app) so users always see when a newer build is ready. */}
    <UpdatePrompt />
    {/* Same reason it sits here rather than in a panel: a recovery can happen
        on any screen, and the rollback it reports is the whole app's, not one
        tab's. */}
    <SessionRecoveredNotice />
  </StrictMode>,
)
