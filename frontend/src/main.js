import './style.css';
import { createIcons, icons } from 'lucide';
import { marked } from 'marked';

// Initialize Lucide icons
createIcons({
  icons,
  nameAttr: 'data-lucide'
});

const API_BASE = 'http://127.0.0.1:8003/api';

const chatBox = document.getElementById('chat-box');
const chatForm = document.getElementById('chat-form');
const questionInput = document.getElementById('question-input');
const docCountEl = document.getElementById('doc-count');

// Fetch stats on load
async function fetchStats() {
  try {
    const res = await fetch(`${API_BASE}/stats`);
    const data = await res.json();
    docCountEl.textContent = data.doc_count;
  } catch (err) {
    docCountEl.textContent = '0';
    console.error("Error fetching stats:", err);
  }
}
fetchStats();

function appendMessage(role, contentHTML, latency = null, sources = []) {
  const msgDiv = document.createElement('div');
  msgDiv.className = `message ${role}-message`;
  
  const icon = role === 'user' ? 'user' : 'bot';
  
  let sourcesHTML = '';
  if (sources.length > 0) {
    const sourcesItems = sources.map((s, i) => `
      <div class="source-item">
        <strong>Source ${i+1}: ${s.filename} (Page ${s.page_number})</strong>
        <div>${s.text ? s.text.substring(0, 150) + '...' : 'Aperçu non disponible'}</div>
      </div>
    `).join('');
    
    sourcesHTML = `
      <div class="sources-container">
        <details>
          <summary class="sources-summary">
            <i data-lucide="chevron-right" style="width:16px;height:16px;"></i>
            Voir les sources (${sources.length})
          </summary>
          <div class="sources-list" style="margin-top: 10px;">
            ${sourcesItems}
          </div>
        </details>
      </div>
    `;
  }

  const latencyHTML = latency ? `<span class="latency">⏱️ Latence: ${latency}ms</span>` : '';

  msgDiv.innerHTML = `
    <div class="avatar"><i data-lucide="${icon}"></i></div>
    <div class="message-content">
      ${contentHTML}
      ${latencyHTML}
      ${sourcesHTML}
    </div>
  `;
  
  chatBox.appendChild(msgDiv);
  
  // Re-init newly added icons
  createIcons({
    icons,
    nameAttr: 'data-lucide',
    attrs: { class: "lucide" }
  });
  
  chatBox.scrollTop = chatBox.scrollHeight;
}

function showTypingIndicator() {
  const msgDiv = document.createElement('div');
  msgDiv.className = `message assistant-message typing-indicator-container`;
  msgDiv.id = 'typing-indicator';
  msgDiv.innerHTML = `
    <div class="avatar"><i data-lucide="bot"></i></div>
    <div class="message-content">
      <div class="typing-indicator">
        <div class="dot"></div>
        <div class="dot"></div>
        <div class="dot"></div>
      </div>
    </div>
  `;
  chatBox.appendChild(msgDiv);
  createIcons({ icons, nameAttr: 'data-lucide' });
  chatBox.scrollTop = chatBox.scrollHeight;
}

function removeTypingIndicator() {
  const indicator = document.getElementById('typing-indicator');
  if (indicator) indicator.remove();
}

chatForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  // Append user message
  appendMessage('user', `<p>${question}</p>`);
  questionInput.value = '';

  showTypingIndicator();

  try {
    const res = await fetch(`${API_BASE}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question })
    });

    const data = await res.json();
    removeTypingIndicator();

    if (res.ok) {
      appendMessage('assistant', marked.parse(data.answer), data.latency_ms, data.sources);
    } else {
      appendMessage('assistant', `<p style="color: #ef4444;">Erreur: ${data.detail || 'Erreur serveur'}</p>`);
    }
  } catch (error) {
    removeTypingIndicator();
    appendMessage('assistant', `<p style="color: #ef4444;">Impossible de se connecter au serveur Backend. Assurez-vous que FastAPI est lance sur le port 8002.</p>`);
  }
});
