import './style.css';
import { createIcons, icons } from 'lucide';
import { marked } from 'marked';

createIcons({ icons, nameAttr: 'data-lucide' });

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000/api';

const chatBox = document.getElementById('chat-box');
const chatForm = document.getElementById('chat-form');
const questionInput = document.getElementById('question-input');
const docCountEl = document.getElementById('doc-count');
const topicSelect = document.getElementById('topic-select');

function refreshIcons() {
  createIcons({ icons, nameAttr: 'data-lucide', attrs: { class: 'lucide' } });
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

async function fetchStats() {
  try {
    const res = await fetch(`${API_BASE}/stats`);
    const data = await res.json();
    docCountEl.textContent = data.doc_count;
  } catch (err) {
    docCountEl.textContent = '0';
    console.error('Error fetching stats:', err);
  }
}

async function fetchTopics() {
  if (!topicSelect) return;
  try {
    const res = await fetch(`${API_BASE}/topics`);
    const data = await res.json();
    for (const topic of data.topics || []) {
      const opt = document.createElement('option');
      opt.value = topic;
      opt.textContent = topic;
      topicSelect.appendChild(opt);
    }
  } catch (err) {
    console.error('Error fetching topics:', err);
  }
}

fetchStats();
fetchTopics();

function renderSourcesHTML(sources) {
  if (!sources || sources.length === 0) return '';
  const items = sources
    .map(
      (s, i) => `
      <div class="source-item">
        <strong>Source ${i + 1}: ${escapeHtml(s.filename)} (Page ${escapeHtml(String(s.page_number))})</strong>
        <div>${s.text ? escapeHtml(s.text.substring(0, 150)) + '...' : 'Aperçu non disponible'}</div>
      </div>`
    )
    .join('');
  return `
    <div class="sources-container">
      <details>
        <summary class="sources-summary">
          <i data-lucide="chevron-right" style="width:16px;height:16px;"></i>
          Voir les sources (${sources.length})
        </summary>
        <div class="sources-list" style="margin-top: 10px;">${items}</div>
      </details>
    </div>`;
}

function appendMessage(role, contentHTML, latency = null, sources = []) {
  const msgDiv = document.createElement('div');
  msgDiv.className = `message ${role}-message`;
  const icon = role === 'user' ? 'user' : 'bot';
  const latencyHTML = latency ? `<span class="latency">⏱️ Latence: ${latency}ms</span>` : '';
  msgDiv.innerHTML = `
    <div class="avatar"><i data-lucide="${icon}"></i></div>
    <div class="message-content">
      ${contentHTML}
      ${latencyHTML}
      ${renderSourcesHTML(sources)}
    </div>`;
  chatBox.appendChild(msgDiv);
  refreshIcons();
  chatBox.scrollTop = chatBox.scrollHeight;
}

// Create an empty assistant message and return handles to update it live.
function createStreamingMessage() {
  const msgDiv = document.createElement('div');
  msgDiv.className = 'message assistant-message';
  msgDiv.innerHTML = `
    <div class="avatar"><i data-lucide="bot"></i></div>
    <div class="message-content">
      <div class="answer-body"><div class="typing-indicator"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div></div>
      <div class="meta-row"></div>
    </div>`;
  chatBox.appendChild(msgDiv);
  refreshIcons();
  chatBox.scrollTop = chatBox.scrollHeight;
  return {
    body: msgDiv.querySelector('.answer-body'),
    meta: msgDiv.querySelector('.meta-row'),
  };
}

let chatHistory = [];

function setSendingState(sending) {
  questionInput.disabled = sending;
  const btn = document.getElementById('send-button');
  if (btn) btn.disabled = sending;
}

async function streamAnswer(question, handles) {
  const res = await fetch(`${API_BASE}/query/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
      history: chatHistory,
      topic: topicSelect && topicSelect.value ? topicSelect.value : null,
    }),
  });

  if (!res.ok || !res.body) {
    throw new Error(`HTTP ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let answer = '';
  let sources = [];

  const flushEvent = (raw) => {
    const line = raw.replace(/^data:\s?/, '');
    if (!line) return;
    let event;
    try {
      event = JSON.parse(line);
    } catch {
      return;
    }
    if (event.type === 'token') {
      answer += event.text;
      handles.body.innerHTML = marked.parse(answer);
    } else if (event.type === 'sources') {
      sources = event.sources || [];
    } else if (event.type === 'done') {
      handles.meta.innerHTML = `<span class="latency">⏱️ Latence: ${event.latency_ms}ms</span>${renderSourcesHTML(sources)}`;
      refreshIcons();
    } else if (event.type === 'error') {
      if (!answer) {
        handles.body.innerHTML = `<p style="color:#ef4444;">Erreur: ${event.error}</p>`;
      }
    }
    chatBox.scrollTop = chatBox.scrollHeight;
  };

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split('\n\n');
    buffer = parts.pop();
    for (const part of parts) flushEvent(part.trim());
  }
  if (buffer.trim()) flushEvent(buffer.trim());

  return answer;
}

chatForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  appendMessage('user', `<p>${escapeHtml(question)}</p>`);
  questionInput.value = '';
  setSendingState(true);

  const handles = createStreamingMessage();

  try {
    const answer = await streamAnswer(question, handles);
    chatHistory.push({ role: 'user', content: question });
    chatHistory.push({ role: 'assistant', content: answer });
    if (chatHistory.length > 6) chatHistory = chatHistory.slice(chatHistory.length - 6);
  } catch (error) {
    handles.body.innerHTML = `<p style="color:#ef4444;">Impossible de se connecter au serveur Backend. Assurez-vous que FastAPI est lancé sur le port 8000.</p>`;
    console.error(error);
  } finally {
    setSendingState(false);
    questionInput.focus();
  }
});
