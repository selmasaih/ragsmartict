import './style.css';
import { createIcons, icons } from 'lucide';
import { marked } from 'marked';
import DOMPurify from 'dompurify';

// Render markdown to sanitized HTML (defends against HTML injected via the
// LLM answer or document sources).
function renderMarkdown(text) {
  return DOMPurify.sanitize(marked.parse(text));
}

createIcons({ icons, nameAttr: 'data-lucide' });

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000/api';

const chatBox = document.getElementById('chat-box');
const chatForm = document.getElementById('chat-form');
const questionInput = document.getElementById('question-input');
const docCountEl = document.getElementById('doc-count');
const topicSelect = document.getElementById('topic-select');
const fileInput = document.getElementById('file-input');
const uploadButton = document.getElementById('upload-button');

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
let currentController = null;
let focusedDoc = null; // when set, questions are scoped to this uploaded document

// ── Conversation persistence (survives page refresh) ──────────────────
const TRANSCRIPT_KEY = 'inpt_transcript';
let transcript = [];

function saveTranscript() {
  try {
    localStorage.setItem(TRANSCRIPT_KEY, JSON.stringify(transcript.slice(-50)));
  } catch {
    /* storage full or unavailable — ignore */
  }
}

function pushTranscript(role, content) {
  transcript.push({ role, content });
  saveTranscript();
}

function restoreTranscript() {
  try {
    transcript = JSON.parse(localStorage.getItem(TRANSCRIPT_KEY) || '[]');
  } catch {
    transcript = [];
  }
  if (!Array.isArray(transcript) || transcript.length === 0) return;
  chatBox.innerHTML = '';
  for (const m of transcript) {
    if (m.role === 'user') appendMessage('user', `<p>${escapeHtml(m.content)}</p>`);
    else appendMessage('assistant', renderMarkdown(m.content));
  }
  chatHistory = transcript.slice(-6);
}

// While sending, the send button becomes a stop button.
function setSendingState(sending) {
  questionInput.disabled = sending;
  const btn = document.getElementById('send-button');
  if (!btn) return;
  btn.disabled = false;
  btn.innerHTML = sending ? '<i data-lucide="square"></i>' : '<i data-lucide="send"></i>';
  refreshIcons();
}

async function streamAnswer(question, handles) {
  currentController = new AbortController();
  const res = await fetch(`${API_BASE}/query/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    signal: currentController.signal,
    body: JSON.stringify({
      question,
      history: chatHistory,
      topic: topicSelect && topicSelect.value ? topicSelect.value : null,
      filename: focusedDoc || null,
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
      handles.body.innerHTML = renderMarkdown(answer);
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

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split('\n\n');
      buffer = parts.pop();
      for (const part of parts) flushEvent(part.trim());
    }
    if (buffer.trim()) flushEvent(buffer.trim());
  } catch (err) {
    if (err.name !== 'AbortError') throw err;
    // Stopped by the user: keep the partial answer.
  }

  return answer;
}

const manageButton = document.getElementById('manage-button');
const docsPanel = document.getElementById('docs-panel');
const docsClose = document.getElementById('docs-close');
const docsList = document.getElementById('docs-list');
const docsFilter = document.getElementById('docs-filter');
let allDocs = [];

function renderDocs() {
  const q = (docsFilter.value || '').toLowerCase();
  const docs = allDocs.filter(
    (d) => !q || (d.filename || '').toLowerCase().includes(q) || (d.topic || '').toLowerCase().includes(q)
  );
  if (docs.length === 0) {
    docsList.innerHTML = `<p style="opacity:.6;padding:8px;">Aucun document.</p>`;
    return;
  }
  docsList.innerHTML = docs
    .map(
      (d) => `
      <div class="doc-row">
        <div class="doc-meta">
          <div class="doc-name">${escapeHtml(d.filename || '?')}</div>
          <div class="doc-sub">${escapeHtml(d.topic || '—')} · ${d.chunks} extraits</div>
        </div>
        <button class="doc-del" data-file="${escapeHtml(d.filename)}" data-sub="${escapeHtml(d.subject || '')}" title="Supprimer">
          <i data-lucide="trash-2"></i>
        </button>
      </div>`
    )
    .join('');
  refreshIcons();
  docsList.querySelectorAll('.doc-del').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const filename = btn.getAttribute('data-file');
      const subject = btn.getAttribute('data-sub') || null;
      if (!confirm(`Supprimer "${filename}" de la base ?`)) return;
      btn.disabled = true;
      try {
        const res = await fetch(`${API_BASE}/documents`, {
          method: 'DELETE',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ filename, subject }),
        });
        if (res.ok) {
          allDocs = allDocs.filter((d) => !(d.filename === filename && (d.subject || '') === (subject || '')));
          renderDocs();
          fetchStats();
        } else {
          const data = await res.json();
          alert(`Erreur: ${data.detail || 'suppression impossible'}`);
          btn.disabled = false;
        }
      } catch {
        alert('Impossible de joindre le serveur.');
        btn.disabled = false;
      }
    });
  });
}

async function openDocsPanel() {
  docsPanel.hidden = false;
  docsList.innerHTML = `<p style="opacity:.6;padding:8px;">Chargement…</p>`;
  try {
    const res = await fetch(`${API_BASE}/documents`);
    const data = await res.json();
    allDocs = data.documents || [];
    renderDocs();
  } catch {
    docsList.innerHTML = `<p style="color:#ef4444;padding:8px;">Erreur de chargement.</p>`;
  }
}

if (manageButton) {
  manageButton.addEventListener('click', openDocsPanel);
  docsClose.addEventListener('click', () => (docsPanel.hidden = true));
  docsFilter.addEventListener('input', renderDocs);
}

// ── Staged attachment: pick a file, keep it as a chip, send with the question ──
const attachmentChip = document.getElementById('attachment-chip');
const attachmentName = document.getElementById('attachment-name');
const attachmentRemove = document.getElementById('attachment-remove');
let stagedFile = null;

function setStagedFile(file) {
  stagedFile = file;
  if (file) {
    attachmentName.textContent = file.name;
    attachmentChip.hidden = false;
    questionInput.placeholder = 'Posez une question sur ce document (optionnel)…';
  } else {
    attachmentChip.hidden = true;
    questionInput.placeholder = "Posez votre question ici (ex: Qu'est-ce que la transformée de Fourier ?)...";
  }
  refreshIcons();
}

if (uploadButton && fileInput) {
  uploadButton.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', () => {
    const file = fileInput.files && fileInput.files[0];
    if (file) setStagedFile(file);
    fileInput.value = '';
  });
  attachmentRemove.addEventListener('click', () => setStagedFile(null));
}

// Ingest a staged file; returns true on success. Renders progress into `handles`.
async function ingestStagedFile(file, handles) {
  handles.body.innerHTML = `<p>📎 Import et indexation de <strong>${escapeHtml(file.name)}</strong>…</p>`;
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${API_BASE}/upload`, { method: 'POST', body: form });
  const data = await res.json();
  if (!res.ok) {
    handles.body.innerHTML = `<p style="color:#ef4444;">Erreur d'import: ${escapeHtml(data.detail || 'import impossible')}</p>`;
    return null;
  }
  handles.body.innerHTML = `<p>✅ <strong>${escapeHtml(data.filename)}</strong> indexé — ${data.chunks_added} extraits (domaine : ${escapeHtml(data.topic)}). Les questions porteront sur ce document.</p>`;
  fetchStats();
  return data.filename;
}

// Active-document focus: scope questions to one uploaded file until cleared.
const focusBanner = document.getElementById('focus-banner');
const focusName = document.getElementById('focus-name');
const focusClear = document.getElementById('focus-clear');

function setFocusedDoc(name) {
  focusedDoc = name || null;
  if (!focusBanner) return;
  if (focusedDoc) {
    focusName.textContent = focusedDoc;
    focusBanner.hidden = false;
    questionInput.placeholder = `Question sur « ${focusedDoc} »…`;
  } else {
    focusBanner.hidden = true;
    questionInput.placeholder = "Posez votre question ici (ex: Qu'est-ce que la transformée de Fourier ?)...";
  }
  refreshIcons();
}
if (focusClear) focusClear.addEventListener('click', () => setFocusedDoc(null));

chatForm.addEventListener('submit', async (e) => {
  e.preventDefault();

  // If a response is streaming, the button acts as "stop".
  if (currentController) {
    currentController.abort();
    return;
  }

  const question = questionInput.value.trim();
  const file = stagedFile;
  if (!question && !file) return;

  // User bubble: show the attachment (if any) and the typed question (if any).
  let bubble = '';
  if (file) bubble += `<p class="attached-line"><i data-lucide="file-text"></i> ${escapeHtml(file.name)}</p>`;
  if (question) bubble += `<p>${escapeHtml(question)}</p>`;
  appendMessage('user', bubble);
  pushTranscript('user', question || `📎 ${file ? file.name : 'document'}`);
  questionInput.value = '';
  setStagedFile(null);
  setSendingState(true);

  try {
    // 1) Ingest the attached file first, if present, and focus questions on it.
    if (file) {
      const upHandles = createStreamingMessage();
      const fname = await ingestStagedFile(file, upHandles);
      if (!fname) return;
      setFocusedDoc(fname);
    }

    // 2) Answer the question, if one was typed.
    if (question) {
      const handles = createStreamingMessage();
      const answer = await streamAnswer(question, handles);
      chatHistory.push({ role: 'user', content: question });
      chatHistory.push({ role: 'assistant', content: answer });
      if (chatHistory.length > 6) chatHistory = chatHistory.slice(chatHistory.length - 6);
      if (answer) pushTranscript('assistant', answer);
    }
  } catch (error) {
    if (error.name !== 'AbortError') {
      appendMessage('assistant', `<p style="color:#ef4444;">Impossible de joindre le serveur Backend.</p>`);
      console.error(error);
    }
  } finally {
    currentController = null;
    setSendingState(false);
    questionInput.focus();
  }
});

const newChatButton = document.getElementById('new-chat-button');
if (newChatButton) {
  newChatButton.addEventListener('click', () => {
    if (currentController) currentController.abort();
    chatHistory = [];
    transcript = [];
    saveTranscript();
    setFocusedDoc(null);
    chatBox.innerHTML = `
      <div class="message assistant-message">
        <div class="avatar"><i data-lucide="bot"></i></div>
        <div class="message-content">
          <p>Nouvelle conversation. Posez votre question sur les cours Smart ICT.</p>
        </div>
      </div>`;
    refreshIcons();
  });
}

// Restore any saved conversation on load.
restoreTranscript();
