import './style.css';
import 'katex/dist/katex.min.css';
import { createIcons, icons } from 'lucide';
import { marked } from 'marked';
import DOMPurify from 'dompurify';
import renderMathInElement from 'katex/contrib/auto-render';

// Render markdown to sanitized HTML (defends against HTML injected via the
// LLM answer or document sources).
function renderMarkdown(text) {
  return DOMPurify.sanitize(marked.parse(text));
}

// Typeset LaTeX ($…$, $$…$$, \(…\), \[…\]) inside an element. Runs on the live
// DOM after sanitization, so KaTeX output isn't stripped by DOMPurify.
function typesetMath(el) {
  if (!el) return;
  try {
    renderMathInElement(el, {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '\\[', right: '\\]', display: true },
        { left: '$', right: '$', display: false },
        { left: '\\(', right: '\\)', display: false },
      ],
      throwOnError: false,
    });
  } catch {
    /* leave raw text on failure */
  }
}

createIcons({ icons, nameAttr: 'data-lucide' });

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000/api';

// Stable per-browser session id so uploaded documents stay private to this user.
function getSessionId() {
  let sid = localStorage.getItem('inpt_session');
  if (!sid) {
    sid = (crypto.randomUUID && crypto.randomUUID()) || Date.now().toString(36) + Math.random().toString(36).slice(2);
    localStorage.setItem('inpt_session', sid);
  }
  return sid;
}
const SESSION_ID = getSessionId();

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

// ── Multi-conversation history (localStorage) ─────────────────────────
const CONV_KEY = 'inpt_conversations';
let conversations = []; // [{ id, title, messages:[{role,content}], updatedAt }]
let currentConvId = null;

const historyButton = document.getElementById('history-button');
const historyPanel = document.getElementById('history-panel');
const historyClose = document.getElementById('history-close');
const historyList = document.getElementById('history-list');
const historyNew = document.getElementById('history-new');

function fmtDate(ts) {
  const d = new Date(ts);
  const diff = (Date.now() - ts) / 1000;
  if (diff < 60) return "à l'instant";
  if (diff < 3600) return `il y a ${Math.floor(diff / 60)} min`;
  if (diff < 86400) return `il y a ${Math.floor(diff / 3600)} h`;
  return d.toLocaleDateString('fr-FR', { day: '2-digit', month: 'short' });
}

function renderHistoryList() {
  if (!historyList) return;
  if (conversations.length === 0) {
    historyList.innerHTML = `<p style="opacity:.6;padding:8px;">Aucune conversation.</p>`;
    return;
  }
  historyList.innerHTML = conversations
    .map(
      (c) => `
      <div class="conv-row ${c.id === currentConvId ? 'active' : ''}" data-id="${c.id}">
        <div class="conv-meta">
          <div class="conv-title">${escapeHtml(c.title || 'Conversation')}</div>
          <div class="conv-date">${fmtDate(c.updatedAt)}</div>
        </div>
        <button class="conv-del" data-id="${c.id}" title="Supprimer"><i data-lucide="trash-2"></i></button>
      </div>`
    )
    .join('');
  refreshIcons();
  historyList.querySelectorAll('.conv-row').forEach((row) => {
    row.addEventListener('click', (e) => {
      if (e.target.closest('.conv-del')) return;
      openConversation(row.getAttribute('data-id'));
    });
  });
  historyList.querySelectorAll('.conv-del').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      deleteConversation(btn.getAttribute('data-id'));
    });
  });
}

if (historyButton) {
  historyButton.addEventListener('click', () => {
    if (!historyPanel) return;
    historyPanel.hidden = !historyPanel.hidden;
    if (!historyPanel.hidden) renderHistoryList();
  });
  if (historyClose) historyClose.addEventListener('click', () => (historyPanel.hidden = true));
  if (historyNew) historyNew.addEventListener('click', newConversation);
}

const WELCOME_HTML = `
  <div class="message assistant-message">
    <div class="avatar"><i data-lucide="bot"></i></div>
    <div class="message-content">
      <p>Bonjour ! Je suis votre assistant pour le cours <strong>Smart ICT</strong>. Posez une question ou importez un document.</p>
    </div>
  </div>`;

function genId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
}

function loadConversations() {
  try {
    conversations = JSON.parse(localStorage.getItem(CONV_KEY) || '[]');
  } catch {
    conversations = [];
  }
  if (!Array.isArray(conversations)) conversations = [];
}

function saveConversations() {
  try {
    localStorage.setItem(CONV_KEY, JSON.stringify(conversations.slice(0, 100)));
  } catch {
    /* storage unavailable — ignore */
  }
}

function currentConv() {
  return conversations.find((c) => c.id === currentConvId) || null;
}

function pushMessage(role, content) {
  let c = currentConv();
  if (!c) {
    c = { id: genId(), title: 'Nouvelle conversation', messages: [], updatedAt: Date.now() };
    currentConvId = c.id;
    conversations.unshift(c);
  }
  c.messages.push({ role, content });
  if (role === 'user' && c.messages.filter((m) => m.role === 'user').length === 1) {
    c.title = (content || 'Conversation').slice(0, 48);
  }
  c.updatedAt = Date.now();
  conversations = [c, ...conversations.filter((x) => x.id !== c.id)]; // most recent first
  saveConversations();
  renderHistoryList();
}

function renderConversation(c) {
  chatBox.innerHTML = '';
  if (!c || c.messages.length === 0) {
    chatBox.innerHTML = WELCOME_HTML;
  } else {
    for (const m of c.messages) {
      if (m.role === 'user') appendMessage('user', `<p>${escapeHtml(m.content)}</p>`);
      else appendMessage('assistant', renderMarkdown(m.content));
    }
    typesetMath(chatBox);
  }
  refreshIcons();
  chatHistory = c ? c.messages.slice(-6) : [];
}

function openConversation(id) {
  if (currentController) currentController.abort();
  currentConvId = id;
  setFocusedDoc(null);
  renderConversation(currentConv());
  if (historyPanel) historyPanel.hidden = true;
  renderHistoryList();
}

function newConversation() {
  if (currentController) currentController.abort();
  currentConvId = null; // created lazily on first message
  chatHistory = [];
  setFocusedDoc(null);
  chatBox.innerHTML = WELCOME_HTML;
  refreshIcons();
  if (historyPanel) historyPanel.hidden = true;
  renderHistoryList();
}

function deleteConversation(id) {
  conversations = conversations.filter((c) => c.id !== id);
  saveConversations();
  if (id === currentConvId) newConversation();
  else renderHistoryList();
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
      session: SESSION_ID,
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
      typesetMath(handles.body); // render LaTeX once the answer is complete
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
  const res = await fetch(`${API_BASE}/upload`, {
    method: 'POST',
    headers: { 'X-Session-Id': SESSION_ID },
    body: form,
  });
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
  pushMessage('user', question || `📎 ${file ? file.name : 'document'}`);
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
      if (answer) pushMessage('assistant', answer);
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
if (newChatButton) newChatButton.addEventListener('click', newConversation);

// Load saved conversations and open the most recent (or a fresh one).
loadConversations();
if (conversations.length > 0) openConversation(conversations[0].id);
else newConversation();
