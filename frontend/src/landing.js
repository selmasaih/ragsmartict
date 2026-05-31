// Populate the live indexed-chunk count on the landing page. Fails silently
// (keeps the placeholder) if the backend isn't running.
const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000/api';

async function loadStats() {
  try {
    const res = await fetch(`${API_BASE}/stats`);
    if (!res.ok) return;
    const { doc_count } = await res.json();
    if (typeof doc_count === 'number' && doc_count > 0) {
      const el = document.getElementById('stat-chunks');
      if (el) el.textContent = doc_count.toLocaleString('fr-FR');
    }
  } catch {
    /* backend offline — leave the placeholder */
  }
}

loadStats();
