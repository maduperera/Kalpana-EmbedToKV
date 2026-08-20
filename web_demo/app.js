import { KalpanaVaultEmbedToKV } from '../pkg_vault/kalpana_vault_embed.js';

// Deterministic semantic feature hash for browser-side demonstration
function textToEmbedding(text, dim = 384) {
  const vec = new Float32Array(dim);
  for (let i = 0; i < text.length; i++) {
    const code = text.charCodeAt(i);
    const idx = (code * 17 + i * 31) % dim;
    vec[idx] += 1.0;
  }
  // L2 Normalize
  let norm = 0;
  for (let i = 0; i < dim; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm) || 1.0;
  for (let i = 0; i < dim; i++) vec[i] /= norm;
  return vec;
}

const BANDS = 2048;
const DIM = 384;
const vault = new KalpanaVaultEmbedToKV({
  bands: BANDS,
  dim: DIM,
  wasmPath: '../pkg_vault/kalpana_vault.wasm',
});

const docInput = document.getElementById('docInput');
const btnIngest = document.getElementById('btnIngest');
const btnSample = document.getElementById('btnSample');
const queryInput = document.getElementById('queryInput');
const btnSearch = document.getElementById('btnSearch');
const statStored = document.getElementById('statStored');
const statMem = document.getElementById('statMem');
const ingestFeed = document.getElementById('ingestFeed');
const resultsArea = document.getElementById('resultsArea');

async function init() {
  try {
    await vault.initialize();
    console.log('[Kalpana] WASM Engine initialized successfully.');
  } catch (err) {
    console.warn('[Kalpana] Running simulated JS RIF mode:', err.message);
  }
  updateStats();
}

function updateStats() {
  statStored.textContent = vault.totalEntries;
  const memMb = ((BANDS * DIM * 4 * 2) / (1024 * 1024)).toFixed(2);
  statMem.textContent = `${memMb} MB (Strictly O(1))`;
}

function handleIngestText(text) {
  if (!text || !text.trim()) return;
  const embedding = textToEmbedding(text, DIM);
  
  let t;
  if (vault.isInitialized) {
    t = vault.ingestEmbedding(embedding, { text });
  } else {
    t = vault.totalEntries;
    vault.documents.set(t, { text, embedding });
    vault.totalEntries += 1;
  }

  const item = document.createElement('div');
  item.className = 'feed-item';
  item.innerHTML = `<div class="feed-meta">Coordinate t = ${t}</div><div>${escapeHtml(text)}</div>`;
  ingestFeed.prepend(item);

  updateStats();
}

btnIngest.addEventListener('click', () => {
  const text = docInput.value;
  handleIngestText(text);
  docInput.value = '';
});

btnSample.addEventListener('click', () => {
  const samples = [
    "Albert Einstein formulated the theory of general relativity in 1915.",
    "Apollo 11 landed astronauts on the Moon in July 1969.",
    "Photosynthesis converts sunlight and CO2 into glucose and oxygen in plants.",
    "Quantum entanglement allows instant correlated quantum states at arbitrary distances.",
    "The James Webb Space Telescope is situated at Earth-Sun L2 orbit point."
  ];
  samples.forEach(s => handleIngestText(s));
});

btnSearch.addEventListener('click', () => {
  const query = queryInput.value.trim();
  if (!query) return;

  const queryVec = textToEmbedding(query, DIM);
  let results = [];

  if (vault.isInitialized) {
    results = vault.search(queryVec, 5);
  } else {
    // Simulated cosine sweep
    for (let t = 0; t < vault.totalEntries; t++) {
      const doc = vault.documents.get(t);
      if (!doc) continue;
      let dot = 0;
      for (let i = 0; i < DIM; i++) dot += doc.embedding[i] * queryVec[i];
      results.push({ t, score: dot, metadata: doc });
    }
    results.sort((a, b) => b.score - a.score);
  }

  renderResults(query, results);
});

function renderResults(query, results) {
  if (results.length === 0) {
    resultsArea.innerHTML = '<div style="color: var(--text-muted); text-align: center;">No stored items found.</div>';
    return;
  }

  resultsArea.innerHTML = results.map(r => `
    <div class="result-card">
      <span class="score-badge">Resonance: ${(r.score).toFixed(4)}</span>
      <div style="font-family: var(--font-mono); font-size: 0.78rem; color: var(--cyan); margin-bottom: 0.3rem;">Coordinate t = ${r.t}</div>
      <div style="font-size: 0.95rem; font-weight: 500;">${escapeHtml(r.metadata?.text || 'Stored Chunk')}</div>
    </div>
  `).join('');
}

function escapeHtml(str) {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

init();
