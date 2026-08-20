/**
 * Kalpana Vault Embed-to-KV: JavaScript/WebAssembly client wrapper
 * Connects semantic vector embeddings to the client-side O(1) holographic RIF memory.
 */

export class KalpanaVaultEmbedToKV {
  /**
   * @param {Object} options
   * @param {number} [options.bands=2048] - Memory bandwidth capacity
   * @param {number} [options.dim=384] - Embedding dimensionality
   * @param {number} [options.kappa=10.0] - Holographic spread multiplier
   * @param {number} [options.minFreq=0.1] - Minimum frequency
   * @param {number} [options.maxFreq=10.0] - Maximum frequency
   * @param {string} [options.wasmPath='./kalpana_vault.wasm'] - Path or URL to wasm binary
   */
  constructor(options = {}) {
    this.bands = options.bands || 2048;
    this.dim = options.dim || 384;
    this.kappa = options.kappa || 10.0;
    this.minFreq = options.minFreq || 0.1;
    this.maxFreq = options.maxFreq || 10.0;
    this.wasmPath = options.wasmPath || './kalpana_vault.wasm';

    this.wasmModule = null;
    this.isInitialized = false;
    this.totalEntries = 0;
    this.documents = new Map(); // Store metadata / original text snippets mapped by temporal index t
  }

  /**
   * Loads the WASM engine and initializes the RIF matrix.
   */
  async initialize() {
    if (this.isInitialized) return;

    let wasmBytes;
    if (typeof window === 'undefined') {
      // Node.js environment
      const fs = await import('node:fs/promises');
      wasmBytes = await fs.readFile(this.wasmPath);
    } else {
      // Browser environment
      const response = await fetch(this.wasmPath);
      wasmBytes = await response.arrayBuffer();
    }

    const { instantiate } = await import('./kalpana_vault.js');
    const wasmCompiled = await WebAssembly.compile(wasmBytes);
    this.wasmModule = await instantiate(wasmCompiled);

    if (this.wasmModule.initEngine) {
      this.wasmModule.initEngine(this.bands, this.dim, this.kappa, this.minFreq, this.maxFreq);
    }

    this.isInitialized = true;
  }

  /**
   * Ingests a semantic embedding vector into the holographic KV memory at coordinate t.
   * @param {Float32Array|number[]} embedding - Float32Array vector matching this.dim
   * @param {Object} [metadata={}] - Optional metadata or raw text
   * @returns {number} The assigned temporal coordinate t
   */
  ingestEmbedding(embedding, metadata = {}) {
    if (!this.isInitialized) {
      throw new Error('Kalpana Vault is not initialized. Call initialize() first.');
    }

    const t = this.totalEntries;
    const floatArray = embedding instanceof Float32Array ? embedding : new Float32Array(embedding);

    if (floatArray.length !== this.dim) {
      throw new Error(`Vector dimension mismatch: expected ${this.dim}, received ${floatArray.length}`);
    }

    this.wasmModule.writeRIF(t, floatArray);
    this.documents.set(t, metadata);
    this.totalEntries += 1;
    return t;
  }

  /**
   * Queries the holographic memory against a query embedding vector.
   * Sweeps across stored temporal coordinates to find the highest resonance match.
   * @param {Float32Array|number[]} queryVector - Query embedding
   * @param {number} [topK=5] - Number of top results to return
   * @returns {Array<{t: number, score: number, metadata: any}>}
   */
  search(queryVector, topK = 5) {
    if (!this.isInitialized) {
      throw new Error('Kalpana Vault is not initialized. Call initialize() first.');
    }

    const floatArray = queryVector instanceof Float32Array ? queryVector : new Float32Array(queryVector);
    const results = [];

    for (let t = 0; t < this.totalEntries; t++) {
      const score = this.wasmModule.readRIF(t, floatArray);
      results.push({
        t,
        score,
        metadata: this.documents.get(t) || null,
      });
    }

    // Sort descending by resonance score
    results.sort((a, b) => b.score - a.score);
    return results.slice(0, topK);
  }

  /**
   * Gets memory statistics
   */
  getStats() {
    // 2 floats per band * dim (Real + Imaginary)
    const memoryBytes = this.bands * this.dim * 4 * 2;
    return {
      totalEntries: this.totalEntries,
      bands: this.bands,
      dim: this.dim,
      memoryFootprintBytes: memoryBytes,
      memoryFootprintMB: (memoryBytes / (1024 * 1024)).toFixed(3),
    };
  }
}
