async function instantiate(module, imports = {}) {
  const adaptedImports = {
    env: Object.setPrototypeOf({
      abort(message, fileName, lineNumber, columnNumber) {
        // ~lib/builtins/abort(~lib/string/String | null?, ~lib/string/String | null?, u32?, u32?) => void
        message = __liftString(message >>> 0);
        fileName = __liftString(fileName >>> 0);
        lineNumber = lineNumber >>> 0;
        columnNumber = columnNumber >>> 0;
        (() => {
          // @external.js
          throw Error(`${message} in ${fileName}:${lineNumber}:${columnNumber}`);
        })();
      },
      seed() {
        // ~lib/builtins/seed() => f64
        return (() => {
          // @external.js
          return Date.now() * Math.random();
        })();
      },
    }, Object.assign(Object.create(globalThis), imports.env || {})),
  };
  const { exports } = await WebAssembly.instantiate(module, adaptedImports);
  const memory = exports.memory || imports.env.memory;
  const adaptedExports = Object.setPrototypeOf({
    setState(re, im, o3, p4) {
      // assembly/index/setState(~lib/typedarray/Float32Array, ~lib/typedarray/Float32Array, ~lib/typedarray/Float32Array, ~lib/typedarray/Float32Array) => void
      re = __retain(__lowerTypedArray(Float32Array, 5, 2, re) || __notnull());
      im = __retain(__lowerTypedArray(Float32Array, 5, 2, im) || __notnull());
      o3 = __retain(__lowerTypedArray(Float32Array, 5, 2, o3) || __notnull());
      p4 = __lowerTypedArray(Float32Array, 5, 2, p4) || __notnull();
      try {
        exports.setState(re, im, o3, p4);
      } finally {
        __release(re);
        __release(im);
        __release(o3);
      }
    },
    getStateRe() {
      // assembly/index/getStateRe() => ~lib/typedarray/Float32Array
      return __liftTypedArray(Float32Array, exports.getStateRe() >>> 0);
    },
    getStateIm() {
      // assembly/index/getStateIm() => ~lib/typedarray/Float32Array
      return __liftTypedArray(Float32Array, exports.getStateIm() >>> 0);
    },
    getStateO3() {
      // assembly/index/getStateO3() => ~lib/typedarray/Float32Array
      return __liftTypedArray(Float32Array, exports.getStateO3() >>> 0);
    },
    getStateP4() {
      // assembly/index/getStateP4() => ~lib/typedarray/Float32Array
      return __liftTypedArray(Float32Array, exports.getStateP4() >>> 0);
    },
    writeRIF(t, emb) {
      // assembly/index/writeRIF(f32, ~lib/typedarray/Float32Array) => void
      emb = __lowerTypedArray(Float32Array, 5, 2, emb) || __notnull();
      exports.writeRIF(t, emb);
    },
    readRIF(t, qV) {
      // assembly/index/readRIF(f32, ~lib/typedarray/Float32Array) => f32
      qV = __lowerTypedArray(Float32Array, 5, 2, qV) || __notnull();
      return exports.readRIF(t, qV);
    },
    getVersion() {
      // assembly/index/getVersion() => ~lib/string/String
      return __liftString(exports.getVersion() >>> 0);
    },
  }, exports);
  function __liftString(pointer) {
    if (!pointer) return null;
    const
      end = pointer + new Uint32Array(memory.buffer)[pointer - 4 >>> 2] >>> 1,
      memoryU16 = new Uint16Array(memory.buffer);
    let
      start = pointer >>> 1,
      string = "";
    while (end - start > 1024) string += String.fromCharCode(...memoryU16.subarray(start, start += 1024));
    return string + String.fromCharCode(...memoryU16.subarray(start, end));
  }
  function __liftTypedArray(constructor, pointer) {
    if (!pointer) return null;
    return new constructor(
      memory.buffer,
      __getU32(pointer + 4),
      __dataview.getUint32(pointer + 8, true) / constructor.BYTES_PER_ELEMENT
    ).slice();
  }
  function __lowerTypedArray(constructor, id, align, values) {
    if (values == null) return 0;
    const
      length = values.length,
      buffer = exports.__pin(exports.__new(length << align, 1)) >>> 0,
      header = exports.__new(12, id) >>> 0;
    __setU32(header + 0, buffer);
    __dataview.setUint32(header + 4, buffer, true);
    __dataview.setUint32(header + 8, length << align, true);
    new constructor(memory.buffer, buffer, length).set(values);
    exports.__unpin(buffer);
    return header;
  }
  const refcounts = new Map();
  function __retain(pointer) {
    if (pointer) {
      const refcount = refcounts.get(pointer);
      if (refcount) refcounts.set(pointer, refcount + 1);
      else refcounts.set(exports.__pin(pointer), 1);
    }
    return pointer;
  }
  function __release(pointer) {
    if (pointer) {
      const refcount = refcounts.get(pointer);
      if (refcount === 1) exports.__unpin(pointer), refcounts.delete(pointer);
      else if (refcount) refcounts.set(pointer, refcount - 1);
      else throw Error(`invalid refcount '${refcount}' for reference '${pointer}'`);
    }
  }
  function __notnull() {
    throw TypeError("value must not be null");
  }
  let __dataview = new DataView(memory.buffer);
  function __setU32(pointer, value) {
    try {
      __dataview.setUint32(pointer, value, true);
    } catch {
      __dataview = new DataView(memory.buffer);
      __dataview.setUint32(pointer, value, true);
    }
  }
  function __getU32(pointer) {
    try {
      return __dataview.getUint32(pointer, true);
    } catch {
      __dataview = new DataView(memory.buffer);
      return __dataview.getUint32(pointer, true);
    }
  }
  return adaptedExports;
}
export const {
  memory,
  initEngine,
  setState,
  getStateRe,
  getStateIm,
  getStateO3,
  getStateP4,
  writeRIF,
  readRIF,
  getVersion,
} = await (async url => instantiate(
  await (async () => {
    const isNodeOrBun = typeof process != "undefined" && process.versions != null && (process.versions.node != null || process.versions.bun != null);
    if (isNodeOrBun) { return globalThis.WebAssembly.compile(await (await import("node:fs/promises")).readFile(url)); }
    else { return await globalThis.WebAssembly.compileStreaming(globalThis.fetch(url)); }
  })(), {
  }
))(new URL("kalpana_vault.wasm", import.meta.url));
