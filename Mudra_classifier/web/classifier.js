// Mudra classifier: same normalisation as features.py, MLP forward pass from model.json.
const N_HANDS = 2, N_LM = 21, PALM = 9;

export function landmarksToRaw(hands) {
  const raw = new Float32Array(N_HANDS * N_LM * 3);
  hands.slice(0, N_HANDS).forEach((hand, h) => {
    const w = hand[0];
    hand.forEach((lm, i) => {
      const o = (h * N_LM + i) * 3;
      raw[o] = lm.x - w.x; raw[o + 1] = lm.y - w.y; raw[o + 2] = lm.z - w.z;
    });
  });
  return raw;
}

export function normalize(raw) {
  const out = new Float32Array(raw.length);
  for (let h = 0; h < N_HANDS; h++) {
    const b = h * N_LM * 3;
    const palm = Math.hypot(raw[b + PALM * 3], raw[b + PALM * 3 + 1]);
    if (palm < 1e-6) continue;
    for (let i = 0; i < N_LM * 3; i++) out[b + i] = raw[b + i] / palm;
  }
  return out;
}

export class MudraModel {
  constructor(json) {
    this.classes = json.classes;
    this.mean = Float32Array.from(json.mean);
    this.scale = Float32Array.from(json.scale);
    this.layers = json.layers.map(({ W, b }) => ({
      rows: W.length, cols: W[0].length,
      W: Float32Array.from(W.flat()), b: Float32Array.from(b),
    }));
  }
  static async load(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`Could not load ${url} (${r.status})`);
    return new MudraModel(await r.json());
  }
  // raw: 126 wrist-relative values -> probabilities per class
  predict(raw) {
    const f = normalize(raw);
    let x = new Float32Array(f.length);
    for (let i = 0; i < f.length; i++) x[i] = (f[i] - this.mean[i]) / this.scale[i];
    this.layers.forEach((L, li) => {
      const y = Float32Array.from(L.b);
      for (let i = 0; i < L.rows; i++) {
        const xi = x[i]; if (xi === 0) continue;
        const row = i * L.cols;
        for (let j = 0; j < L.cols; j++) y[j] += xi * L.W[row + j];
      }
      if (li < this.layers.length - 1) for (let j = 0; j < y.length; j++) y[j] = Math.max(0, y[j]);
      x = y;
    });
    const m = Math.max(...x);
    let s = 0;
    for (let j = 0; j < x.length; j++) { x[j] = Math.exp(x[j] - m); s += x[j]; }
    for (let j = 0; j < x.length; j++) x[j] /= s;
    return x;
  }
}
