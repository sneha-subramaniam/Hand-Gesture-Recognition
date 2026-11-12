// Mudra recogniser for the browser. Mirrors features.py + recognizer.py exactly.
const PALM = 9;

// MediaPipe landmarks of one hand -> array of 21 [x, y, z] in pixel units
export function toPixels(hand, width, height) {
  return hand.map((lm) => [lm.x * width, lm.y * height, lm.z * width]);
}

const palmLength = (h) => Math.hypot(h[PALM][0] - h[0][0], h[PALM][1] - h[0][1]);

export function handFeatures(h) {
  const p = palmLength(h), out = new Float32Array(63);
  h.forEach((lm, i) => { for (let k = 0; k < 3; k++) out[i * 3 + k] = (lm[k] - h[0][k]) / p; });
  return out;
}

export function handGap(a, b) {
  let m = Infinity;
  for (const p of a) for (const q of b) m = Math.min(m, Math.hypot(p[0] - q[0], p[1] - q[1]));
  return m / ((palmLength(a) + palmLength(b)) / 2);
}

export const orderPair = (a, b) => (a[0][0] > b[0][0] ? [b, a] : [a, b]);

export function pairFeatures(a, b) {
  [a, b] = orderPair(a, b);
  const pa = palmLength(a), pb = palmLength(b), mean = (pa + pb) / 2;
  const out = new Float32Array(130);
  out.set(handFeatures(a), 0);
  out.set(handFeatures(b), 63);
  out[126] = (b[0][0] - a[0][0]) / mean;
  out[127] = (b[0][1] - a[0][1]) / mean;
  out[128] = Math.log(pb / pa);
  out[129] = handGap(a, b);
  return out;
}

class MLP {
  constructor(j) {
    this.classes = j.classes;
    this.mean = Float32Array.from(j.mean);
    this.scale = Float32Array.from(j.scale);
    this.layers = j.layers.map(({ W, b }) => ({ rows: W.length, cols: W[0].length, W: Float32Array.from(W.flat()), b: Float32Array.from(b) }));
  }
  predict(f) {
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
    const m = Math.max(...x); let s = 0;
    for (let j = 0; j < x.length; j++) { x[j] = Math.exp(x[j] - m); s += x[j]; }
    for (let j = 0; j < x.length; j++) x[j] /= s;
    return x;
  }
}

export class Recognizer {
  constructor(j) {
    this.one = new MLP(j.one);
    this.two = new MLP(j.two);
    this.touchGap = j.touch_gap;
    this.classes = this.one.classes;
    const dbl = new Set(j.double);
    this.isDouble = this.classes.map((c) => dbl.has(c));
    this.twoIndex = this.two.classes.map((c) => this.classes.indexOf(c));
  }
  static async load(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`Could not load ${url} (${r.status})`);
    return new Recognizer(await r.json());
  }
  _one(h, allowed) {
    const p = this.one.predict(handFeatures(h));
    let s = 0;
    for (let i = 0; i < p.length; i++) { if (!allowed(i)) p[i] = 0; s += p[i]; }
    for (let i = 0; i < p.length; i++) p[i] /= Math.max(s, 1e-9);
    return p;
  }
  _two(a, b) {
    const q = this.two.predict(pairFeatures(a, b)), p = new Float32Array(this.classes.length);
    this.twoIndex.forEach((ci, k) => (p[ci] = q[k]));
    return p;
  }
  // hands: arrays from toPixels(). mode: "auto" | "single" | "double".
  // Returns [{ who, probs }] with who = "Both hands" | "Hand" | "Right hand" | "Left hand".
  predict(hands, mode = "auto") {
    const single = (i) => !this.isDouble[i];
    if (hands.length >= 2) {
      const [a, b] = orderPair(hands[0], hands[1]);
      if (mode !== "single" && handGap(a, b) < this.touchGap) return [{ who: "Both hands", probs: this._two(a, b) }];
      if (mode === "double") return [];
      // a is further left in the camera image = the dancer's right hand
      return [{ who: "Right hand", probs: this._one(a, single) }, { who: "Left hand", probs: this._one(b, single) }];
    }
    if (hands.length === 1) {
      const allowed = { auto: () => true, single, double: (i) => this.isDouble[i] }[mode];
      return [{ who: "Hand", probs: this._one(hands[0], allowed) }];
    }
    return [];
  }
}
