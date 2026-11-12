// Mudra recogniser for the browser. Mirrors features.py + recognizer.py exactly,
// plus "taught" samples recorded by the user (nearest-neighbour blending).
const PALM = 9;
const CHAINS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]];
const TIPS = [4, 8, 12, 16, 20];

// MediaPipe landmarks of one hand -> array of 21 [x, y, z] in pixel units
export function toPixels(hand, width, height) {
  return hand.map((lm) => [lm.x * width, lm.y * height, lm.z * width]);
}

const palmLength = (h) => Math.hypot(h[PALM][0] - h[0][0], h[PALM][1] - h[0][1]);

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const norm3 = (v) => Math.hypot(v[0], v[1], v[2]);
const dist2 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

export function handFeatures(h) {
  const p = palmLength(h), out = new Float32Array(93);
  h.forEach((lm, i) => { for (let k = 0; k < 3; k++) out[i * 3 + k] = (lm[k] - h[0][k]) / p; });
  let o = 63;
  for (const ch of CHAINS) for (let j = 1; j < 4; j++) {
    const u = sub(h[ch[j]], h[ch[j - 1]]), v = sub(h[ch[j + 1]], h[ch[j]]);
    out[o++] = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (norm3(u) * norm3(v) + 1e-6);
  }
  for (let i = 0; i < 5; i++) {
    out[o++] = dist2(h[TIPS[i]], h[0]) / p;
    for (let j = i + 1; j < 5; j++) out[o++] = dist2(h[TIPS[i]], h[TIPS[j]]) / p;
  }
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
  const out = new Float32Array(190);
  out.set(handFeatures(a), 0);
  out.set(handFeatures(b), 93);
  out[186] = (b[0][0] - a[0][0]) / mean;
  out[187] = (b[0][1] - a[0][1]) / mean;
  out[188] = Math.log(pb / pa);
  out[189] = handGap(a, b);
  return out;
}

class MLP {
  constructor(j) {
    this.classes = j.classes;
    this.mean = Float32Array.from(j.mean);
    this.scale = Float32Array.from(j.scale);
    this.layers = j.layers.map(({ W, b }) => ({ rows: W.length, cols: W[0].length, W: Float32Array.from(W.flat()), b: Float32Array.from(b) }));
  }
  standardize(f) {
    const x = new Float32Array(f.length);
    for (let i = 0; i < f.length; i++) x[i] = (f[i] - this.mean[i]) / this.scale[i];
    return x;
  }
  predict(f) {
    let x = this.standardize(f);
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
    this.tau = j.personal_tau || { one: 2.0, two: 2.8 };
    this.taught = { one: [], two: [] }; // { label index, z: standardised features }
  }

  // Which model a set of hands goes to, and its features (null if none applies)
  route(hands, mode = "auto") {
    if (hands.length >= 2) {
      const [a, b] = orderPair(hands[0], hands[1]);
      if (mode !== "single" && handGap(a, b) < this.touchGap) return { kind: "two", f: pairFeatures(a, b) };
      return null; // far-apart hands: ambiguous which one is the mudra
    }
    if (hands.length === 1) return { kind: "one", f: handFeatures(hands[0]) };
    return null;
  }

  // samples: [{ label, hands: [[[x,y,z] x21], ...] }] in pixel units
  setTaught(samples) {
    this.taught = { one: [], two: [] };
    for (const s of samples) {
      const ci = this.classes.indexOf(s.label);
      const r = ci >= 0 && this.route(s.hands);
      if (r) this.taught[r.kind].push({ ci, z: this[r.kind].standardize(r.f) });
    }
  }

  // Blend model probabilities with the nearest taught samples
  _personal(kind, f, p, allowed) {
    const list = this.taught[kind].filter((t) => allowed(t.ci));
    if (!list.length) return { p, match: 0 };
    const z = this[kind].standardize(f), tau = this.tau[kind];
    const near = list.map((t) => {
      let d = 0;
      for (let i = 0; i < z.length; i++) d += (z[i] - t.z[i]) ** 2;
      return { ci: t.ci, d: Math.sqrt(d) };
    }).sort((a, b) => a.d - b.d).slice(0, 5);
    const w = Math.exp(-((near[0].d / tau) ** 2));
    const knn = new Float32Array(p.length);
    let s = 0;
    for (const n of near) { const k = Math.exp(-((n.d / tau) ** 2)); knn[n.ci] += k; s += k; }
    const out = new Float32Array(p.length);
    for (let i = 0; i < p.length; i++) out[i] = (1 - w) * p[i] + w * (s > 0 ? knn[i] / s : 0);
    return { p: out, match: w };
  }
  static async load(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`Could not load ${url} (${r.status})`);
    return new Recognizer(await r.json());
  }
  _one(h, allowed) {
    const f = handFeatures(h), p = this.one.predict(f);
    let s = 0;
    for (let i = 0; i < p.length; i++) { if (!allowed(i)) p[i] = 0; s += p[i]; }
    for (let i = 0; i < p.length; i++) p[i] /= Math.max(s, 1e-9);
    return this._personal("one", f, p, allowed);
  }
  _two(a, b) {
    const f = pairFeatures(a, b), q = this.two.predict(f), p = new Float32Array(this.classes.length);
    this.twoIndex.forEach((ci, k) => (p[ci] = q[k]));
    return this._personal("two", f, p, () => true);
  }
  // hands: arrays from toPixels(). mode: "auto" | "single" | "double".
  // Returns [{ who, probs, match }] with who = "Both hands" | "Hand" | "Right hand" | "Left hand"
  // and match = 0..1, how strongly a taught sample decided the result.
  predict(hands, mode = "auto") {
    const single = (i) => !this.isDouble[i];
    if (hands.length >= 2) {
      const [a, b] = orderPair(hands[0], hands[1]);
      const wrap = (who, r) => ({ who, probs: r.p, match: r.match });
      if (mode !== "single" && handGap(a, b) < this.touchGap) return [wrap("Both hands", this._two(a, b))];
      if (mode === "double") return [];
      // a is further left in the camera image = the dancer's right hand
      return [wrap("Right hand", this._one(a, single)), wrap("Left hand", this._one(b, single))];
    }
    if (hands.length === 1) {
      const allowed = { auto: () => true, single, double: (i) => this.isDouble[i] }[mode];
      const r = this._one(hands[0], allowed);
      return [{ who: "Hand", probs: r.p, match: r.match }];
    }
    return [];
  }
}
