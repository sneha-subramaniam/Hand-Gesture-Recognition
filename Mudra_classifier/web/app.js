import { FilesetResolver, HandLandmarker, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
import { Recognizer, toPixels } from "./classifier.js";

const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const LANDMARKER_LOCAL = "../hand_landmarker.task";
const LANDMARKER_CDN = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const SMOOTH = 8; // frames of probability averaging

const $ = (id) => document.getElementById(id);
const video = $("video"), canvas = $("canvas"), ctx = canvas.getContext("2d");
let landmarker, recognizer, stream, threshold = 0.6, mode = "auto", lastTs = -1, frames = 0, fpsT = performance.now(), fps = 0;
let history = new Map(); // who -> recent probability vectors

$("thr").oninput = (e) => { threshold = e.target.value / 100; $("thrv").textContent = e.target.value + "%"; };
$("mode").onchange = (e) => { mode = e.target.value; history.clear(); };

async function init() {
  recognizer = await Recognizer.load("model.json");
  const vision = await FilesetResolver.forVisionTasks(WASM);
  const opts = (path, delegate) => ({
    baseOptions: { modelAssetPath: path, delegate },
    runningMode: "VIDEO", numHands: 2,
    minHandDetectionConfidence: 0.4, minHandPresenceConfidence: 0.4, minTrackingConfidence: 0.4,
  });
  for (const [path, delegate] of [[LANDMARKER_LOCAL, "GPU"], [LANDMARKER_CDN, "GPU"], [LANDMARKER_CDN, "CPU"]]) {
    try { landmarker = await HandLandmarker.createFromOptions(vision, opts(path, delegate)); return; }
    catch (e) { console.warn("Landmarker load failed", path, delegate, e); }
  }
  throw new Error("Could not load the hand landmark model.");
}
const ready = init();

async function listCameras() {
  const cams = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
  const sel = $("cams");
  sel.innerHTML = cams.map((c, i) => `<option value="${c.deviceId}">${c.label || "Camera " + (i + 1)}</option>`).join("");
  sel.hidden = cams.length < 2;
  if (stream) sel.value = stream.getVideoTracks()[0].getSettings().deviceId;
}

async function startCamera(deviceId) {
  stream?.getTracks().forEach((t) => t.stop());
  $("overlay").hidden = false; $("overlay").textContent = "Starting camera...";
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { deviceId: deviceId ? { exact: deviceId } : undefined, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false,
    });
  } catch (e) {
    $("overlay").textContent = e.name === "NotAllowedError"
      ? "Camera access was blocked. Allow it in the browser address bar (and in macOS System Settings > Privacy & Security > Camera for your browser), then try again."
      : `Could not open the camera (${e.name}). Close other apps using it (Zoom, FaceTime, Photo Booth) and try again.`;
    return;
  }
  video.srcObject = stream;
  await video.play();
  await listCameras();
  $("overlay").textContent = "Loading hand model...";
  try { await ready; } catch (e) { $("overlay").textContent = e.message; return; }
  $("overlay").hidden = true;
  $("start").textContent = "Restart camera";
  requestAnimationFrame(loop);
}
$("start").onclick = () => startCamera($("cams").value || undefined);
$("cams").onchange = (e) => startCamera(e.target.value);

function smooth(who, probs) {
  const h = history.get(who) || [];
  h.push(probs); if (h.length > SMOOTH) h.shift();
  history.set(who, h);
  const avg = new Float32Array(probs.length);
  h.forEach((p) => p.forEach((v, i) => (avg[i] += v / h.length)));
  return avg;
}

function block(who, probs, showWho) {
  const order = [...probs.keys()].sort((a, b) => probs[b] - probs[a]).slice(0, 3);
  const names = recognizer.classes, ok = probs[order[0]] >= threshold;
  return `${showWho ? `<div class="who">${who}</div>` : ""}
    <div class="big ${ok ? "ok" : "wait"}">${ok ? names[order[0]] : "Not sure yet"}</div>
    ${order.map((i) => `<div class="row"><span>${names[i]}</span><span>${Math.round(probs[i] * 100)}%</span></div><div class="bar"><i style="width:${probs[i] * 100}%"></i></div>`).join("")}`;
}

function loop() {
  if (!stream?.active) return;
  if (video.readyState >= 2) {
    const W = video.videoWidth, H = video.videoHeight;
    if (canvas.width !== W) { canvas.width = W; canvas.height = H; }
    let ts = performance.now(); if (ts <= lastTs) ts = lastTs + 1; lastTs = ts;
    const res = landmarker.detectForVideo(video, ts);
    ctx.clearRect(0, 0, W, H);
    const du = new DrawingUtils(ctx);
    res.landmarks.forEach((lm) => {
      du.drawConnectors(lm, HandLandmarker.HAND_CONNECTIONS, { color: "#ec8a5f", lineWidth: 3 });
      du.drawLandmarks(lm, { color: "#fff", radius: 3 });
    });

    const out = recognizer.predict(res.landmarks.map((lm) => toPixels(lm, W, H)), mode);
    const seen = new Set(out.map((o) => o.who));
    for (const k of [...history.keys()]) if (!seen.has(k)) history.delete(k);
    let html;
    if (!res.landmarks.length) html = `<div class="big wait">Show your hand(s)</div>`;
    else if (!out.length) html = `<div class="big wait">Bring your hands together</div>`;
    else html = out.map((o) => block(o.who, smooth(o.who, o.probs), out.length > 1 || o.who === "Both hands")).join("");
    $("results").innerHTML = html;

    frames++;
    const now = performance.now();
    if (now - fpsT > 1000) { fps = frames; frames = 0; fpsT = now; }
    $("stat").textContent = `${res.landmarks.length} hand(s) detected, ${fps} fps`;
  }
  requestAnimationFrame(loop);
}
