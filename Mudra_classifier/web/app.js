import { FilesetResolver, HandLandmarker, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
import { MudraModel, landmarksToRaw } from "./classifier.js";

const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const LANDMARKER_LOCAL = "../hand_landmarker.task";
const LANDMARKER_CDN = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const SMOOTH = 8; // frames of probability averaging

const $ = (id) => document.getElementById(id);
const video = $("video"), canvas = $("canvas"), ctx = canvas.getContext("2d");
let landmarker, model, stream, history = [], threshold = 0.6, lastTs = -1, frames = 0, fpsT = performance.now(), fps = 0;

$("thr").oninput = (e) => { threshold = e.target.value / 100; $("thrv").textContent = e.target.value + "%"; };

async function init() {
  model = await MudraModel.load("model.json");
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

function predict(hands) {
  // Score both hands together and each hand alone. Use the combined result
  // unless a single hand is clearly more confident (e.g. a one-hand mudra with
  // the other hand resting in view).
  const cands = [model.predict(landmarksToRaw(hands))];
  if (hands.length > 1) hands.forEach((h) => cands.push(model.predict(landmarksToRaw([h]))));
  const best = (p) => Math.max(...p);
  let p = cands[0];
  for (const c of cands.slice(1)) if (best(c) > best(p) + 0.25) p = c;
  return p;
}

function render(probs) {
  const order = [...probs.keys()].sort((a, b) => probs[b] - probs[a]).slice(0, 3);
  const top = probs[order[0]];
  const el = $("label");
  if (top >= threshold) { el.textContent = model.classes[order[0]]; el.className = "big ok"; }
  else { el.textContent = "Not sure yet"; el.className = "big wait"; }
  $("top").innerHTML = order.map((i) => `<div class="row"><span>${model.classes[i]}</span><span>${Math.round(probs[i] * 100)}%</span></div><div class="bar"><i style="width:${probs[i] * 100}%"></i></div>`).join("");
}

function loop() {
  if (!stream?.active) return;
  if (video.readyState >= 2) {
    if (canvas.width !== video.videoWidth) { canvas.width = video.videoWidth; canvas.height = video.videoHeight; }
    let ts = performance.now(); if (ts <= lastTs) ts = lastTs + 1; lastTs = ts;
    const res = landmarker.detectForVideo(video, ts);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (res.landmarks.length) {
      const du = new DrawingUtils(ctx);
      res.landmarks.forEach((lm) => {
        du.drawConnectors(lm, HandLandmarker.HAND_CONNECTIONS, { color: "#ec8a5f", lineWidth: 3 });
        du.drawLandmarks(lm, { color: "#fff", radius: 3 });
      });
      history.push(predict(res.landmarks));
      if (history.length > SMOOTH) history.shift();
      const avg = new Float32Array(model.classes.length);
      history.forEach((p) => p.forEach((v, i) => (avg[i] += v / history.length)));
      render(avg);
    } else {
      history = [];
      $("label").textContent = "Show your hand(s)"; $("label").className = "big wait"; $("top").innerHTML = "";
    }
    frames++;
    const now = performance.now();
    if (now - fpsT > 1000) { fps = frames; frames = 0; fpsT = now; }
    $("stat").textContent = `${res.landmarks.length} hand(s) detected, ${fps} fps`;
  }
  requestAnimationFrame(loop);
}
