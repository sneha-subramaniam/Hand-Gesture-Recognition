# Mudra Classifier

Recognises 50 Bharatanatyam mudras (29 single-hand, 21 two-hand) from hand landmarks, using MediaPipe Hand Landmarker plus two small neural networks.

## How it decides single vs two-hand

- Two hands close together (within 1.5 palm lengths): two-hand model, which also uses how the hands sit relative to each other.
- Two hands far apart: the other hand is just resting in view, so each hand is classified on its own against single-hand mudras.
- One hand: one-hand model over all mudras. Overlapping two-hand mudras such as Matsya or Varaha often look like a single hand to the detector, so they stay possible.

Use the mode menu ("Single-hand mudras only" or "Two-hand mudras only") when practising one group. That removes most remaining mix-ups.

## Teaching your own version

The training photos are close-ups on a green screen and mostly show one way of holding each mudra (for example Kartarimukha with ring and little fingers folded flat under the thumb, back of the hand to the camera). If you hold a mudra correctly but differently, the model may not know it.

On the webpage, use the "Teach your version" panel: pick the mudra, press Record, hold it for 3 seconds. From then on, frames that closely match your recording use your label, and the page shows "Matched your taught version". Recordings stay in that browser only. Teach a mudra 2 or 3 times at slightly different angles for best results. "Export" saves them as `my_mudras.json`, and "Import" loads them in another browser.

To build your recordings into the model itself:

    python train_model.py --extra my_mudras.json

## Option 1: Browser (recommended)

Live on Vercel, or locally from this folder:

    python3 -m http.server 8000

Then open http://localhost:8000/web/ and click "Start camera". Everything runs in the browser and no video leaves the device.

Vercel settings: Framework Preset "Other", no build command, Root Directory either the repo root or `Mudra_classifier`. The included `vercel.json` sends `/` to the webpage.

## Option 2: Desktop Python app

    pip install -r requirements.txt
    python train_model.py          # about 1 minute, creates mudra_model.p and web/model.json
    python app.py                  # options: --camera 1, --mode single|double, --threshold 0.6

On macOS, if the camera does not open: System Settings > Privacy & Security > Camera, enable the app you run Python from (Terminal, iTerm, VS Code, PyCharm) and fully restart it. Close Zoom, FaceTime and Photo Booth. If an iPhone shows up as a Continuity Camera, try `--camera 1`.

## Retraining from the images

    git clone --depth 1 https://github.com/jisharajr/Bharatanatyam-Mudra-Dataset
    python collect_data.py --data ./Bharatanatyam-Mudra-Dataset   # about 3 minutes, creates mudra_landmarks.npz
    python train_model.py

## Files

- `features.py` feature maths shared by training and both apps
- `recognizer.py` picks the one-hand or two-hand model for each frame
- `collect_data.py` images to `mudra_landmarks.npz` (pixel-space landmarks of up to 2 hands)
- `train_model.py` trains both models, prints held-out accuracy, exports `mudra_model.p` and `web/model.json`
- `app.py` desktop webcam app
- `web/` browser app (`classifier.js` mirrors `features.py` and `recognizer.py`, plus taught-sample matching)
- `mudra_data.csv` older landmark export, no longer used by training (it lacks hand positions)
