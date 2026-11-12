# Mudra Classifier

Recognises 50 Bharatanatyam mudras from hand landmarks (MediaPipe Hand Landmarker + a small neural network).

## Option 1: Browser (recommended, works on Mac without camera hassles)

From this folder run:

    python3 -m http.server 8000

Then open http://localhost:8000/web/ in Chrome, Safari or Edge, click "Start camera" and allow access. Everything runs locally in the browser. The camera page must be served over http://localhost or https (opening index.html as a file will not get camera access).

To share it, enable GitHub Pages for the repo and open `https://<user>.github.io/<repo>/Mudra_classifier/web/`.

## Option 2: Desktop Python app

    pip install -r requirements.txt
    python train_model.py      # about 1 minute, creates mudra_model.p and web/model.json
    python app.py              # or: python app.py --camera 1

On macOS, if the camera does not open: System Settings > Privacy & Security > Camera, enable the app you run Python from (Terminal, iTerm, VS Code, PyCharm) and fully restart it. Close Zoom, FaceTime and Photo Booth. If an iPhone shows up as a Continuity Camera, try `--camera 1`.

## Files

- `features.py` shared feature pipeline (normalisation + augmentation)
- `train_model.py` trains the model and exports it for Python and the browser
- `app.py` desktop webcam app
- `web/` browser app (`classifier.js` mirrors `features.py`)
- `collect_data.py` rebuilds `mudra_data.csv` from the image dataset folder
