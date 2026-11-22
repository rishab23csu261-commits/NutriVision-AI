# Running NutriVision AI on Windows

This project is tested for **Windows 10/11 (64-bit)** with **Python 3.11**.
TensorFlow is CPU-only on Windows; no GPU is required.

The trained food classifier, calibration file, ANN, sample photos, and nutrition
database are already included in the repository. **You do not need the image
dataset to run the app.** The dataset is only for retraining. No `.env` file or
API key is required.

---

## Option A — one-click launcher

1. Install **64-bit Python 3.11** from
   <https://www.python.org/downloads/windows/>.
   Tick **“Add python.exe to PATH”** during setup.
2. Install the **Microsoft Visual C++ Redistributable 2015–2022 (x64)**:
   <https://aka.ms/vs/17/release/vc_redist.x64.exe>
   (TensorFlow needs it.)
3. Clone or download the repository.
4. Double-click **`launch.bat`** in the project folder.

`launch.bat` will:

- find Python 3.11,
- create `.venv`,
- install `requirements.txt`,
- check the model files,
- start the app.

The first run can take 5–15 minutes because TensorFlow is large. Later runs
start in seconds. When the app is ready, open:

**http://localhost:8501**

---

## Option B — manual PowerShell / Command Prompt

```bat
cd C:\path\to\NutriVision-AI

py -3.11 -m venv .venv
.venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m streamlit run app.py
```

If `py` is not available, use `python` instead, but make sure it is 64-bit
Python 3.11:

```bat
python --version
python -c "import struct; print(struct.calcsize('P') * 8, 'bit')"
```

You want `Python 3.11.x` and `64 bit`.

---

## First-run checklist

1. The app opens at **http://localhost:8501**.
2. Click **Analyze → Pizza Margherita → Analyze meal**.
3. You should see a result with roughly **99% confidence** and nutrition values.
4. Open **Model** and confirm **91.10% top-1 / 98.58% top-3**.
5. Open **Diary** and confirm the saved meal appears.

---

## Optional: retrain the image classifier

Only needed if you change the dataset or model. The app works without this.

```bat
.venv\Scripts\activate
python training\download_dataset.py
python training\train_mobilenet_linear.py
python training\evaluate.py --model mobilenet --dataset real
```

Notes:

- `download_dataset.py` downloads about **600 MB** and respects the official
  train/validation/test splits.
- `train_mobilenet_linear.py` caches features in `data\features\`; retraining
  the linear classifier after that takes seconds.
- The optional custom CNN is **not** included (its checkpoint is over 100 MB).
  The Model page shows a clear “not trained” message for it.

---

## Troubleshooting

### “Python was not found” / `py` is not recognized
Reinstall Python 3.11 and tick **Add python.exe to PATH**. Or use the full path
to `python.exe` in `launch.bat`.

### TensorFlow fails to install
- Confirm you are on 64-bit Python **3.11**.
- Install the Visual C++ Redistributable linked above.
- Try a fresh environment: delete `.venv` and run `launch.bat` again.

### `streamlit` is not recognized
Always use:

```bat
python -m streamlit run app.py
```

or let `launch.bat` call `.venv\Scripts\python.exe -m streamlit run app.py`.

### Port 8501 is already in use
```bat
python -m streamlit run app.py --server.port 8502
```
Then open <http://localhost:8502>.

### Browser shows an old version
Press **Ctrl+F5** to hard-refresh. Stop and restart the terminal if you changed
Python files.

### App opens but food recognition is weak
Check `saved_models\mobilenet_model.keras` exists. If it is missing, the app
falls back to a simpler path. The checkpoint is normally included with the
repository.

### OneDrive / spaces in the folder path
Windows handles spaces, but OneDrive sync can lock the SQLite diary file. If
you see database errors, move the project to a local folder such as:

```text
C:\NutriVision-AI
```

### CPU is slow on first analysis
TensorFlow loads the model on the first request. The first analysis can take a
few seconds; later ones are faster.

---

## What is included vs. optional

| Item | Included? | Needed to run? |
|---|---|---|
| MobileNetV2 food classifier | Yes (`saved_models/mobilenet_model.keras`) | Yes |
| Non-food calibration | Yes (`saved_models/ood_calibration.npz`) | Recommended |
| Nutrition ANN | Yes (`saved_models/ann_model.keras`) | No, model page only |
| Nutrition database | Yes (`data/nutrition_database.json`) | Yes |
| Sample photos | Yes (`assets/samples/`) | No |
| Training images | No (`data/food_images/`) | Only for retraining |
| Custom 4-block CNN | No | No |
