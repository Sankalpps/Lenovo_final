# NVMe Drive Health Intelligence System

A predictive maintenance tool that detects NVMe drive degradation **before failure occurs** using a trained XGBoost classifier combined with 5 rule-based diagnostic algorithms.

---

## What This Project Does

NVMe drives don't fail instantly — they show measurable degradation patterns over time. This system:

1. **Detects signals** — Identifies abnormal telemetry (high temperature, rising error rates, excessive writes)
2. **Predicts before failure** — Uses ML to classify the likely failure mode before the drive actually dies
3. **Identifies patterns** — 5 independent algorithms score specific failure risks with explainable reasons

---

## Project Structure

```
lenovo2/
├── app.py                  # Flask web server (API + routes)
├── ml_pipeline.py          # ML training + rule-based algorithms
├── requirements.txt        # Python dependencies
├── pipeline_output.txt     # Training log output
│
├── data/
│   └── NVMe_Drive_Failure_Dataset.csv   # Your training dataset
│
├── models/
│   ├── xgb_model.pkl       # Trained XGBoost model
│   ├── scaler.pkl          # StandardScaler (feature normalization)
│   ├── label_encoder.pkl   # Maps encoded labels ↔ failure modes
│   └── metadata.json       # Model accuracy, feature importance
│
├── templates/
│   └── index.html          # Dashboard UI
│
└── static/
    ├── app.js              # Frontend logic (sliders, API calls, rendering)
    └── style.css           # Dark glassmorphism theme
```

---

## How It Works (Step by Step)

### Step 1: Training the Model (`ml_pipeline.py`)

```
python ml_pipeline.py
```

This reads `NVMe_Drive_Failure_Dataset.csv` and:

1. **Loads 6,000+ drive records** from the CSV
2. **Handles missing data** — fills empty cells with the column median (robust against outliers)
3. **Engineers 3 new features** from raw data:
   - `Error_Rate` = (Media_Errors + CRC_Errors) / Power_On_Hours
   - `Write_Intensity` = Total_TBW / Power_On_Hours
   - `Read_Intensity` = Total_TBR / Power_On_Hours
4. **Splits data** 80/20 into train/test sets (stratified to preserve class balance)
5. **Scales features** using StandardScaler (zero mean, unit variance)
6. **Trains XGBoost** with 400 estimators, max_depth=8, learning_rate=0.1
7. **Saves** the trained model, scaler, and label encoder to `models/`

The model classifies drives into **6 categories**:

| Code | Failure Mode |
|------|-------------|
| 0 | No Failure |
| 1 | Wear-Out Failure |
| 2 | Thermal Failure |
| 3 | Firmware Failure |
| 4 | Media Error Failure |
| 5 | Unsafe Shutdown Failure |

### Step 2: Rule-Based Algorithms (`ml_pipeline.py`)

Five independent diagnostic algorithms run alongside the ML model. Each scores a specific failure type from **0 to 100** using domain-expert thresholds:

| Algorithm | What It Checks | Key Thresholds |
|-----------|---------------|----------------|
| **Wear-Out Detection** | Life Used, TBW, Power-On Hours | Life ≥ 80% → high score |
| **Thermal Failure** | Temperature, workload under heat | Temp ≥ 75°C → critical |
| **Firmware Failure** | Read/Write Error Rates, CRC | Error rates ≥ 20 → flagged |
| **Media Error** | Media Errors, CRC, NAND wear | Errors ≥ 5 → significant |
| **Unsafe Shutdown** | Shutdown count, secondary damage | Shutdowns ≥ 10 → severe |

### How AI and Rule-Based Algorithms Differ

| | AI Prediction (XGBoost) | Rule-Based Algorithms (Strict Rules) |
|---|---|---|
| **How it works** | Learned patterns from the dataset during training. It "guesses" based on statistical correlations it discovered. | We manually coded exact thresholds (e.g., Temp ≥ 75°C = critical). These are **our rules**, not learned. |
| **Knows why?** | No — it sees a pattern and predicts, but can't explain the reasoning. | Yes — every score has a clear reason (e.g., "Life at 95% → NAND cells exhausted"). |
| **Trained on data?** | Yes — trained on 6,000+ rows from your CSV dataset. | No — thresholds are hard-coded by domain knowledge. |
| **Output** | Failure mode label + probability % | Score 0–100 + list of specific reasons |
| **Dashboard label** | "AI Failure Prediction" | "Independent Algorithms" |

**Why we use both together:**
- The AI gives a quick probabilistic guess (e.g., "75% Wear-Out Failure")
- The strict rules give **explainable scores** with exact reasons
- If the rules detect degradation but the AI says "No Failure", the system triggers an **Early Warning** — this is the key safety net that makes the system proactive

### Step 3: Web Dashboard (`app.py`)

```
python app.py
```

Opens at `http://localhost:5000`. The Flask server exposes:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Serves the dashboard HTML |
| `/api/predict` | POST | Takes 10 metric inputs → returns prediction + insights |
| `/api/metadata` | GET | Returns model accuracy & feature importance |

### Step 4: User Interaction (Frontend)

1. **Adjust sliders** — 10 telemetry inputs (Power-On Hours, Temperature, TBW, Errors, etc.)
2. **Click "Analyze & Predict"** — sends the values to `/api/predict`
3. **Backend processes**:
   - Builds a 13-feature vector (10 base + 3 engineered)
   - Runs through the trained XGBoost model
   - Runs all 5 rule-based algorithms
   - Generates dynamic Contributing Factors based on your exact input values
   - Generates specific Suggested Actions
4. **Dashboard displays**:
   - **Verdict banner** (Green/Amber/Red)
   - **ML prediction** with probability bars for each failure mode
   - **Risk Level** (Low/Medium/High)
   - **Contributing Factors** — what's actually wrong (e.g., "Thermal Spike (85°C)")
   - **Suggested Action** — what to do about it (e.g., "Check chassis airflow")
   - **5 Algorithm scores** — ranked by severity

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model (first time only)
```bash
python ml_pipeline.py
```
This creates the `models/` directory with the trained artifacts.

### 3. Start the dashboard
```bash
python app.py
```
Open `http://localhost:5000` in your browser.

---

## Input Features (What the Sliders Control)

| Feature | Unit | Range | What It Means |
|---------|------|-------|---------------|
| Power_On_Hours | hrs | 0–60,000 | How long the drive has been running |
| Total_TBW_TB | TB | 0–1,000 | Total data written to drive |
| Total_TBR_TB | TB | 0–1,000 | Total data read from drive |
| Temperature_C | °C | 20–100 | Current operating temperature |
| Percent_Life_Used | % | 0–100 | NAND wear level |
| Media_Errors | count | 0–20 | Unrecoverable read/write errors |
| Unsafe_Shutdowns | count | 0–20 | Unexpected power loss events |
| CRC_Errors | count | 0–20 | Data transfer checksum failures |
| Read_Error_Rate | rate | 0–50 | Read errors per unit time |
| Write_Error_Rate | rate | 0–50 | Write errors per unit time |

---

## Model Details
- **Accuracy**: ~98.7%
- **Algorithm**: XGBoost (Extreme Gradient Boosting)
- **Training Data**: `NVMe_Drive_Failure_Dataset.csv`
- **Cross-Validation**: 5-fold stratified

### Why XGBoost?
We specifically selected **XGBoost** for this NVMe failure prediction over other algorithms (like Random Forest, Neural Networks, or Logic Regression) for several critical reasons:

1. **Tabular Data Dominance:** XGBoost is widely considered the state-of-the-art algorithm for structured, tabular data (like our CSV dataset of telemetry metrics). Neural Networks typically underperform compared to tree-based models on tabular data.
2. **Handling Non-Linear Relationships:** SSD failure isn't linear. A drive sitting at 85°C isn't just "twice as likely" to fail as 42.5°C — the risk grows exponentially, often acting as a trigger event. XGBoost captures these complex non-linear inflection points easily.
3. **Imbalanced Classification:** In real life (and in our dataset), drives usually *don't* fail. "No Failure" makes up ~98% of the data. XGBoost supports assigning `sample_weight` metrics natively, allowing us to mathematically force the model to focus deeply on the rare 2% of failure cases so they don't get ignored.
4. **Built-in Handling of Missing Data:** If a sensor fails to log `Temperature_C`, XGBoost natively knows how to route the node path for missing values without breaking the pipeline.
5. **Feature Importance:** XGBoost provides excellent feature interpretability, allowing us to extract the exact weight/importance of each telemetry metric to understand what primarily drives wear-out or thermal failures.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | XGBoost, Scikit-learn |
| Backend | Flask (Python) |
| Frontend | HTML5, CSS3 (Glassmorphism), Vanilla JS |
| Data | Pandas, NumPy |
