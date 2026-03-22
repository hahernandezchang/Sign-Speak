# Sign-Speak

SignSpeak is an AI-powered translation tool that gives a spoken voice to sign languages. By instantly converting visual signs from video into natural spoken audio, SignSpeak bridges the communication gap between Deaf and hearing communities across multiple global sign languages.

## Table of Contents

- [Project Overview](#project-overview)
- [ASL End-to-End Flow Checklist](#asl-end-to-end-flow-checklist)
- [Quick Repeat Loop](#quick-repeat-loop)

## Project Overview

Use this repository to:

- Collect sign data for letters and words
- Split datasets into alphabet and word subsets
- Train separate alphabet and word models
- Run live runtime inference for letters, words, or hybrid behavior

## ASL End-to-End Flow Checklist

Use this checklist from the repository root folder:

`C:\Users\HaH4C\Code\Hack\Sign-Speak`

All Python commands below use this pattern:

`python .\<script>.py`

### 1) Collect More Letter Data

Run:

```powershell
python .\collector_v2.py --mode letters
```

What to do:

- Follow prompts for A to Z
- Collect both right and left hand (guided automatically)
- Press SPACE to start each recording
- Wait for 3-second countdown before each capture

Target:

- At least 100 to 200 samples per letter to start
- Collect extra for confusing letters (A, S, E, N)

### 2) Collect More Word Data

Run (non-interactive, recommended):

```powershell
python .\collector_v2.py --mode words --words "hello,thanks,help,project,time"
```

Run (interactive prompt mode):

```powershell
python .\collector_v2.py --mode words
```

What to do:

- If using `--words`, provide up to 10 comma-separated words
- If interactive mode works in your terminal, enter up to 10 words at prompts
- Record each word 5 times per session
- Repeat sessions if needed

Target:

- Minimum 30 to 50 samples per word
- Better: 100+ samples per word

### 3) Split Data Into Alphabet and Word Datasets

Run:

```powershell
python .\split_datasets.py --input-csv .\data\asl_data.csv --alphabet-csv .\data\asl_alphabet_data.csv --words-csv .\data\asl_words_data.csv
```

Expected result:

- Alphabet labels go to `.\data\asl_alphabet_data.csv`
- Word labels go to `.\data\asl_words_data.csv`

### 4) Train Alphabet Model

Run:

```powershell
python .\train.py --csv-path .\data\asl_alphabet_data.csv --model-out .\models\asl_alphabet_classifier.pt --label-map-out .\models\asl_alphabet_label_map.json --epochs 120 --batch-size 8 --learning-rate 0.0005 --dropout 0.4 --use-class-weights --augment
```

Expected artifacts:

- `.\models\asl_alphabet_classifier.pt`
- `.\models\asl_alphabet_label_map.json`

### 5) Train Word Model

Run:

```powershell
python .\train.py --csv-path .\data\asl_words_data.csv --model-out .\models\asl_words_classifier.pt --label-map-out .\models\asl_words_label_map.json --epochs 120 --batch-size 8 --learning-rate 0.0005 --dropout 0.4 --use-class-weights --augment
```

Note:

- You need at least 2 distinct word labels in `.\data\asl_words_data.csv`

Expected artifacts:

- `.\models\asl_words_classifier.pt`
- `.\models\asl_words_label_map.json`

### 6) Validate During Training

Validation is automatic inside `train.py`.

Watch these lines while training:

- train loss, train acc
- val loss, val acc

What good looks like:

- val acc trends upward over epochs
- val loss generally decreases

If validation is weak:

- Collect more balanced data
- Add more samples for confused classes
- Increase epochs (for example 150)

### 7) Runtime Testing

Letters mode:

```powershell
python .\dual_runtime.py --voice --mode letters
```

Words mode:

```powershell
python .\dual_runtime.py --voice --mode words
```

Hybrid mode:

```powershell
python .\dual_runtime.py --voice --mode hybrid
```

## Quick Repeat Loop

1. Collect letters
2. Collect words
3. Split data
4. Train alphabet
5. Train words
6. Test letters/words/hybrid
7. Repeat with more data where errors appear

## Web Runtime (React + Vite)

This branch now includes a web stack:

- Python API bridge: `runtime_api.py`
- React client: `client/`

### 1) Start the API

Install dependencies (once):

```powershell
python -m pip install -r .\requirements.txt
```

Run server:

```powershell
python -m uvicorn runtime_api:app --reload --host 0.0.0.0 --port 8000
```

### 2) Start the React UI

In a second terminal:

```powershell
cd .\client
npm install
npm run dev
```

Open:

`http://localhost:5173`

### 3) Use the UI

- Choose `letters`, `words`, or `hybrid`
- Keep source `0` for default webcam
- Click Start to begin inference
- Click Stop to release camera
- Use `Clear Text`, `Clear Phrase`, and `Speak` for runtime commands

The client streams runtime state from `ws://localhost:8000/stream`.

Additional API endpoints now available:

- `POST /command` with `{"action":"clear_text"|"clear_phrase"|"speak"}`
- `GET /frame` for latest webcam frame (`image/jpeg`)
