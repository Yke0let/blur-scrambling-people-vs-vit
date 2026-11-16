# AGENTS.md — Project Context & Rules for Antigravity

## 1. Project Overview
**Title:** Blur and Scrambling: People vs. Vision Models
**Team:** Uriel-Darlin Azemfack-Himele, Fanès Houeha, Halik Mamam, Franck Tchangani
**Research Question:** Which kinds of image damage (Gaussian Blur vs. Offset Tile Shuffling) break vision models (CLIP ViT-L/14, DINOv2 ViT-B/14) and humans together, and which break only one of them?

## 2. Strict Scientific Invariants (DO NOT MODIFY)
- **10 Categories (10-AFC):** `["airplane", "bicycle", "car", "cat", "chair", "clock", "fish", "house", "tree", "umbrella"]`
- **Image Geometry:** `224x224` px grayscale line drawings from *Quick, Draw!* (CC BY 4.0), centered with `20px` white margin, `4px` stroke width.
- **ViT Patch Geometry:** Patch size = `14px` (`16x16` patches). Shuffle grid offset = `7px` (half a patch) so tile borders never align with ViT patch borders.
- **7 Experimental Conditions:**
  - `intact` (baseline)
  - `blur_low` ($\sigma=2$), `blur_medium` ($\sigma=5$), `blur_high` ($\sigma=10$)
  - `shuffle_low` ($2\times2 = 4$ tiles), `shuffle_medium` ($4\times4 = 16$ tiles), `shuffle_high` ($8\times8 = 64$ tiles)
- **Human Protocol Constraints:**
  - 12 participants (`P01` to `P12`), Latin-square rotation across 70 test images (`7` per category = `10` trials per condition per participant).
  - Fixed exposure time = **exact 2000 ms** (`performance.now()`), immediately followed by a **backward visual mask** (`400 ms`) before unlocking the 10-choice response grid.
  - Informed consent required, anonymous IDs only (`P01`–`P12`), right to quit anytime.

## 3. Data Schemas & File Contracts
- **`human_trials.csv` (Input to Web UI):**
  Columns: `participant,uid,condition,file,correct_label,trial`
  Images are located in `./stimuli_human/<file>`.
- **`responses_PXX.csv` (Output from each human participant):**
  Columns: `participant,trial,uid,condition,file,correct_label,response,correct,rt_ms`
  (`correct` is `1` or `0`).
- **`human_results.csv` (Aggregated human table consumed by the Notebook):**
  Columns: `condition,accuracy,ci_low,ci_high` (all values in percentage `0–100`, with 95% CI computed via 2000-resample bootstrap).
- **Reference Model Accuracies (%) from `model_results.csv` (N=1000 test images):**
  - `CLIP ViT-L/14`: intact=97.7, blur_low=91.9, blur_medium=84.5, blur_high=50.8, shuffle_low=63.5, shuffle_medium=23.8, shuffle_high=15.5
  - `DINOv2 ViT-B/14`: intact=97.5, blur_low=97.5, blur_medium=76.5, blur_high=58.2, shuffle_low=76.8, shuffle_medium=21.2, shuffle_high=10.4

## 4. Coding & Deployment Standards
- Keep the Web Experiment UI dependency-light (Vanilla HTML/JS + Tailwind CDN + Chart.js + PapaParse) so it works **both** on GitHub Pages / Vercel AND offline via `file://` or `python -m http.server` (Plan B).
- Always use relative paths (`./human_trials.csv` and `./stimuli_human/...`) so GitHub Pages subpath hosting (`https://<user>.github.io/<repo>/`) works out of the box.