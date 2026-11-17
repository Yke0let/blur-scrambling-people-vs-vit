# AGENTS.md — Project Context & Rules for Antigravity

## 1. Project Overview
**Title:** Blur and Scrambling: People vs. Vision Models
**Team:** Uriel-Darlin Azemfack-Himele, Fanès Houeha, Halik Mamam, Franck Tchangani
**Research Question:** Which kinds of image damage (Gaussian Blur vs. Offset Tile Shuffling) break vision models (CLIP ViT-L/14, DINOv2 ViT-B/14) and humans together, and which break only one of them?

## 2. Strict Scientific Invariants (DO NOT MODIFY)
- **10 Categories (10-AFC):** `["airplane", "bicycle", "car", "cat", "chair", "clock", "fish", "house", "tree", "umbrella"]`
- **Image Geometry:** `224x224` px grayscale line drawings from *Quick, Draw!* (CC BY 4.0), centered with `20px` white margin, `4px` stroke width (rendered at `384x384` px in the web UI for comfortable viewing).
- **ViT Patch Geometry:** Patch size = `14px` (`16x16` patches). Shuffle grid offset = `7px` (half a patch) so tile borders never align with ViT patch borders.
- **7 Experimental Conditions:**
  - `intact` (baseline)
  - `blur_low` ($\sigma=2$), `blur_medium` ($\sigma=5$), `blur_high` ($\sigma=10$)
  - `shuffle_low` ($2\times2 = 4$ tiles), `shuffle_medium` ($4\times4 = 16$ tiles), `shuffle_high` ($8\times8 = 64$ tiles)
- **Human Protocol Constraints (21 Participants = 3 Groups × 7 Slices):**
  - **21 participants (`P01` to `P21`)** organized into **3 Groups of 7 (`G1`: `P01`–`P07`, `G2`: `P08`–`P14`, `G3`: `P15`–`P21`)**.
  - Within each group of 7, the 7 participants partition all 490 stimuli (`70` unique test `uid`s × `7` conditions) into 7 disjoint Latin-square slices (`S1`–`S7`, 70 trials per participant = 10 per condition, 7 per category). Person $k$ ($k \in \{1..7\}$) of Group 1, Group 2, and Group 3 all evaluate the exact same 70 stimulus images (Slice $k$).
  - **Trial Timing:** `600 ms` fixation cross $\rightarrow$ **exact `1500 ms` (`1.5s`)** stimulus exposure (`performance.now()`) $\rightarrow$ **`400 ms`** procedural backward visual mask $\rightarrow$ 10-AFC response grid with a **hidden `15000 ms` (`15s`) answer timer** (if `15s` elapses without response, the trial is re-queued to the end of the participant's trial queue).
  - **Strict Anonymity & Blind Completion:** No participant name is collected; participants are randomly assigned an available ID (`P01`–`P21`), practice uses 2 intact drawings only (no repeat), and upon finishing 70 trials, responses are automatically transmitted to the Admin cloud store without showing scores or charts to the participant.
  - **Admin / Manager Portal (`admin2026_Aims`):** Password-protected dashboard providing Candidate-level (`P01`–`P21`), Group-level (`G1`–`G3`), and Global results with 1-click CSV exports.

## 3. Data Schemas & File Contracts
- **`human_trials.csv` (Input to Web UI):**
  Columns: `participant,group,slot_in_group,uid,condition,file,correct_label,trial`
  Images are located in `./stimuli_human/<file>`.
- **`responses_PXX.csv` (Output from each human participant):**
  Columns: `participant,group,slot_in_group,trial,uid,condition,file,correct_label,response,correct,rt_ms`
  (`correct` is `1` or `0`).
- **`human_results.csv` (Aggregated human table consumed by the Notebook):**
  Columns: `condition,accuracy,ci_low,ci_high` (all values in percentage `0–100`, with 95% CI computed via 2000-resample bootstrap).
- **Reference Model Accuracies (%) on the 70 `human_uids` (`model_results_human_subset.csv`) & Full `N=1000` (`model_results.csv`):**
  - `CLIP ViT-L/14 (N=70 human_uids)`: intact=98.6, blur_low=92.9, blur_medium=88.6, blur_high=52.9, shuffle_low=54.3, shuffle_medium=21.4, shuffle_high=17.1
  - `DINOv2 ViT-B/14 (N=70 human_uids)`: intact=95.7, blur_low=97.1, blur_medium=85.7, blur_high=54.3, shuffle_low=67.1, shuffle_medium=15.7, shuffle_high=10.0
  - `CLIP ViT-L/14 (N=1000)`: intact=97.7, blur_low=91.9, blur_medium=84.5, blur_high=50.8, shuffle_low=63.5, shuffle_medium=23.8, shuffle_high=15.5
  - `DINOv2 ViT-B/14 (N=1000)`: intact=97.5, blur_low=97.5, blur_medium=76.5, blur_high=58.2, shuffle_low=76.8, shuffle_medium=21.2, shuffle_high=10.4

## 4. Coding & Deployment Standards
- Keep the Web Experiment UI dependency-light (Vanilla HTML/JS + Tailwind CDN + Chart.js + PapaParse) so it works **both** on GitHub Pages / Vercel AND offline via `file://` or `python -m http.server` (Plan B).
- Always use relative paths (`./human_trials.csv` and `./stimuli_human/...`) so GitHub Pages subpath hosting (`https://<user>.github.io/<repo>/`) works out of the box.
