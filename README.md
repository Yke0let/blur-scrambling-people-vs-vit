# Blur and Scrambling: People vs. Vision Models

**Authors:** Uriel-Darlin Azemfack-Himele, Fanès Houeha, Halik Mamam, Franck Tchangani  
**Live Psychophysics Platform:** [https://yke0let.github.io/blur-scrambling-people-vs-vit/](https://yke0let.github.io/blur-scrambling-people-vs-vit/)

---

## 1. Research Question

> *Which kinds of image damage—low-pass **Gaussian Blur** vs. **7px-Offset Tile Shuffling**—break Vision Transformers (`CLIP ViT-L/14`, `DINOv2 ViT-B/14`) and human observers together, and which break only one of them?*

![Model Accuracy under Blur and Tile Scrambling](./comparison_figure.png)

---

## 2. Experimental Design & Scientific Invariants

- **Dataset & Stimulus Geometry:**
  - `224 × 224` px grayscale line drawings rendered from *Quick, Draw!* (CC BY 4.0), centered with a `20 px` white margin and `4 px` stroke width.
  - **10 Forced-Choice Categories (10-AFC, chance = 10%):** `airplane`, `bicycle`, `car`, `cat`, `chair`, `clock`, `fish`, `house`, `tree`, `umbrella`.
- **ViT Patch Geometry & Offset Scrambling:**
  - Both vision backbones use a `14 × 14` px patch grid (`16 × 16` patches).
  - Tile shuffling shifts the cutting grid by **`+7 px`** (half a patch) and applies a random **derangement** (no tile remains in its original position) so tile boundaries never align with ViT patch boundaries.
- **7 Experimental Conditions:**
  1. `intact` — Uncorrupted baseline drawing
  2. `blur_low` ($\sigma = 2\text{ px}$), `blur_medium` ($\sigma = 5\text{ px}$), `blur_high` ($\sigma = 10\text{ px}$) — Gaussian blur
  3. `shuffle_low` ($2\times 2 = 4$ tiles), `shuffle_medium` ($4\times 4 = 16$ tiles), `shuffle_high` ($8\times 8 = 64$ tiles) — Offset tile scrambling
- **Human Psychophysics Protocol:**
  - **12 participants (`P01`–`P12`)**, Latin-square rotation across `70` test drawings (`7` per category = `10` trials per condition per participant, `840` total human trials).
  - **Trial Timing:** `600 ms` central fixation cross $\rightarrow$ **exact `2000 ms`** (`performance.now()`) stimulus exposure $\rightarrow$ **`400 ms`** procedural backward visual mask (`60 fps` checker noise + random `4 px` strokes) $\rightarrow$ untimed 10-AFC category selection.

---

## 3. Vision Model Benchmark Results

Linear probes were trained strictly on **intact** training images (`N = 5000`, validated on `N = 1000`) on top of frozen **CLIP ViT-L/14** (`openai/clip-vit-large-patch14`) and **DINOv2 ViT-B/14** (`facebook/dinov2-base`) representations, then evaluated zero-shot across all 7 conditions on both the full test split (`N = 1000` per condition) and the exact `70` unique `human_uids` subset (`N = 70` per condition):

| Condition | Parameter | CLIP ViT-L/14 (`N=1000`) | CLIP ViT-L/14 (`N=70` UIDs) | DINOv2 ViT-B/14 (`N=1000`) | DINOv2 ViT-B/14 (`N=70` UIDs) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `intact` | Baseline | 97.7% `[96.7, 98.6]` | 98.6% `[95.7, 100.0]` | 97.5% `[96.5, 98.4]` | 95.7% `[90.0, 100.0]` |
| `blur_low` | $\sigma = 2\text{ px}$ | 91.9% `[90.1, 93.6]` | 92.9% `[85.7, 98.6]` | 97.5% `[96.5, 98.4]` | 97.1% `[92.9, 100.0]` |
| `blur_medium` | $\sigma = 5\text{ px}$ | 84.5% `[82.3, 86.8]` | 88.6% `[80.0, 95.7]` | 76.5% `[73.9, 79.1]` | 85.7% `[77.1, 92.9]` |
| `blur_high` | $\sigma = 10\text{ px}$ | 50.8% `[47.6, 53.8]` | 52.9% `[41.4, 64.3]` | 58.2% `[55.1, 61.1]` | 54.3% `[42.9, 65.7]` |
| `shuffle_low` | $2\times 2$ (`4` tiles) | 63.5% `[60.7, 66.5]` | 54.3% `[42.9, 65.7]` | 76.8% `[74.2, 79.5]` | 67.1% `[55.7, 78.6]` |
| `shuffle_medium` | $4\times 4$ (`16` tiles) | 23.8% `[21.2, 26.4]` | 21.4% `[12.9, 31.4]` | 21.2% `[18.7, 23.8]` | 15.7% `[7.1, 24.3]` |
| `shuffle_high` | $8\times 8$ (`64` tiles) | 15.5% `[13.2, 17.7]` | 17.1% `[8.6, 25.7]` | 10.4% `[8.6, 12.3]` | 10.0% `[4.3, 17.1]` |

*(Brackets indicate 95% confidence intervals computed via 2,000-resample percentile bootstrap.)*

---

## 4. Repository Structure

```text
├── index.html                        # 5-step web psychophysics platform (hosted on GitHub Pages & offline-ready)
├── human_trials.csv                  # Latin-square schedule for P01–P12 (840 rows = 12 participants × 70 trials)
├── stimuli_human/                    # 490 PNG stimuli (70 test UIDs × 7 conditions, 224×224 grayscale)
├── analyze_results.py                # Post-collection aggregation & model vs. human comparison script
├── blur_scrambling_people_vs_vit.ipynb # End-to-end Colab notebook (stimulus generation, ViT probing, figures)
├── model_results.csv                 # Model accuracies & 95% bootstrap CIs on the full N=1000 test set
├── model_results_human_subset.csv    # Model accuracies & 95% bootstrap CIs on the 70 human_uids subset
├── comparison_results.csv            # Side-by-side table comparing N=1000 vs. N=70 human_uids (& humans)
├── model_predictions.csv             # Per-image predictions (2 models × 7 conditions × 1000 UIDs)
├── stimuli_manifest.csv              # Seeds, tile permutations, and ink-tile fractions for all 7000 stimuli
├── main_figure.png                   # Baseline two-panel accuracy plot (Blur | Tile Shuffle)
├── comparison_figure.png             # Two-panel plot overlaying N=1000 and N=70 human_uids curves
├── human_logs/                       # Directory for collected participant files (responses_P01.csv ... P12.csv)
└── project_outputs/                  # Raw Colab archive & trained linear probe weights (probe_*.pt)
```

---

## 5. Running the Experiment & Analyzing Human Data

### A. Collecting Human Responses (Online or Local)
1. **Online (GitHub Pages):** Send participants the study URL:
   - General link: `https://yke0let.github.io/blur-scrambling-people-vs-vit/`
   - Direct participant link (pre-selects slot `P01`–`P12`): e.g. `https://yke0let.github.io/blur-scrambling-people-vs-vit/?p=P04`
2. **Local Server (Plan B):**
   ```bash
   python3 -m http.server 8000
   # Open http://localhost:8000 in any browser
   ```
3. At the end of the 70 trials, each participant's browser automatically downloads `responses_<PXX>_<Name>.csv` and provides a 1-click **Result Code (`VL1:...`)** that can be sent via WhatsApp or email.

### B. Aggregating Human Logs & Updating Figures
Place collected `responses_*.csv` files (or `.txt` files containing `VL1:...` Result Codes) into `human_logs/` and run:

```bash
python3 analyze_results.py
```

This script:
- Aggregates all participant trials in `human_logs/` into `human_results.csv` (`condition,accuracy,ci_low,ci_high`) with 2,000-resample bootstrap 95% confidence intervals.
- Evaluates `CLIP ViT-L/14` and `DINOv2 ViT-B/14` on the 70 `human_uids` alongside the full `N = 1000` test set (`comparison_results.csv`).
- Updates `main_figure.png` and `comparison_figure.png` with the human psychometric curve overlaid.

*(Alternatively, click **`Chercheur / Admin`** in the footer of `index.html` to merge CSVs or `VL1:...` codes and export `human_results.csv` directly in the browser.)*

---

## 6. License & Attribution

- Drawing data sourced from the [Google *Quick, Draw!* Dataset](https://github.com/googlecreativelab/quickdraw-dataset) under **CC BY 4.0**.
- Pretrained encoders from OpenAI (`openai/clip-vit-large-patch14`) and Meta AI (`facebook/dinov2-base`) via Hugging Face `transformers`.
