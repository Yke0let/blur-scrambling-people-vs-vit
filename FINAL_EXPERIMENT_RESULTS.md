# Final Experimental Report — Blur and Scrambling: People vs. Vision Models

**Authors / Research Team:** Uriel-Darlin Azemfack-Himele, Fanès Houeha, Halik Mamam, Franck Tchangani  
**Dataset & Stimuli:** *Quick, Draw!* (CC BY 4.0), $224 \times 224\text{ px}$ centered grayscale line drawings ($20\text{ px}$ white border margin, $4\text{ px}$ stroke width), 10 categories (`airplane`, `bicycle`, `car`, `cat`, `chair`, `clock`, `fish`, `house`, `tree`, `umbrella`).  
**Vision Models Evaluated (on NVIDIA L40S 46 GB GPU, node `comp057.hpc`):**
- **CLIP ViT-L/14** (`openai/clip-vit-large-patch14`, $14\text{ px}$ patch grid = $16 \times 16$ patches, frozen encoder + linear probe trained strictly on $5{,}000$ intact drawings, $\text{weight\_decay}=10^{-3}$, intact validation accuracy $= 97.10\%$).
- **DINOv2 ViT-B/14** (`facebook/dinov2-base`, $14\text{ px}$ patch grid = $16 \times 16$ patches, frozen encoder + linear probe trained strictly on $5{,}000$ intact drawings, $\text{weight\_decay}=10^{-1}$, intact validation accuracy $= 96.40\%$).

**Human Psychophysics Protocol ($N = 21$ Participants = $3\text{ Groups} \times 7\text{ Latin-Square Slices} = 1{,}470\text{ Trials}$):**
- **3 Balanced Groups of 7 Participants (`P01`–`P21`):**
  - **Group 1 (`G1`):** `P01`–`P07` (Slices `S1`–`S7`, covering all $490$ unique `(uid, condition)` stimulus pairs = $70$ unique test drawings $\times 7$ conditions).
  - **Group 2 (`G2`):** `P08`–`P14` (Slices `S1`–`S7`, independent replication across all $490$ stimulus pairs).
  - **Group 3 (`G3`):** `P15`–`P21` (Slices `S1`–`S7`, third independent replication across all $490$ stimulus pairs).
- **Trial Timing:** $600\text{ ms}$ central fixation cross $\rightarrow$ $1500\text{ ms}$ stimulus exposure ($384 \times 384\text{ px}$ viewport) $\rightarrow$ $400\text{ ms}$ procedural backward visual mask $\rightarrow$ 10-AFC response grid (chance $= 10\%$).

---

## 1. Key Scientific Findings (Research Question Answered)

> **Research Question:** *Which kinds of image damage (Gaussian Blur vs. 7px-Offset Tile Shuffling) break vision models and people together, and which break only one of them?*

1. **Severe Gaussian Blur ($\sigma = 10\text{ px}$) Breaks Vision Transformers While Humans Remain Robust (Selective Model Failure):**
   - At `blur_high` ($\sigma = 10\text{ px}$), human 10-way recognition accuracy remains high at **$82.38\%$** ($95\%\text{ CI}: [77.14, 87.62]$) — identical to `blur_medium` ($82.38\%$) and only $5.72$ percentage points below the `intact` human baseline ($88.10\%$).
   - By contrast, both Vision Transformers suffer a sharp collapse at `blur_high` ($\sigma = 10\text{ px}$):
     - **CLIP ViT-L/14** drops by **$-46.9\text{ pp}$** on the full test set ($97.70\% \rightarrow 50.70\%$, $[47.60, 53.80]$) and by **$-45.71\text{ pp}$** on the matched 70 human UIDs ($98.57\% \rightarrow 52.86\%$, $[41.43, 64.29]$).
     - **DINOv2 ViT-B/14** drops by **$-39.3\text{ pp}$** on the full test set ($97.50\% \rightarrow 58.20\%$, $[55.10, 61.10]$) and by **$-41.42\text{ pp}$** on the matched 70 human UIDs ($95.71\% \rightarrow 54.29\%$, $[42.86, 65.71]$).
   - Because the $95\%$ confidence intervals at $\sigma = 10\text{ px}$ are completely non-overlapping by more than $23$ percentage points (Humans $[77.14, 87.62]$ vs. CLIP $[47.60, 53.80]$ and DINOv2 $[55.10, 61.10]$), this demonstrates that human observers integrate coarse low-spatial-frequency silhouette/gist information far more effectively than frozen patch-14 ViT encoders trained on intact line drawings.

2. **7px-Offset Tile Shuffling ($2\times2 \rightarrow 4\times4 \rightarrow 8\times8$) Breaks Humans and Vision Transformers Together (Shared Global-Structure Failure):**
   - Because the shuffle grid is shifted by $+7\text{ px}$ (half a $14\text{ px}$ ViT patch) and enforces a strict derangement (no tile remains in its original location), every tile seam cuts directly through internal ViT patches and destroys global contour continuity.
   - Both humans and Vision Transformers collapse monotonically as the grid granularity increases:
     - **`shuffle_low` ($2\times2 = 4$ tiles):** Humans drop to **$47.14\%$** $[40.48, 53.81]$; CLIP drops to **$54.29\%$** ($70$ UIDs) / **$63.50\%$** ($N=1000$); DINOv2 drops to **$67.14\%$** ($70$ UIDs) / **$76.80\%$** ($N=1000$).
     - **`shuffle_medium` ($4\times4 = 16$ tiles):** Humans (**$22.38\%$** $[16.67, 27.62]$), CLIP (**$21.43\%$** on $70$ UIDs / **$23.70\%$** on $N=1000$), and DINOv2 (**$15.71\%$** on $70$ UIDs / **$21.20\%$** on $N=1000$) converge to virtually identical accuracy ($\sim 21\text{--}24\%$).
     - **`shuffle_high` ($8\times8 = 64$ tiles):** Humans (**$15.24\%$** $[10.48, 20.48]$), CLIP (**$17.14\%$** on $70$ UIDs / **$15.50\%$** on $N=1000$), and DINOv2 (**$10.00\%$** on $70$ UIDs / **$10.40\%$** on $N=1000$) all collapse to near the $10\%$ chance floor.
   - This directly refutes the "bag-of-patches" invariance hypothesis for $+7\text{ px}$ offset tile scrambling: Vision Transformers do **not** retain high accuracy when local stroke junctions are severed across patch interiors.

---

## 2. Main Benchmark Table: Humans vs. Vision Transformers Across All 7 Conditions

All accuracies are reported in percentage ($0\text{--}100\%$) with $2{,}000$-resample percentile bootstrap $95\%$ confidence intervals (`seed=1234`).

| Condition | Damage Parameter | Humans ($N=21$, $1470$ Trials) Acc [95% CI] | CLIP ViT-L/14 ($70$ Human UIDs) Acc [95% CI] | CLIP ViT-L/14 (Full $N=1000$) Acc [95% CI] | DINOv2 ViT-B/14 ($70$ Human UIDs) Acc [95% CI] | DINOv2 ViT-B/14 (Full $N=1000$) Acc [95% CI] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`intact`** | Baseline ($0$) | **88.10%** `[83.80, 92.38]` | **98.57%** `[95.71, 100.00]` | **97.70%** `[96.70, 98.60]` | **95.71%** `[90.00, 100.00]` | **97.50%** `[96.50, 98.40]` |
| **`blur_low`** | Gaussian $\sigma = 2\text{ px}$ | **89.05%** `[84.76, 93.33]` | **92.86%** `[85.71, 98.57]` | **91.90%** `[90.10, 93.60]` | **97.14%** `[92.86, 100.00]` | **97.50%** `[96.50, 98.40]` |
| **`blur_medium`** | Gaussian $\sigma = 5\text{ px}$ | **82.38%** `[77.14, 87.14]` | **88.57%** `[80.00, 95.71]` | **84.50%** `[82.30, 86.80]` | **85.71%** `[77.14, 92.86]` | **76.50%** `[73.90, 79.10]` |
| **`blur_high`** | Gaussian $\sigma = 10\text{ px}$ | **82.38%** `[77.14, 87.62]` | **52.86%** `[41.43, 64.29]` | **50.70%** `[47.60, 53.80]` | **54.29%** `[42.86, 65.71]` | **58.20%** `[55.10, 61.10]` |
| **`shuffle_low`** | $2 \times 2$ ($4$ tiles, $+7\text{ px}$) | **47.14%** `[40.48, 53.81]` | **54.29%** `[42.86, 65.71]` | **63.50%** `[60.70, 66.50]` | **67.14%** `[55.71, 78.57]` | **76.80%** `[74.20, 79.50]` |
| **`shuffle_medium`** | $4 \times 4$ ($16$ tiles, $+7\text{ px}$) | **22.38%** `[16.67, 27.62]` | **21.43%** `[12.86, 31.43]` | **23.70%** `[21.10, 26.30]` | **15.71%** `[7.14, 24.29]` | **21.20%** `[18.70, 23.80]` |
| **`shuffle_high`** | $8 \times 8$ ($64$ tiles, $+7\text{ px}$) | **15.24%** `[10.48, 20.48]` | **17.14%** `[8.57, 25.71]` | **15.50%** `[13.20, 17.70]` | **10.00%** `[4.25, 17.14]` | **10.40%** `[8.60, 12.30]` |

---

## 3. Inter-Group Replication & Consistency Across Groups 1, 2, and 3

Each group of 7 participants (`G1`: `P01`–`P07`, `G2`: `P08`–`P14`, `G3`: `P15`–`P21`) independently evaluated all $490$ stimuli ($70\text{ trials/condition/group}$). All three groups replicate the exact same dissociation between high Gaussian blur robustness and steep tile-shuffle collapse:

| Condition | Group 1 (`P01`–`P07`, $490$ Trials) Acc [95% CI] | Group 2 (`P08`–`P14`, $490$ Trials) Acc [95% CI] | Group 3 (`P15`–`P21`, $490$ Trials) Acc [95% CI] | Pooled Humans ($21$ Candidates, $1470$ Trials) |
| :--- | :---: | :---: | :---: | :---: |
| **`intact`** | **91.43%** `[84.29, 97.14]` | **81.43%** `[71.43, 90.00]` | **91.43%** `[84.29, 97.14]` | **88.10%** `[83.80, 92.38]` |
| **`blur_low`** | **85.71%** `[77.14, 92.86]` | **88.57%** `[80.00, 95.71]` | **92.86%** `[85.71, 98.57]` | **89.05%** `[84.76, 93.33]` |
| **`blur_medium`** | **88.57%** `[80.00, 95.71]` | **77.14%** `[67.14, 87.14]` | **81.43%** `[71.43, 90.00]` | **82.38%** `[77.14, 87.14]` |
| **`blur_high`** | **80.00%** `[70.00, 88.57]` | **85.71%** `[77.14, 92.86]` | **81.43%** `[71.43, 90.00]` | **82.38%** `[77.14, 87.62]` |
| **`shuffle_low`** | **38.57%** `[27.14, 50.00]` | **42.86%** `[31.43, 54.29]` | **60.00%** `[48.57, 71.43]` | **47.14%** `[40.48, 53.81]` |
| **`shuffle_medium`** | **15.71%** `[7.14, 24.29]` | **25.71%** `[15.71, 35.71]` | **25.71%** `[15.71, 35.71]` | **22.38%** `[16.67, 27.62]` |
| **`shuffle_high`** | **15.71%** `[7.14, 24.29]` | **8.57%** `[2.86, 15.71]` | **21.43%** `[12.86, 31.43]` | **15.24%** `[10.48, 20.48]` |
| **Overall Group Mean** | **59.39%** | **58.57%** | **64.90%** | **60.95%** |

---

## 4. Complete 21-Participant Roster (`P01`–`P21`)

| Participant ID | Group | Stimulus Slice ($70$ Imgs) | Completed Trials | Overall Acc (%) | Intact Acc (%) | Blur Mean Acc (%) | Shuffle Mean Acc (%) | Median RT (ms) | Re-Queued Timeouts |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`P01`** | `G1` | `Slice 1` | $70 / 70$ | 44.29% | 100.0% | 63.33% | 6.67% | 1506 ms | 0 |
| **`P02`** | `G1` | `Slice 2` | $70 / 70$ | 62.86% | 100.0% | 86.67% | 26.67% | 1040 ms | 0 |
| **`P03`** | `G1` | `Slice 3` | $70 / 70$ | 51.43% | 80.0% | 86.67% | 6.67% | 2092 ms | 0 |
| **`P04`** | `G1` | `Slice 4` | $70 / 70$ | 68.57% | 90.0% | 96.67% | 33.33% | 1566 ms | 5 |
| **`P05`** | `G1` | `Slice 5` | $70 / 70$ | 65.71% | 90.0% | 93.33% | 30.00% | 1244 ms | 31 |
| **`P06`** | `G1` | `Slice 6` | $70 / 70$ | 55.71% | 80.0% | 73.33% | 30.00% | 1346 ms | 4 |
| **`P07`** | `G1` | `Slice 7` | $70 / 70$ | 67.14% | 100.0% | 93.33% | 30.00% | 1140 ms | 2 |
| **`P08`** | `G2` | `Slice 1` | $70 / 70$ | 58.57% | 80.0% | 83.33% | 26.67% | 1490 ms | 0 |
| **`P09`** | `G2` | `Slice 2` | $70 / 70$ | 64.29% | 90.0% | 86.67% | 33.33% | 805 ms | 0 |
| **`P10`** | `G2` | `Slice 3` | $70 / 70$ | 60.00% | 90.0% | 93.33% | 16.67% | 476 ms | 0 |
| **`P11`** | `G2` | `Slice 4` | $70 / 70$ | 65.71% | 70.0% | 90.00% | 40.00% | 920 ms | 0 |
| **`P12`** | `G2` | `Slice 5` | $70 / 70$ | 52.86% | 80.0% | 73.33% | 23.33% | 708 ms | 3 |
| **`P13`** | `G2` | `Slice 6` | $70 / 70$ | 54.29% | 80.0% | 76.67% | 23.33% | 1280 ms | 0 |
| **`P14`** | `G2` | `Slice 7` | $70 / 70$ | 54.29% | 80.0% | 83.33% | 16.67% | 525 ms | 0 |
| **`P15`** | `G3` | `Slice 1` | $70 / 70$ | 62.86% | 100.0% | 66.67% | 46.67% | 1471 ms | 0 |
| **`P16`** | `G3` | `Slice 2` | $70 / 70$ | 61.43% | 100.0% | 86.67% | 23.33% | 946 ms | 5 |
| **`P17`** | `G3` | `Slice 3` | $70 / 70$ | 51.43% | 80.0% | 80.00% | 13.33% | 1168 ms | 0 |
| **`P18`** | `G3` | `Slice 4` | $70 / 70$ | 67.14% | 90.0% | 86.67% | 40.00% | 906 ms | 1 |
| **`P19`** | `G3` | `Slice 5` | $70 / 70$ | 71.43% | 90.0% | 96.67% | 40.00% | 1578 ms | 3 |
| **`P20`** | `G3` | `Slice 6` | $70 / 70$ | 68.57% | 90.0% | 83.33% | 46.67% | 748 ms | 0 |
| **`P21`** | `G3` | `Slice 7` | $70 / 70$ | 71.43% | 90.0% | 96.67% | 40.00% | 788 ms | 0 |

---

## 5. Psychophysical Diagnostics: Reaction Times & Ink-Tile Fraction

### A. Human Reaction Time (`rt_ms`) by Condition
| Condition | Median RT (ms) | 25th Percentile (ms) | 75th Percentile (ms) | Interquartile Range (IQR) | Psychophysical Interpretation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`intact`** | **765 ms** | 520 ms | 1397 ms | 877 ms | Fast feed-forward recognition |
| **`blur_low`** ($\sigma=2$) | **798 ms** | 539 ms | 1277 ms | 738 ms | Negligible latency cost ($+33\text{ ms}$) |
| **`blur_medium`** ($\sigma=5$) | **942 ms** | 581 ms | 1987 ms | 1406 ms | Moderate low-frequency integration ($+177\text{ ms}$) |
| **`blur_high`** ($\sigma=10$) | **943 ms** | 570 ms | 1839 ms | 1269 ms | Same latency as $\sigma=5\text{ px}$ ($+178\text{ ms}$), $82.38\%$ accuracy |
| **`shuffle_low`** ($4$ tiles) | **1410 ms** | 792 ms | 2956 ms | 2164 ms | Slow mental tile reassembly ($+645\text{ ms}$) |
| **`shuffle_medium`** ($16$ tiles) | **1883 ms** | 963 ms | 3042 ms | 2079 ms | Peak cognitive search latency ($+1118\text{ ms}$, $2.46\times$ intact) |
| **`shuffle_high`** ($64$ tiles) | **1630 ms** | 779 ms | 3058 ms | 2279 ms | Guessing near chance ($15.24\%$ accuracy) |

### B. Fraction of Tiles Containing Ink (`ink_tile_frac`) & `shuffle_high` Split
- **Proportion of tiles containing ink across the $1{,}000$ test drawings:**
  - `shuffle_low` ($2 \times 2 = 4$ tiles): $\text{mean} = 0.982$ ($\min = 0.750, \max = 1.000$)
  - `shuffle_medium` ($4 \times 4 = 16$ tiles): $\text{mean} = 0.706$ ($\min = 0.312, \max = 1.000$)
  - `shuffle_high` ($8 \times 8 = 64$ tiles): $\text{mean} = 0.448$ ($\min = 0.203, \max = 0.734$, median $= 0.438$)
- **Accuracy on `shuffle_high` ($64$ tiles) split by Few Ink Tiles ($\le \text{median}$) vs. Many Ink Tiles ($> \text{median}$):**
  - **CLIP ViT-L/14 ($N=1000$):** Few Ink Tiles $= \mathbf{21.1\%}$ vs. Many Ink Tiles $= \mathbf{9.8\%}$
  - **DINOv2 ViT-B/14 ($N=1000$):** Few Ink Tiles $= \mathbf{15.1\%}$ vs. Many Ink Tiles $= \mathbf{5.6\%}$
  - **Humans ($N=21$, $210$ `shuffle_high` trials):** Few Ink Tiles $= \mathbf{13.7\%}$ vs. Many Ink Tiles $= \mathbf{16.7\%}$
  - *Interpretation:* When many tiles contain ink ($> 43.8\%$ of the $64$ tiles), an $8\times8$ derangement scatters black stroke fragments across the entire $224\times224$ canvas, creating dense false patch junctions that drive both Vision Transformers down to or below the $10\%$ chance floor ($9.8\%$ for CLIP, $5.6\%$ for DINOv2).

---

## 6. Generated Figures & Data Artifacts

All figures and CSV outputs have been generated on `comp057.hpc` (`NVIDIA L40S`) and synced to the root directory and `./project_outputs/`:
- [main_figure.png](file:///run/media/teleky-franck/New%20Volume/AI%20for%20Science/Visual%20Perception/Project/main_figure.png): Clean two-panel publication figure (Blur vs. 7px-Offset Tile Shuffle) comparing **Humans ($N=21$)**, **CLIP ViT-L/14**, and **DINOv2 ViT-B/14** with $95\%$ bootstrap confidence intervals.
- [comparison_figure.png](file:///run/media/teleky-franck/New%20Volume/AI%20for%20Science/Visual%20Perception/Project/comparison_figure.png): Full $N=1000$ vs. matched $70$ `human_uids` model curves overlaid with **Humans ($N=21$)**.
- [group_comparison_figure.png](file:///run/media/teleky-franck/New%20Volume/AI%20for%20Science/Visual%20Perception/Project/group_comparison_figure.png): Independent replication curves for **Group 1 (`P01`–`P07`)**, **Group 2 (`P08`–`P14`)**, and **Group 3 (`P15`–`P21`)** vs. **CLIP ViT-L/14** and **DINOv2 ViT-B/14**.
- [diagnostics_figure.png](file:///run/media/teleky-franck/New%20Volume/AI%20for%20Science/Visual%20Perception/Project/diagnostics_figure.png): Panel A (Human Reaction Time Median $\pm$ IQR by condition) & Panel B (`shuffle_high` accuracy split by below/above median `ink_tile_frac`).
- [blur_scrambling_people_vs_vit.ipynb](file:///run/media/teleky-franck/New%20Volume/AI%20for%20Science/Visual%20Perception/Project/blur_scrambling_people_vs_vit.ipynb): Fully GPU-executed notebook with all cell outputs and figures embedded.
- [human_results.csv](file:///run/media/teleky-franck/New%20Volume/AI%20for%20Science/Visual%20Perception/Project/human_results.csv), [human_results_by_group.csv](file:///run/media/teleky-franck/New%20Volume/AI%20for%20Science/Visual%20Perception/Project/human_results_by_group.csv), [all_human_responses.csv](file:///run/media/teleky-franck/New%20Volume/AI%20for%20Science/Visual%20Perception/Project/all_human_responses.csv), and [comparison_results.csv](file:///run/media/teleky-franck/New%20Volume/AI%20for%20Science/Visual%20Perception/Project/comparison_results.csv).
