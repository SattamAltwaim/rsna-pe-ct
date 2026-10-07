# Project Plan — PE Detection on CT with a Frozen Foundation Model

**Dataset:** RSNA STR Pulmonary Embolism (RSPECT), CT pulmonary angiograms (CTPA, chest CT with contrast dye)
**Goal (v1):** Binary classification per scan — *PE present* vs *no PE* — using a **frozen** pretrained 3D CT backbone + a **linear** classifier. Then understand *what* the model gets wrong and *where* it looks.
**Compute:** Google Colab Pro, 1× A100. Storage: Google Drive.
**Audience:** The owner is an ML/CV engineer with **no medical background**. Every notebook must be readable by someone who knows deep learning but has never opened a CT scan.

---

## 0. Instructions for Claude Code (read first)

### 0.1 Notebook style rules — apply to every notebook

1. **One cell = one task.** A cell loads *or* transforms *or* plots *or* saves — never two of these. Keep code cells short (aim ≤ 15 lines).
2. **Logic lives in the `pe_ct` package, not in cells.** Reusable functions go in `pe_ct/` (`io.py`, `volume.py`, `viz.py`, `embed.py`, `probe.py`, `explain.py`, `storage.py`). Notebook cells only call them. This keeps cells short, and code fixes reach every notebook through `git pull` (see 0.2).
3. **Every plot gets a markdown cell *before* it** with three short parts:
   - **What you're looking at** — axes, colors, units, in plain words.
   - **Why it matters** — what decision or insight this plot supports.
   - **What to look for** — the pattern that would be good / bad / surprising.
4. **Explain medical terms in ML/CV language** the first time they appear (e.g., "a *slice* is one 2D frame of the 3D volume, like a frame of a video"). Each notebook starts with a short glossary of only the terms it uses.
5. **No redundant plots.** If two plots answer the same question, keep the clearer one. Prefer one well-designed multi-panel figure over five similar figures.
6. **Visual by default.** Show images of scans wherever possible; numbers alone are not enough for a first-time CT user.
7. **Every notebook ends with a "What we learned" markdown cell** (3–5 bullets) — filled with the actual observations after running.
8. **Reproducible:** one `CONFIG` cell at the top (paths, seeds, sizes); fixed random seeds; save every figure to `figures/<notebook>/`.
9. **Resumable long jobs:** Colab *will* disconnect. Long loops skip work that is already done and never leave half-written files that look complete (see the storage rules in 0.3).

### 0.2 Code on GitHub, data on Drive

**Code lives in a GitHub repo; data lives on Google Drive.** Claude Code edits and tests the code locally (MacBook, no CUDA) and pushes. Each notebook's first cell clones or pulls the latest code on Colab. Fixing a bug in `pe_ct/` therefore never means editing notebooks by hand.

**Repo:** `github.com/SattamAltwaim/rsna-pe-ct` (create it if missing; confirm the name with the owner). Keep it **public**, like HALO, so Colab can clone without a token. If it must be private, read a GitHub token from Colab Secrets (`userdata.get("GH_TOKEN")`) and never hard-code it.

```
rsna-pe-ct/
├── pe_ct/                 importable package (all logic)
│   ├── __init__.py
│   ├── config.py          Drive paths, bucket URL, seeds, subset sizes
│   ├── io.py              zip index, parallel range-request download
│   ├── volume.py          DICOM → HU volume, z-sorting, slice-label alignment
│   ├── storage.py         shard writer/reader, atomic writes, .done markers
│   ├── viz.py             windowing presets, slice/orthoview/overlay plots
│   ├── embed.py           SPECTRE loading + preprocessing + extraction
│   ├── probe.py           linear probe, metrics, threshold selection
│   └── explain.py         token occlusion heatmaps
├── notebooks/             00_…ipynb to 05_…ipynb (committed with outputs cleared)
├── tests/                 pytest, CPU-only, tiny synthetic volumes + 1 real study fixture
├── pyproject.toml         `pip install -e .`
└── README.md              how to run each notebook on Colab
```

**First cell of every notebook** (adapted from the wiki's Colab setup pattern in `global/tooling.md`). Unlike the HALO version, it also **pulls on re-run**, so a running session picks up newly pushed code without a restart:

```python
import os, sys, subprocess

BRANCH = "main"
REPO = "https://github.com/SattamAltwaim/rsna-pe-ct.git"
DEST = "/content/rsna-pe-ct"

if os.path.exists("/content"):                       # on Colab
    if not os.path.exists(DEST):
        subprocess.run(["git", "clone", "--branch", BRANCH, REPO, DEST], check=True)
    else:
        subprocess.run(["git", "-C", DEST, "fetch", "origin", BRANCH], check=True)
        subprocess.run(["git", "-C", DEST, "reset", "--hard", f"origin/{BRANCH}"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", DEST], check=True)
    if DEST not in sys.path:
        sys.path.insert(0, DEST)
    from google.colab import drive; drive.mount("/content/drive")
else:                                                # local fallback (Mac)
    parent = os.path.abspath(os.path.join(os.getcwd(), ".."))
    if os.path.exists(os.path.join(parent, "pe_ct")) and parent not in sys.path:
        sys.path.insert(0, parent)

%load_ext autoreload
%autoreload 2
import pe_ct
print("pe_ct", pe_ct.__version__, "| commit:",
      subprocess.run(["git", "-C", DEST if os.path.exists(DEST) else "..", "rev-parse", "--short", "HEAD"],
                     capture_output=True, text=True).stdout.strip())
```

Printing the commit hash means every notebook output records which code version produced it. With `autoreload`, re-running the setup cell after a push is enough; a kernel restart is not needed.

**Dev loop:** Claude Code edits `pe_ct/` → runs `pytest` locally on CPU → commits → pushes. The owner re-runs cell 1 on Colab. Code that needs a GPU is written so it also runs on CPU with a tiny input, so tests catch shape and logic bugs locally.

### 0.3 Data layout on Google Drive

```
/MyDrive/rsna-pe/
├── labels/            train.csv, study_labels.parquet, zip_index.parquet
├── splits/            splits.parquet  (study_uid → dev/test, fold)
├── volumes_eda/       shard_000.tar … + shard_000.done.json   (EDA volumes, int16 HU .nii.gz + meta JSON)
├── embeddings/
│   └── spectre/       shard_000.npz … + shard_000.done.json   (cls, patch_tokens, grid_shape, z-ranges)
├── results/           probe outputs, predictions, metrics
└── figures/           all saved plots, by notebook
```

**Storage rules** (following the StarX decision of 2026-07-24 in the wiki: Drive is very slow with many small files, and a half-written file that looks complete is the worst failure):

- **Write in shards, not one file per study.** Group ~50 studies per shard. Build each shard on Colab local disk (`/content/work/`), then copy the finished shard to Drive.
- **A shard only counts once its `.done.json` marker exists.** Write the marker last; it lists the shard's study UIDs and a checksum. On resume, list the markers, skip their studies, and rebuild any shard without a marker from scratch.
- **Atomic writes** for every single-file output (parquet, results, checkpoints): write to a temp name, then `os.replace`.
- **Log failed studies** to `results/failures_<notebook>.csv` with the reason, and don't retry them forever.
- **Check Drive free space at the start of NB01** and print it. The EDA volumes are the only large item (~10–20 GB). Raw DICOMs never touch Drive: they go to Colab local disk and are deleted after processing.

### 0.4 Build order and acceptance checks

| Step | Done when… |
|---|---|
| Repo scaffold | repo pushed; `pip install -e .` works; `pytest` passes locally; cell 1 clones and imports on Colab |
| `00_setup_and_splits` | `splits.parquet` exists; split class ratios match within ±1% |
| `01_download_subset` | EDA volumes on Drive; random 5 volumes look like chests (visual check cell) |
| `02_eda` | all plots render; "What we learned" filled in |
| `03_extract_embeddings` | every dev+test study is in a shard with a `.done.json` marker; failures logged with reason |
| `04_linear_probe` | metrics table + all plots; test set evaluated **once** at the very end |
| `05_errors_and_heatmaps` | misclassification gallery + heatmaps for top FN/FP |

---

## 1. Background the notebooks should explain (once, in NB00 or NB02)

Write this as a friendly intro markdown cell. Key points:

- **A CT scan is a 3D grayscale volume**, shape ≈ `(slices, 512, 512)`. Closest CV analogy: a video clip where frames are cross-sections from neck to belly.
- **Pixel values are Hounsfield Units (HU)** — a calibrated physical density scale. Air ≈ −1000, fat ≈ −100, water 0, soft tissue ≈ +40, contrast-filled blood ≈ +200 to +400, bone ≈ +1000. Same number = same tissue on any scanner.
- **Windowing** = clip to an HU range and rescale to [0,1] for display or for a model. Different windows reveal different tissue (lung window vs vessel window).
- **Voxel spacing** = physical size of a voxel in mm. Varies by scanner, so models usually resample to a fixed spacing — the 3D version of resizing.
- **DICOM** = medical image file format: one file per slice, with a metadata header. Hierarchy: **Study** (one exam = one sample) → **Series** (one 3D reconstruction; exactly one per study here) → **SOP Instance** (one slice file).
- **Pulmonary embolism (PE)** = a blood clot blocking arteries in the lungs. Dangerous and urgent.
- **CTPA** = chest CT taken right after injecting contrast dye, so blood in the lung arteries appears **bright**.
- **A clot looks like a dark spot inside a bright vessel** (radiologists call it a *filling defect*) — the clot blocks the dye. In CV terms: small, low-contrast dark objects inside thin bright tubes, somewhere in a large 3D volume.
- **Important limitation of this dataset:** labels say *which slices* contain a clot, but **not where in the slice**. There are no bounding boxes or masks. So we can locate clots along the head-to-feet axis only.

---

## 2. Facts already verified about the data (use these, re-verify in NB00)

- Source: public AWS bucket `s3://pulmonary-embolism-detection/rsna-ped-dataset.zip` (us-west-2), **no Kaggle login needed**. Single zip, **~534 GB**, ~1.95M entries. Readable over HTTP range requests (e.g., `remotezip`), so individual files can be fetched without downloading the whole zip.
  HTTPS URL: `https://pulmonary-embolism-detection.s3.us-west-2.amazonaws.com/rsna-ped-dataset.zip`
- Zip contents: `train.csv` (120 MB), `test.csv`, `sample_submission.csv`, folders `train/<StudyUID>/<SeriesUID>/<SOPUID>.dcm` and `test/...`. **`test.csv` has no labels → use only `train/`.**
- `train.csv`: **7,279 studies, 1,790,594 slices**, one row per slice. Exactly **1 series per study**. Median **234 slices/study**. ~70 MB of DICOM per study.
- All study-level columns are constant within a study.
- Study groups (mutually exclusive):

| Group | Count | Encoding |
|---|---|---|
| Negative | 4,911 | `negative_exam_for_pe == 1` |
| PE positive | **2,211 (30.4% of usable)** | `negative_exam_for_pe == 0 and indeterminate == 0` |
| Indeterminate | 157 | `indeterminate == 1` (also has `negative_exam_for_pe == 0`! — don't count as positive) |

- Indeterminate studies are the poor-quality ones: `qa_motion` (63), `qa_contrast` (122). **Drop all 157.**
- Slice label `pe_present_on_image`: 5.4% of all slices positive. PE studies have a median of **34** positive slices (range 1–242). Negative studies have **zero** positive slices.
- Sub-labels exist only for PE-positive studies: `rv_lv_ratio_gte_1` / `rv_lv_ratio_lt_1` (heart strain — bigger, more dangerous clots), `central_pe`, `leftsided_pe`, `rightsided_pe` (location; not exclusive), `chronic_pe`, `acute_and_chronic_pe` (age of clot; acute is the default ≈ 82%).
- Other flags: `flow_artifact`, `true_filling_defect_not_pe` (image artifacts / clot look-alikes — potential hard negatives).

**Binary target:** `y = 1` if PE positive, `y = 0` if negative. Indeterminate excluded. Usable: 7,122 studies, 30.4% positive.

---

## 3. Experiments

### E0 — Setup & splits → `00_setup_and_splits.ipynb`

**Goal:** fixed, leak-free splits decided *once*, before looking at any images.

Cells:
1. CONFIG cell (paths, seed, subset sizes).
2. Mount Drive, create folder layout, print free space.
3. Read `train.csv` from the zip (remotezip) → save to `labels/`.
4. Build the **zip index**: one row per DICOM entry with `filename, header_offset, compress_size, file_size, compress_type`. Opening the 1.95M-entry central directory is slow, so do it once and save as `labels/zip_index.parquet`. Print the compression type counts (stored vs deflate).
5. Collapse to one row per study → `study_labels.parquet` with: `study_uid, series_uid, n_slices, y, group, n_pos_slices, <all sub-labels>, <QA flags>`.
6. Drop indeterminate studies; print class counts.
7. **Freeze the test pool:** 20% of usable studies, stratified by `y` and by sub-type (`central_pe`, `rv_lv_ratio_gte_1`). This pool is never looked at until NB04's final cell. (Each study is a different patient within a hospital in this dataset, so study-level split ≈ patient-level split.)
8. From the remaining 80% (the **dev pool**), sample the **working subsets** (see sizes below), stratified the same way.
9. Plot: class balance per split (grouped bar) — confirms stratification worked.
10. Save `splits/splits.parquet`.

**Subset sizes (v1):**

| Subset | Size | Drawn from | Used in |
|---|---|---|---|
| EDA | 300 studies (150 PE / 150 negative; PE half covers all sub-types) | dev pool | NB02 |
| Dev (train+val) | 2,000 studies | dev pool (EDA studies included) | NB03, NB04 |
| Test sample | 600 studies | frozen test pool | NB03, NB04 final eval |

Within dev: 5-fold stratified CV (store `fold` column). Can grow later to all 7,122 — embeddings are tiny, only extraction time matters.

---

### E1 — Download the subset → `01_download_subset.ipynb`

**Goal:** turn zipped DICOM slices into clean 3D volumes in HU, with slice labels aligned to the volume's z-axis.

**Two download modes (same code path, different outputs):**
- **EDA mode (300 studies):** each volume (int16 HU `.nii.gz`) and its meta JSON go into a ~50-study tar shard in `volumes_eda/`, built locally, then copied to Drive with a `.done.json` marker. `pe_ct.storage` gives a `load_volume(uid)` reader, so notebooks never handle tar files directly. For speed, NB02 copies the EDA shards to Colab local disk once at its start.
- **Streaming mode (used by NB03):** fetch → build volume → hand to model → delete. Nothing large kept.

Pipeline per study (each step = one function in `pe_ct/io.py` / `pe_ct/volume.py`):
1. Look up the study's DICOM entries in `zip_index.parquet`.
2. Fetch them with **parallel HTTP range requests** (thread pool, ~16–32 workers) to Colab local disk. Retry with backoff. Verify byte counts.
3. Read the series with **SimpleITK `ImageSeriesReader`** (handles slice sorting, spacing and orientation → correct 3D geometry). If some DICOMs use compressed pixel encodings, install the needed decoders (`pylibjpeg`, `gdcm`) — test on 20 random studies first and log any failures.
4. Pixel values → HU (`RescaleSlope`, `RescaleIntercept`; SimpleITK applies these — verify on one study by checking that air ≈ −1000 and blood in the aorta ≈ +200–400).
5. **Align slice labels to z-index:** map each `SOPInstanceUID` to its sorted z-position, so `slice_labels[z]` matches `volume[z]`. Store as an array in the meta JSON. **This alignment is critical** — NB02's clot plots and NB05's heatmap checks depend on it.
6. Save metadata: spacing, shape, orientation, scanner `Manufacturer`, `SliceThickness`, `KVP`, z-sorted slice labels, sub-labels.
7. Delete raw DICOMs.

Cells:
1. CONFIG + Drive free space check.
2. Load splits; select EDA subset.
3. Test the pipeline on **1 study**, print shape/spacing/HU range.
4. **Visual sanity check:** one study as 3 orthogonal views (axial / coronal / sagittal). *Markdown: explain the three viewing planes — top-down, front, side — using the "loaf of bread" analogy.*
5. Run the EDA subset loop (resumable, with a progress bar, logs failures to CSV).
6. Print summary: done / failed / time per study / GB written. **Report measured seconds per study** — this sets the realistic budget for NB03.
7. Visual check: 5 random volumes, coronal view grid.

---

### E2 — Exploratory data analysis → `02_eda.ipynb`

**Goal:** a first-time CT user understands what the data looks like, what a clot looks like, and where clots occur.

Glossary cell first (slice, axial/coronal/sagittal, HU, window, CTPA, filling defect, sub-labels).

**A. The labels (from `study_labels.parquet`, all 7,279 studies)**
1. **Study groups bar chart** (negative / PE / indeterminate). Why: shows the class imbalance we must handle.
2. **Sub-label prevalence among PE studies** (horizontal bar). Why: tells us which kinds of PE are common vs rare.
3. **Sub-label co-occurrence heatmap** (PE studies only). Why: e.g., central clots often come with heart strain — the labels are related, not independent.
4. **Positive slices per PE study** (histogram, log x-axis). Why: some scans have 1 positive slice (tiny clot, hard), others 200 (huge clot, easy). This spread predicts difficulty.

**B. The scans (EDA subset, 300 volumes)**
5. **Interactive slice scroller** (ipywidgets slider over z; dropdown to pick study; shows axial slice + a red bar marking positive slices along z). Why: the single most useful way to "get" a CT.
6. **Windowing comparison**: one axial slice through the lung arteries shown with 4 windows — raw full range, lung window (L −600 / W 1500), soft-tissue window (L 40 / W 400), vessel/PE window (L 100 / W 700). Why: the clot is invisible in some windows and visible in others; explains why preprocessing matters.
7. **HU histogram** of a few volumes, with vertical lines at air, fat, soft tissue, contrast blood, bone. Why: shows the intensity peaks match physical tissue, i.e., the data is calibrated correctly.
8. **Acquisition variability**: histograms of slice count, slice spacing (mm), in-plane spacing (mm), and a bar chart of scanner manufacturers. Why: shows how inconsistent raw scans are — motivates resampling — and flags a possible *shortcut* (if one manufacturer has more PE cases, a model could learn the scanner instead of the disease).

**C. What a clot looks like**
9. **Healthy vs PE side-by-side** (2×N grid): PE studies at a positive slice vs negative studies at a similar anatomical level (match by relative z-position), vessel window. Markdown must guide the eye: "look inside the bright branching vessels near the center for darker gray spots."
10. **Zoomed crop of the central chest region** (pulmonary arteries) for the same PE examples, vessel window, larger. Why: clots are small; full slices hide them.
11. **Clot sequence**: for one PE study, a strip of consecutive slices crossing from negative → positive → negative. Why: shows the clot appearing and disappearing along z, like an object entering and leaving a video.
12. **Central vs peripheral examples** (one `central_pe` study, one PE study without `central_pe`). Why: shows the easy vs hard end of the problem.
13. **Look-alikes**: examples flagged `flow_artifact` or `true_filling_defect_not_pe` if present in the subset. Why: these fool humans and models — future hard negatives.

**D. Where along the body clots appear (uses all 2,211 PE studies from `train.csv`, no images needed except the z-order)**
14. **Clot position distribution**: histogram of positive-slice positions normalized to [0 = top of scan, 1 = bottom]. Overlay the same for `central_pe` vs non-central. Why: tells us which part of the volume the model must pay attention to.
    *Note for Claude Code:* the normalization needs the z-sorted order. For EDA-subset studies use the aligned labels from meta JSON. For the full 2,211, check whether `train.csv` row order already follows slice position on the EDA subset; if it does not, compute the plot on the EDA subset only and say so in the markdown.
15. **"Barcode" plot**: rows = PE studies (sorted by number of positive slices), columns = normalized z, colored where a slice is positive. Why: one picture showing clot extent and position across many patients — contiguous blocks vs scattered.

"What we learned" cell.

---

### E3 — Extract frozen embeddings → `03_extract_embeddings.ipynb`

**Backbone: SPECTRE** (`cclaess/SPECTRE-Large` on Hugging Face). Chosen because it was pretrained on **chest and abdomen CT** with a 3D ViT, provides **both** a whole-scan vector and per-crop tokens (needed for heatmaps), and is a current, well-benchmarked CT foundation model. License: CC-BY-NC-SA (research use OK, not commercial).
*Optional later comparison:* CT-FM (`project-lighter/ct_fm_feature_extractor`, Apache-2.0). Merlin is excluded (abdomen-only pretraining).

**Match the pretraining preprocessing exactly** (from the SPECTRE README — Claude Code must re-check the current README/model card and use the library's own helpers rather than reimplementing):
- Input: raw volume in **HU**. The model's helpers do the windowing to **[−1000, 1000]**.
- Orientation: **RAS**.
- The scan is tiled into **128 × 128 × 64 crops** at native voxel spacing (resampling optional via `spacing`). Record whether we resample; pick one setting and keep it fixed for all studies.
- Outputs: `cls` = one vector per scan; `patch_tokens` = one vector per crop, plus the crop **grid shape** (needed to map tokens back to positions).

Cells:
1. CONFIG; install `spectre-fm[inference]`; load model in fp16/bf16 on GPU; print parameter count and output dims.
2. **"What the model sees"** figure for one study: original vessel-window coronal view vs model input (after orientation + windowing), with the **128×128×64 crop grid drawn on top**. Markdown: each grid box becomes one token; the model sees the scan as a set of boxes, and the heatmap later will be at this box resolution.
3. Run on 3 studies; print shapes and timing (GPU time vs download+decode time separately).
4. Main loop over dev (2,000) + test sample (600) in **streaming mode**: fetch → volume → embed → add to the current shard (`cls`, `patch_tokens`, `grid_shape`, crop z-ranges in slice indices, `y`, `uid`) → delete DICOMs. Each ~50-study shard is copied to Drive with its `.done.json` marker. Resumable, failures logged.
   *Speed tip:* run download/decoding for the next studies in background threads while the GPU embeds the current one (the CPU side is the bottleneck, not the A100).
5. Summary: count, failures, mean time per study.
6. **Embedding sanity plot**: UMAP of `cls` vectors, two panels — colored by PE label, colored by scanner manufacturer. Why: if points cluster strongly by *manufacturer*, the embedding encodes the scanner more than the anatomy (a shortcut risk). Some separation by PE label is a good sign; none is expected-ish for subtle disease and is not a failure.

---

### E4 — Linear probe → `04_linear_probe.ipynb`

**Goal:** how much PE information is linearly readable from the frozen scan embedding.

Model: **one `nn.Linear(d, 1)`** on standardized `cls` embeddings (PyTorch, so we get train/val curves). BCE loss with `pos_weight = n_neg / n_pos` to handle imbalance; AdamW with weight decay; early stopping on val AUPRC. Also fit `sklearn LogisticRegression(class_weight="balanced")` as a sanity baseline — both should agree roughly.
Feature standardization: fit mean/std on **train fold only**.

Cells and plots:
1. Load embeddings + labels + folds; build arrays; print shapes and class ratio per split.
2. Train on fold 0 (train = folds 1–4, val = fold 0).
3. **Train vs val curves** (loss and AUPRC per epoch). Why: shows over/underfitting. Explain: val loss rising while train falls = memorizing.
4. **Score distribution per class** (overlapping histograms of predicted probability, val set). Why: the cleanest picture of separability — two humps far apart = easy, overlapping = hard.
5. **ROC + PR curves** (side by side; PR includes a horizontal line at prevalence ≈ 0.30 = "random guess" baseline). Explain why PR is more honest than ROC under imbalance.
6. **Threshold sweep**: precision, recall, F1 vs threshold; mark the threshold that maximizes F1 on **val**. Why: 0.5 is arbitrary; this picks a sensible operating point.
7. **Confusion matrix** at the chosen threshold (counts + row-normalized).
8. **Per-class precision / recall / F1 table** (`classification_report`), for both classes, train vs val side by side.
9. **5-fold CV summary**: mean ± std of AUROC, AUPRC, F1, recall. Why: one split can be lucky; this shows stability.
10. **Learning curve**: train on 250 / 500 / 1,000 / 1,600 studies, evaluate on the same val fold. Why: tells us whether downloading more data would help (still rising) or not (flat).
11. **Recall by PE sub-type** (bar chart with counts): central vs non-central, RV/LV ≥ 1 vs < 1, chronic vs acute, and bins of positive-slice count (1–10, 11–50, 51+). Why: shows *which* PEs the model catches — expected: big/central easy, small/peripheral hard.
12. **Final test evaluation — run once, at the very end:** retrain on all dev, apply the val-chosen threshold, report the same metrics on the 600-study test sample with bootstrap 95% confidence intervals.
13. Save predictions (`uid, y, prob, pred, fold/test`) to `results/` for NB05.

"What we learned" cell.

---

### E5 — Errors and heatmaps → `05_errors_and_heatmaps.ipynb`

**Goal:** see which scans the model gets wrong and where it "looks".

**A. Misclassification analysis**
1. Load predictions + labels + meta.
2. **Error breakdown table**: FN and FP counts, and how often errors carry each flag/sub-type (`flow_artifact`, `true_filling_defect_not_pe`, small positive-slice count, scanner manufacturer). Compare to the rate among correct predictions. Why: finds systematic failure modes.
3. **Gallery: top-8 most confident false negatives** (PE scans scored lowest). Each tile: the most representative positive slice (middle of the positive run) in vessel window, plus a coronal view with the positive z-band shaded, and a caption with score, number of positive slices, sub-labels. Markdown: what these have in common?
4. **Gallery: top-8 most confident false positives** (healthy scans scored highest). Same layout, captions show artifact flags. Markdown: what could the model be reacting to?

These galleries use studies that need volumes on hand: re-fetch them with the streaming pipeline (only ~16 studies).

**B. Heatmaps — where the model looks**

Method: **crop-level occlusion on tokens** (cheap, no backbone re-run):
- For a study, take its `patch_tokens` (already saved). For each crop *i*, remove/mask that token, run **only SPECTRE's feature combiner** (the small global transformer) + the trained linear layer, and record the drop in PE score.
- Importance of crop *i* = score(all crops) − score(without crop *i*). Big drop = the model relied on that region.
- Map importances back to the 3D crop grid → upsample → overlay on the scan.
- Claude Code: confirm from the SPECTRE code how to run the combiner on a subset of tokens (masking vs removal) and keep the choice consistent.
- **Resolution warning for the markdown:** one crop is 128×128×64 voxels, so the heatmap is coarse (a few dozen boxes per scan). It shows *which region*, not *which pixel*.

Plots:
5. **Heatmap overlay figure** for 2 true positives, the top FNs, and the top FPs: coronal + axial views, vessel window, importance heatmap overlaid semi-transparently, positive z-band (ground truth) outlined in green.
6. **"Did it look in the right place?" check** (true positives + FNs): importance profile along z (sum over in-plane crops) plotted against the ground-truth positive-slice band. Because labels have no in-plane location, "right place" can only be judged along z.
   Metric: *z-hit rate* = fraction of PE scans where the highest-importance crop overlaps the positive slice range. Compare to a random-crop baseline.
7. **Missed-focus examples**: FNs where the ground-truth band got low importance. Markdown: these are the "should have looked here but didn't" cases the owner asked for.

"What we learned" + **next-step recommendations** cell, based on the results.

---

## 4. Planned follow-ups (not in v1 — decide after E4/E5)

- **E6 — Second backbone:** repeat E3–E4 with CT-FM; paired comparison (DeLong test) on the same studies.
- **E7 — Attention/MIL head on crop tokens:** instead of the single `cls` vector, learn attention weights over `patch_tokens` (better for small clots; attention weights give a built-in heatmap). Optionally supervise attention with the slice-level labels.
- **E8 — More data:** if the learning curve in E4 is still rising, extract embeddings for the remaining dev pool (embeddings are small; cost is ~CPU time).
- **E9 — Light fine-tuning** if frozen features plateau too low.

---

## 5. Known risks and how the plan handles them

| Risk | Mitigation |
|---|---|
| Slice order / HU conversion bugs silently corrupt everything | SimpleITK reader + visual checks in NB01 + HU histogram in NB02 |
| Slice labels misaligned with the volume | explicit SOPInstanceUID → z mapping, verified visually (clot sequence plot) |
| Colab disconnects mid-run | marker-gated shards (incomplete shards rebuilt), atomic writes, failures logged |
| Notebooks run stale code | cell 1 fetches and resets to the latest pushed commit and prints the commit hash |
| CPU is the bottleneck, not GPU | parallel downloads + background prefetch; measure seconds/study in NB01 |
| Test-set leakage | test pool frozen in NB00; touched once at the end of NB04 |
| Model learns scanner, not disease | manufacturer-colored UMAP + error breakdown by manufacturer |
| Global embedding misses small clots | expected; measured by recall-by-subtype; addressed by E7 |
| Drive fills up | only EDA volumes stored; free-space check at start |
