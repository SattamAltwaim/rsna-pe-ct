"""02_finetune: self-contained Kaggle notebook (no repo clone); every function is visible in the cells."""

from scripts.nb.tools import code, learned_cell, md, plot_note

TITLE = "02_finetune"


def cells():
    return [
        md("""
        # 02 - Fine-tune SPECTRE for PE detection (self-contained)

        **Goal.** Train a PE / no-PE classifier on a few thousand CT studies by adapting a pretrained
        3D CT foundation model, and report honest numbers: loss, accuracy, macro F1, confusion matrix,
        with early stopping and a cosine learning-rate schedule, on group-stratified folds.

        **Plan.** SPECTRE has two parts: a ViT-L *backbone* that turns each 128 x 128 x 64 crop of a
        scan into a 2160-d descriptor, and a small *feature combiner* transformer that turns the ~50
        crop descriptors of a scan into one scan embedding. The backbone is strong and our subset is
        small, so it stays frozen and runs **once** per study (stage A, the only slow part; each
        study's descriptors are saved as one small `.npz`, so a dropped session resumes). Training
        (stage B) then touches only the combiner, through **LoRA** adapters by default or its **last
        blocks**, plus a new MLP head; an epoch takes seconds, so 5-fold cross-validation is cheap.

        Everything the notebook needs is defined in its own cells; only the libraries are installed.

        Glossary: **LoRA** = train a small low-rank update `B A x` next to each frozen weight matrix;
        **cosine schedule** = learning rate warms up, then decays along a cosine; **early stopping** =
        stop when the validation metric stops improving and keep the best weights;
        **StratifiedGroupKFold** = folds with the same class mix where every group (study = patient)
        lands in exactly one fold; **macro F1** = F1 averaged over both classes (headline metric).
        """),
        md("## Install and imports"),
        code("""
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "spectre-fm", "pydicom", "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg", "SimpleITK"], check=True)
        """),
        code("""
        import io, json, math, os, time, copy, warnings
        from concurrent.futures import ThreadPoolExecutor
        from pathlib import Path

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import pydicom
        import SimpleITK as sitk
        import torch
        from torch import nn
        from tqdm.auto import tqdm
        from sklearn.model_selection import StratifiedGroupKFold, train_test_split
        from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix
        warnings.filterwarnings("ignore", message="Invalid value for VR UI")   # the dataset's anonymised UIDs
        plt.rcParams.update({"figure.dpi": 100, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
        PALETTE = plt.get_cmap("tab10").colors
        """),
        md("""
        ## Configuration

        Paths, subset sizes and hyperparameters. The test pool is a fixed 20 % of the usable studies
        and is only touched once at the very end; `N_DEV` and `N_TEST` control how many studies are
        actually embedded (lower them for a faster first run).
        """),
        code("""
        INPUT = Path("/kaggle/input/competitions/rsna-str-pulmonary-embolism-detection")
        OUT = Path("/kaggle/working/rsna-pe"); DESC_DIR = OUT / "descriptors"; FIG = OUT / "figures"
        for d in [DESC_DIR, FIG]: d.mkdir(parents=True, exist_ok=True)
        SEED = 0
        N_DEV, N_TEST, TEST_FRAC, N_FOLDS = 2000, 600, 0.20, 5
        READ_THREADS, PREFETCH = 32, 3
        FINETUNE_MODE = "lora"          # "lora" | "last_blocks" | "head_only"
        LORA_RANK, N_LAST_BLOCKS = 8, 2
        EPOCHS, BATCH_SIZE, PATIENCE, WARMUP_EPOCHS = 40, 32, 6, 2
        LR_HEAD, LR_ADAPT, WEIGHT_DECAY, CROP_DROPOUT = 1e-3, 2e-4, 1e-2, 0.1
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        DTYPE = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if DEVICE == "cuda" else torch.float32
        print("device", DEVICE, DTYPE, "| GPU:", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "none")
        """),
        md("""
        ## Labels and splits

        `train.csv` has one row per slice; it is collapsed to one row per study. Three groups:
        negative, PE, indeterminate (poor-quality scans, dropped). The target is `y = 1` for PE.
        The split is stratified on the label and two clinically important sub-types (central clot,
        heart strain) so every subset has the same mix; it is seeded, so it is identical every run.
        """),
        code("""
        SUBLABELS = ["rv_lv_ratio_gte_1", "rv_lv_ratio_lt_1", "central_pe", "leftsided_pe", "rightsided_pe", "chronic_pe", "acute_and_chronic_pe"]
        QA = ["qa_motion", "qa_contrast", "flow_artifact", "true_filling_defect_not_pe"]

        train = pd.read_csv(INPUT / "train.csv")
        g = train.groupby("StudyInstanceUID")
        studies = g.agg(series_uid=("SeriesInstanceUID", "first"), n_slices=("SOPInstanceUID", "size"), n_pos_slices=("pe_present_on_image", "sum"),
                        **{c: (c, "first") for c in ["negative_exam_for_pe", "indeterminate"] + SUBLABELS + QA}).reset_index().rename(columns={"StudyInstanceUID": "study_uid"})
        studies["group"] = np.where(studies.indeterminate == 1, "indeterminate", np.where(studies.negative_exam_for_pe == 1, "negative", "pe"))
        studies["y"] = (studies.group == "pe").astype(int)
        usable = studies[studies.group != "indeterminate"].reset_index(drop=True)
        print(len(studies), "studies |", len(usable), "usable |", f"{usable.y.mean():.1%} PE")
        """),
        code("""
        def strat_key(df):
            return df[["y", "central_pe", "rv_lv_ratio_gte_1"]].astype(int).astype(str).agg("-".join, axis=1)

        def stratified_sample(df, n, seed):
            if n >= len(df): return df
            key = strat_key(df); counts = key.value_counts(); ok = ~key.isin(counts[counts < 2].index)
            try: picked, _ = train_test_split(df[ok], train_size=n, stratify=key[ok], random_state=seed)
            except ValueError: picked = df[ok].sample(n=n, random_state=seed)
            return picked

        dev_pool_idx, test_pool_idx = train_test_split(np.arange(len(usable)), test_size=TEST_FRAC, stratify=strat_key(usable), random_state=SEED)
        dev_pool, test_pool = usable.iloc[dev_pool_idx], usable.iloc[test_pool_idx]
        dev = stratified_sample(dev_pool, N_DEV, SEED + 1)
        test = stratified_sample(test_pool, N_TEST, SEED)
        assert not set(dev.study_uid) & set(test.study_uid)
        print(f"dev {len(dev)} ({dev.y.mean():.1%} PE) | test {len(test)} ({test.y.mean():.1%} PE), untouched until the end")
        """),
        md("""
        ## Reading one CT study

        A study is a folder of DICOM files, one per slice. The files are read in parallel (the
        Kaggle mount has high per-file latency), sorted by physical position along the slice
        normal, converted to Hounsfield units with each file's rescale slope/intercept, and
        assembled into a 3D image with the right voxel spacing and orientation. SPECTRE wants the
        volume in **RAS** orientation (axis 0 toward the patient's right, 1 toward anterior, 2
        toward the head) as a `(R, A, S)` array in HU.
        """),
        code("""
        def read_dicom_files(files, threads=READ_THREADS):
            with ThreadPoolExecutor(max_workers=max(1, min(threads, len(files)))) as pool:
                blobs = list(pool.map(lambda f: Path(f).read_bytes(), files))
            return [pydicom.dcmread(io.BytesIO(b)) for b in blobs]

        def study_dir(uid, series_uid):
            return INPUT / "train" / uid / series_uid

        def read_study_ras(uid, series_uid):
            \"\"\"-> (R, A, S) float32 HU array of one study.\"\"\"
            ds_list = read_dicom_files(sorted(study_dir(uid, series_uid).glob("*.dcm")))
            iop = np.array(ds_list[0].ImageOrientationPatient, float); row_dir, col_dir = iop[:3], iop[3:]
            normal = np.cross(row_dir, col_dir); normal /= np.linalg.norm(normal); col_dir = np.cross(normal, row_dir)
            positions = np.array([d.ImagePositionPatient for d in ds_list], float)
            order = np.argsort(positions @ normal, kind="stable"); ds_list, positions = [ds_list[i] for i in order], positions[order]
            vol = np.stack([d.pixel_array.astype(np.float32) * float(d.get("RescaleSlope", 1)) + float(d.get("RescaleIntercept", 0)) for d in ds_list])
            dz = float(np.median(np.abs(np.diff(positions @ normal)))) if len(ds_list) > 1 else float(ds_list[0].get("SliceThickness", 1))
            img = sitk.GetImageFromArray(vol)                                        # (z, y, x) -> sitk (x, y, z)
            img.SetSpacing((float(ds_list[0].PixelSpacing[1]), float(ds_list[0].PixelSpacing[0]), dz))
            img.SetOrigin(tuple(float(v) for v in positions[0]))
            img.SetDirection(tuple(float(v) for v in np.stack([row_dir, col_dir, normal], axis=1).reshape(-1)))
            ras = sitk.GetArrayFromImage(sitk.DICOMOrient(img, "RAS"))              # (S, A, R)
            return np.ascontiguousarray(ras.transpose(2, 1, 0))                       # (R, A, S)
        """),
        md("""
        ## The backbone, loaded once

        The published weights are plain PyTorch state dicts; they are downloaded from the Hub and
        loaded directly (recent `huggingface_hub` versions refuse pickle files through the library's
        own loader).
        """),
        code("""
        from spectre import SpectreImageFeatureExtractor, window_scan
        from spectre.presets import get_preset
        from huggingface_hub import hf_hub_download

        def load_spectre(name="spectre-large"):
            model = SpectreImageFeatureExtractor.from_pretrained(name, pretrained=False)   # architecture only
            preset = get_preset(name)
            for module, url in [(model.backbone, preset.backbone_weights), (model.feature_combiner, preset.feature_combiner_weights)]:
                filename = url.split("/")[-1].split("?")[0]
                state = torch.load(hf_hub_download(repo_id="cclaess/SPECTRE", filename=filename), map_location="cpu", weights_only=True)
                module.load_state_dict(state, strict=True)
            return model.to(device=DEVICE, dtype=DTYPE).eval()

        model = load_spectre()
        print(f"backbone {sum(p.numel() for p in model.backbone.parameters())/1e6:.0f}M params | combiner {sum(p.numel() for p in model.feature_combiner.parameters())/1e6:.0f}M | crop {model.crop_size}")
        """),
        md("""
        ## From a volume to crop descriptors

        `window_scan` (the library's own preprocessing) maps HU to [0, 1] and tiles the volume into
        non-overlapping 128 x 128 x 64 crops laid out on a grid `(n_R, n_A, n_S)`. Each crop goes
        through the backbone; its 513 tokens are pooled to one 2160-d descriptor (CLS ++ mean of
        the patch tokens). The combiner then needs only these descriptors plus the grid shape.
        """),
        code("""
        @torch.inference_mode()
        def crop_descriptors(ras_hu, max_crops_per_forward=16):
            crops, grid = window_scan(torch.from_numpy(ras_hu).unsqueeze(0))              # (N, 1, 128, 128, 64), (nR, nA, nS)
            out = []
            for chunk in torch.split(crops, max_crops_per_forward):
                tokens = model.backbone(chunk.to(device=DEVICE, dtype=DTYPE))             # (n, 513, 1080)
                out.append(torch.cat([tokens[:, 0], tokens[:, 1:].mean(1)], dim=-1))        # (n, 2160)
            desc = torch.cat(out)
            cls = model.feature_combiner(desc.unsqueeze(0), tuple(grid))[0, 0]              # the frozen scan embedding, for reference
            return desc.float().cpu().numpy().astype(np.float16), np.array(grid), cls.float().cpu().numpy()

        with torch.inference_mode():
            air_crops, _ = window_scan(torch.full((1, *model.crop_size), -1000.0))
            AIR = torch.cat([model.backbone(air_crops.to(DEVICE, DTYPE))[:, 0], model.backbone(air_crops.to(DEVICE, DTYPE))[:, 1:].mean(1)], -1)[0].float().cpu().numpy()
        print("air-crop descriptor (used as the crop-dropout filler):", AIR.shape)
        """),
        md("""
        ## Stage A: embed every dev and test study once

        One `.npz` per study in the output folder; a study whose file exists is skipped, so the
        loop resumes after a disconnect. Reading and decoding of the next studies happens in
        background threads while the GPU embeds the current one. The first cell times three
        studies and prints the projected total.
        """),
        code("""
        todo = pd.concat([dev.assign(split="dev"), test.assign(split="test")]).reset_index(drop=True)
        def desc_path(uid): return DESC_DIR / f"{uid}.npz"

        def embed_study(row):
            t0 = time.time(); ras = read_study_ras(row.study_uid, row.series_uid); t1 = time.time()
            desc, grid, cls = crop_descriptors(ras)
            if DEVICE == "cuda": torch.cuda.synchronize()
            return {"desc": desc, "grid": grid, "cls": cls, "y": int(row.y), "shape": np.array(ras.shape), "t_read": t1 - t0, "t_gpu": time.time() - t1}

        rows = []
        for row in todo.itertuples():
            if len(rows) == 3: break
            r = embed_study(row); rows.append({"uid": row.study_uid, "crops": len(r["desc"]), "read+decode_s": round(r["t_read"], 1), "gpu_s": round(r["t_gpu"], 1)})
        rows = pd.DataFrame(rows); per = (rows["read+decode_s"] + rows["gpu_s"]).mean(); print(rows.to_string())
        print(f"about {per:.1f} s per study sequentially -> at most {per * len(todo) / 60:.0f} min for {len(todo)} studies (reads overlap the GPU below, so less)")
        """),
        code("""
        pending = [row for row in todo.itertuples() if not desc_path(row.study_uid).exists()]
        print(len(todo) - len(pending), "already done |", len(pending), "to embed")
        failures, t_read, t_gpu = [], [], []
        with ThreadPoolExecutor(max_workers=PREFETCH) as pool:
            futures = {}
            for row in pending[:PREFETCH]: futures[row.study_uid] = pool.submit(read_study_ras, row.study_uid, row.series_uid)
            for i, row in enumerate(tqdm(pending, desc="embedding")):
                nxt = i + PREFETCH
                if nxt < len(pending): futures[pending[nxt].study_uid] = pool.submit(read_study_ras, pending[nxt].study_uid, pending[nxt].series_uid)
                try:
                    t0 = time.time(); ras = futures.pop(row.study_uid).result(); t1 = time.time()
                    desc, grid, cls = crop_descriptors(ras)
                    tmp = desc_path(row.study_uid).with_suffix(".tmp.npz")
                    np.savez(tmp, desc=desc, grid=grid, cls=cls, y=int(row.y), shape=np.array(ras.shape)); os.replace(tmp, desc_path(row.study_uid))
                    t_read.append(t1 - t0); t_gpu.append(time.time() - t1)
                except Exception as exc:
                    failures.append({"uid": row.study_uid, "error": repr(exc)[:300]})
        pd.DataFrame(failures).to_csv(OUT / "embedding_failures.csv", index=False)
        print(f"done: {len(pending) - len(failures)} | failed: {len(failures)} | wait-for-read {np.mean(t_read) if t_read else 0:.1f} s, GPU {np.mean(t_gpu) if t_gpu else 0:.1f} s per study")
        """),
        md("""
        ## Load the descriptors into memory

        The backbone is no longer needed and is dropped from the GPU; only the pretrained combiner
        stays (it is copied fresh for every fold).
        """),
        code("""
        records = {}
        for row in tqdm(todo.itertuples(), total=len(todo), desc="loading"):
            p = desc_path(row.study_uid)
            if p.exists():
                with np.load(p) as z: records[row.study_uid] = {"desc": z["desc"], "grid": tuple(int(v) for v in z["grid"]), "cls": z["cls"], "y": int(z["y"])}
        todo = todo[todo.study_uid.isin(records)].reset_index(drop=True)
        uids = todo.study_uid.tolist(); Y = todo.y.to_numpy(); GRID = [records[u]["grid"] for u in uids]; DESC = [records[u]["desc"] for u in uids]
        CLS = np.stack([records[u]["cls"] for u in uids]); dev_idx = np.where(todo.split == "dev")[0]; test_idx = np.where(todo.split == "test")[0]
        combiner_pretrained = copy.deepcopy(model.feature_combiner).float(); del model; torch.cuda.empty_cache() if DEVICE == "cuda" else None
        print(len(uids), "studies loaded |", len(dev_idx), "dev |", len(test_idx), "test | descriptor", DESC[0].shape)
        """),
        plot_note(
            "Crop grids per scan",
            "How many scans have each crop grid (R x A x S boxes).",
            "The combiner can only batch scans that share a grid, and scans with few boxes along the head-to-feet "
            "axis have had their top and bottom slices cut off by the tiling.",
            "One or two dominant grids; rare grids just form tiny batches.",
        ),
        code("""
        counts = pd.Series([f"{g[0]}x{g[1]}x{g[2]}" for g in GRID]).value_counts()
        fig, ax = plt.subplots(figsize=(7, 3.5)); ax.bar(counts.index, counts.values, color=PALETTE[0]); ax.set_title("crop grid (R x A x S) per scan")
        fig.savefig(FIG / "crop_grids.png", dpi=130, bbox_inches="tight")
        """),
        md("""
        ## Group-stratified folds over the dev subset

        Stratified on the label; the group is the study (one patient), so no study is split
        across folds. Each fold takes a turn as the validation set.
        """),
        code("""
        folds = np.full(len(dev_idx), -1)
        for f, (_, va) in enumerate(StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(dev_idx)), Y[dev_idx], np.array(uids)[dev_idx])):
            folds[va] = f
        pd.DataFrame({"n": pd.Series(folds).value_counts().sort_index(), "pe_frac": pd.Series(Y[dev_idx]).groupby(folds).mean().round(3)})
        """),
        md("""
        ## Reference: frozen embedding + logistic regression

        One number to beat before any fine-tuning: a linear classifier on the frozen scan embedding,
        validated on fold 0.
        """),
        code("""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        tr, va = dev_idx[folds != 0], dev_idx[folds == 0]
        sc = StandardScaler().fit(CLS[tr]); lin = LogisticRegression(class_weight="balanced", C=0.5, max_iter=5000).fit(sc.transform(CLS[tr]), Y[tr])
        p_lin = lin.predict_proba(sc.transform(CLS[va]))[:, 1]
        """),
        md("""
        ## Metrics

        Accuracy and macro F1 at a threshold, plus the threshold-free AUROC and AUPRC; a sweep to
        pick the operating threshold; a bootstrap for confidence intervals.
        """),
        code("""
        def metrics(y, prob, thr=0.5):
            pred = (prob >= thr).astype(int); tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            return {"accuracy": (tp + tn) / len(y), "macro_f1": f1_score(y, pred, average="macro", zero_division=0), "f1_pe": f1_score(y, pred, zero_division=0),
                    "recall_pe": tp / max(1, tp + fn), "precision_pe": tp / max(1, tp + fp), "auroc": roc_auc_score(y, prob), "auprc": average_precision_score(y, prob)}

        def best_threshold(y, prob):
            ts = np.linspace(0.05, 0.95, 91); return float(ts[np.argmax([metrics(y, prob, t)["macro_f1"] for t in ts])])

        def bootstrap(y, prob, thr, n=1000, seed=SEED):
            rng = np.random.default_rng(seed); rows = []
            for _ in range(n):
                i = rng.integers(0, len(y), len(y))
                if len(set(y[i])) == 2: rows.append(metrics(y[i], prob[i], thr))
            rows = pd.DataFrame(rows); point = metrics(y, prob, thr)
            return pd.DataFrame({"value": point, "ci_lo": rows.quantile(0.025), "ci_hi": rows.quantile(0.975)}).round(4)

        def confusion_fig(y, prob, thr, title):
            cm = confusion_matrix(y, (prob >= thr).astype(int), labels=[0, 1]); fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
            for ax, mat, sub, fmt in [(axes[0], cm, "counts", "d"), (axes[1], cm / cm.sum(1, keepdims=True), "row-normalized (recall on the diagonal)", ".2f")]:
                ax.imshow(mat, cmap="Blues"); ax.set_title(sub, fontsize=9); ax.set_xticks([0, 1]); ax.set_xticklabels(["pred no PE", "pred PE"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["true no PE", "true PE"])
                for i in range(2):
                    for j in range(2): ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center", color="w" if mat[i, j] > mat.max() / 2 else "k")
            fig.suptitle(title); fig.tight_layout(); return fig

        frozen = metrics(Y[va], p_lin, best_threshold(Y[va], p_lin)); print("frozen + logistic regression (fold 0):", {k: round(v, 3) for k, v in frozen.items()})
        """),
        md("""
        ## What trains: LoRA (or the last blocks) on the combiner, plus an MLP head

        `LoRALinear` keeps the pretrained weight frozen and adds a rank-`r` update `B A x`, with `B`
        starting at zero so the model begins exactly at its pretrained behaviour. It is applied to
        the attention projections (`q`, `kv`, `proj`) of every combiner block. `last_blocks` instead
        unfreezes whole blocks. The final LayerNorm always adapts; the MLP head is new.
        """),
        code("""
        class LoRALinear(nn.Module):
            def __init__(self, base, rank=8, alpha=16.0, dropout=0.05):
                super().__init__(); self.base, self.scale, self.dropout = base, alpha / rank, nn.Dropout(dropout)
                self.lora_a, self.lora_b = nn.Linear(base.in_features, rank, bias=False), nn.Linear(rank, base.out_features, bias=False)
                nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5)); nn.init.zeros_(self.lora_b.weight)
                for p in self.base.parameters(): p.requires_grad_(False)
            def forward(self, x): return self.base(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scale

        class Classifier(nn.Module):
            def __init__(self, combiner, hidden=256, dropout=0.2):
                super().__init__(); self.combiner = combiner; d = combiner.embed_dim
                self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
            def forward(self, desc, grid): return self.head(self.combiner(desc, tuple(grid))[:, 0]).squeeze(-1)
            def trainable_state(self): return {k: v.detach().cpu().clone() for k, v in self.state_dict().items() if k in self.trainable_keys}

        def build_classifier(mode=FINETUNE_MODE):
            combiner = copy.deepcopy(combiner_pretrained)
            for p in combiner.parameters(): p.requires_grad_(False)
            if mode == "lora":
                for blk in combiner.blocks:
                    for name in ["q", "kv", "proj"]: setattr(blk.attn, name, LoRALinear(getattr(blk.attn, name), LORA_RANK))
            elif mode == "last_blocks":
                for blk in list(combiner.blocks)[-N_LAST_BLOCKS:]:
                    for p in blk.parameters(): p.requires_grad_(True)
            for p in combiner.norm.parameters(): p.requires_grad_(mode != "head_only")
            clf = Classifier(combiner).to(DEVICE); clf.trainable_keys = {n for n, p in clf.named_parameters() if p.requires_grad}
            return clf

        clf = build_classifier()
        print(f"trainable {sum(p.numel() for p in clf.parameters() if p.requires_grad)/1e6:.3f}M of {sum(p.numel() for p in clf.parameters())/1e6:.1f}M params ({FINETUNE_MODE})")
        del clf
        """),
        md("""
        ## Training loop

        Batches group scans that share a crop grid (the combiner's position encoding needs that).
        Loss: BCE with a positive-class weight for the imbalance. Augmentation: with probability
        `CROP_DROPOUT`, a crop's descriptor is replaced by the air-crop descriptor (the scan
        "loses" a region). AdamW with separate learning rates for the head and the adapters, cosine
        schedule with linear warm-up, mixed precision, gradient clipping, early stopping on
        validation macro F1 with the best weights kept.
        """),
        code("""
        def grid_batches(indices, batch_size, rng=None):
            buckets = {}
            for i in indices: buckets.setdefault(GRID[i], []).append(int(i))
            batches = []
            for idx in buckets.values():
                if rng is not None: rng.shuffle(idx)
                batches += [idx[k:k + batch_size] for k in range(0, len(idx), batch_size)]
            if rng is not None: rng.shuffle(batches)
            return batches

        def collate(idx, crop_dropout=0.0, rng=None):
            desc = np.stack([DESC[i] for i in idx]).astype(np.float32)
            if crop_dropout > 0 and rng is not None: desc[rng.random(desc.shape[:2]) < crop_dropout] = AIR
            return torch.as_tensor(desc, device=DEVICE), GRID[idx[0]], torch.as_tensor(Y[idx], dtype=torch.float32, device=DEVICE)

        @torch.no_grad()
        def predict(clf, indices, batch_size=64):
            clf.eval(); out = np.zeros(len(indices), np.float32); pos = {int(i): k for k, i in enumerate(indices)}
            for idx in grid_batches(indices, batch_size):
                desc, grid, _ = collate(idx)
                with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"): logits = clf(desc, grid)
                for i, p in zip(idx, torch.sigmoid(logits.float()).cpu().numpy()): out[pos[i]] = p
            return out
        """),
        code("""
        def train_model(clf, train_idx, val_idx, seed=SEED, log=None):
            rng = np.random.default_rng(seed); torch.manual_seed(seed)
            y_tr = Y[train_idx]; pos_weight = torch.tensor([(y_tr == 0).sum() / max(1, (y_tr == 1).sum())], device=DEVICE)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            head = [p for p in clf.head.parameters()]; adapt = [p for n, p in clf.named_parameters() if p.requires_grad and not n.startswith("head.")]
            opt = torch.optim.AdamW([{"params": head, "lr": LR_HEAD}] + ([{"params": adapt, "lr": LR_ADAPT}] if adapt else []), weight_decay=WEIGHT_DECAY)
            steps = len(grid_batches(train_idx, BATCH_SIZE)); warm, total = int(WARMUP_EPOCHS * steps), EPOCHS * steps
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / max(1, warm) if s < warm else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm)))))
            scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda"); rows, best, best_state, best_prob, since = [], -np.inf, None, None, 0
            for epoch in range(EPOCHS):
                clf.train(); losses = []
                for idx in grid_batches(train_idx, BATCH_SIZE, rng):
                    desc, grid, y = collate(idx, CROP_DROPOUT, rng); opt.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"): logits = clf(desc, grid)
                    loss = loss_fn(logits.float(), y); scaler.scale(loss).backward(); scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(head + adapt, 1.0); scaler.step(opt); scaler.update(); sched.step(); losses.append(loss.item())
                prob = predict(clf, val_idx); y_val = Y[val_idx]
                val_loss = loss_fn(torch.logit(torch.as_tensor(prob).clamp(1e-6, 1 - 1e-6)).to(DEVICE), torch.as_tensor(y_val, dtype=torch.float32, device=DEVICE)).item()
                m = metrics(y_val, prob); row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss, "val_acc": m["accuracy"], "val_macro_f1": m["macro_f1"], "val_auroc": m["auroc"]}; rows.append(row)
                if log: log(f"epoch {epoch:3d} train {row['train_loss']:.4f} val {val_loss:.4f} acc {m['accuracy']:.3f} macroF1 {m['macro_f1']:.3f} AUROC {m['auroc']:.3f}")
                if row["val_macro_f1"] > best + 1e-6: best, best_state, best_prob, since = row["val_macro_f1"], clf.trainable_state(), prob, 0
                else:
                    since += 1
                    if since >= PATIENCE: break
            clf.load_state_dict(best_state, strict=False); clf.eval(); hist = pd.DataFrame(rows)
            return {"history": hist, "best_epoch": int(hist.val_macro_f1.idxmax()), "val_prob": best_prob, "state": best_state}
        """),
        md("""
        ## Cross-validated training

        A fresh classifier per fold. Out-of-fold probabilities are collected for the analysis below,
        and the fold models are kept for the test-time ensemble.
        """),
        code("""
        oof = np.full(len(uids), np.nan, np.float32); per_fold, histories, states = [], [], []
        for f in range(N_FOLDS):
            tr, va = dev_idx[folds != f], dev_idx[folds == f]; clf = build_classifier(); t0 = time.time()
            res = train_model(clf, tr, va); oof[va] = res["val_prob"]; m = metrics(Y[va], res["val_prob"])
            per_fold.append({"fold": f, "best_epoch": res["best_epoch"], "epochs_run": len(res["history"]), "val_loss": res["history"].val_loss.min(), **{k: m[k] for k in ["accuracy", "macro_f1", "f1_pe", "recall_pe", "auroc", "auprc"]}, "minutes": round((time.time() - t0) / 60, 1)})
            print(f"fold {f}: best epoch {res['best_epoch']} | acc {m['accuracy']:.3f} macro F1 {m['macro_f1']:.3f} AUROC {m['auroc']:.3f}")
            histories.append(res["history"].assign(fold=f)); states.append(res["state"]); del clf; torch.cuda.empty_cache() if DEVICE == "cuda" else None
        per_fold = pd.DataFrame(per_fold).set_index("fold"); histories = pd.concat(histories, ignore_index=True)
        pd.DataFrame({"mean": per_fold.mean(), "std": per_fold.std()}).round(4)
        """),
        plot_note(
            "Training curves",
            "Left: training (dashed) and validation (solid) loss per epoch. Middle: validation accuracy. Right: "
            "validation macro F1. One colour per fold; curves end where early stopping triggered.",
            "Shows whether the adapter under- or over-fits and whether the schedule length is right.",
            "Validation loss flattening or rising while training loss keeps falling = overfitting (lower the "
            "learning rate, raise crop dropout or weight decay). All folds behaving alike = a stable recipe.",
        ),
        code("""
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        for f, h in histories.groupby("fold"):
            c = PALETTE[f % 10]; axes[0].plot(h.epoch, h.train_loss, "--", color=c, lw=1); axes[0].plot(h.epoch, h.val_loss, color=c, lw=1.5, label=f"fold {f}")
            axes[1].plot(h.epoch, h.val_acc, color=c, lw=1.5); axes[2].plot(h.epoch, h.val_macro_f1, color=c, lw=1.5)
        for ax, t in zip(axes, ["loss (dashed train, solid val)", "validation accuracy", "validation macro F1"]): ax.set_title(t); ax.set_xlabel("epoch")
        axes[0].legend(fontsize=7); fig.savefig(FIG / "training_curves.png", dpi=130, bbox_inches="tight")
        """),
        md("""
        ## Out-of-fold results on the dev subset

        Every dev study was predicted by the fold model that never saw it. The operating threshold
        maximises macro F1 on these out-of-fold probabilities.
        """),
        code("""
        y_dev, p_dev = Y[dev_idx], oof[dev_idx]; THR = best_threshold(y_dev, p_dev); oof_metrics = metrics(y_dev, p_dev, THR)
        print(f"threshold {THR:.2f} |", {k: round(v, 3) for k, v in oof_metrics.items()})
        """),
        plot_note(
            "Confusion matrix (out-of-fold, dev)",
            "Left: counts, rows are the true class, columns the predicted class. Right: row-normalized, so the "
            "diagonal is each class's recall.",
            "The two mistakes have different costs: a missed PE (bottom-left) is dangerous, a false alarm (top-right) "
            "costs a radiologist's time.",
            "How much of the PE row lands in the PE column while the healthy row stays on its diagonal.",
        ),
        code("""
        fig = confusion_fig(y_dev, p_dev, THR, f"dev, out-of-fold, threshold {THR:.2f}"); fig.savefig(FIG / "confusion_dev_oof.png", dpi=130, bbox_inches="tight")
        """),
        plot_note(
            "Score distribution per class (out-of-fold)",
            "Histograms of the predicted PE probability: negatives in blue, PE in red; dashed line = threshold.",
            "The cleanest picture of separability; the other numbers summarise these two humps.",
            "Two humps far apart = easy; heavy overlap = hard. PE scans piling up just below the threshold are the near misses.",
        ),
        code("""
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.hist(p_dev[y_dev == 0], bins=30, range=(0, 1), alpha=0.6, color=PALETTE[0], label="negative"); ax.hist(p_dev[y_dev == 1], bins=30, range=(0, 1), alpha=0.6, color=PALETTE[3], label="PE")
        ax.axvline(THR, color="k", ls="--", lw=1); ax.set_xlabel("predicted probability of PE"); ax.set_ylabel("dev studies"); ax.legend(); fig.savefig(FIG / "score_distribution_oof.png", dpi=130, bbox_inches="tight")
        """),
        plot_note(
            "Recall by PE sub-type (out-of-fold)",
            "Among PE studies: the fraction caught, split by sub-type and by number of positive slices (a proxy for clot "
            "size); the count of studies in each group is written on the bar.",
            "Shows *which* PEs the model catches. Expected: big, central, heart-straining clots are easy, small "
            "peripheral ones hard.",
            "A gradient from 51+ slices down to 1-10 slices; groups with few studies have noisy bars.",
        ),
        code("""
        pe = todo.iloc[dev_idx].assign(pred=(p_dev >= THR).astype(int)); pe = pe[pe.y == 1]
        groups = {"central": pe.central_pe == 1, "not central": pe.central_pe == 0, "RV/LV >= 1": pe.rv_lv_ratio_gte_1 == 1, "RV/LV < 1": pe.rv_lv_ratio_lt_1 == 1,
                  "chronic": (pe.chronic_pe == 1) | (pe.acute_and_chronic_pe == 1), "1-10 slices": pe.n_pos_slices <= 10, "11-50 slices": (pe.n_pos_slices > 10) & (pe.n_pos_slices <= 50), "51+ slices": pe.n_pos_slices > 50}
        rb = pd.DataFrame({"n": {k: int(m.sum()) for k, m in groups.items()}, "recall": {k: pe.loc[m, "pred"].mean() for k, m in groups.items()}})
        fig, ax = plt.subplots(figsize=(10, 4)); ax.bar(rb.index, rb.recall.fillna(0), color=[PALETTE[i % 10] for i in range(len(rb))]); ax.set_ylim(0, 1); ax.set_title("recall among PE studies, by sub-type")
        for i, (n, r) in enumerate(zip(rb.n, rb.recall)): ax.text(i, r if np.isfinite(r) else 0, f"n={n}\\n{r:.2f}", ha="center", va="bottom", fontsize=8)
        ax.tick_params(axis="x", rotation=30); fig.savefig(FIG / "recall_by_subtype.png", dpi=130, bbox_inches="tight"); rb.round(3)
        """),
        md("""
        ## Test sample (run once)

        The fold models vote (average probability) on the frozen test sample at the threshold chosen
        above; bootstrap 95 % confidence intervals say how much to trust each number.
        """),
        code("""
        p_test = np.zeros(len(test_idx), np.float32)
        for state in states:
            clf = build_classifier(); clf.load_state_dict(state, strict=False); p_test += predict(clf, test_idx) / len(states); del clf
        y_test = Y[test_idx]; test_metrics = metrics(y_test, p_test, THR)
        bootstrap(y_test, p_test, THR)
        """),
        code("""
        fig = confusion_fig(y_test, p_test, THR, f"test sample, {len(states)}-model ensemble, threshold {THR:.2f}"); fig.savefig(FIG / "confusion_test.png", dpi=130, bbox_inches="tight")
        pd.DataFrame({"frozen + logistic regression (fold 0)": frozen, "fine-tuned, out-of-fold (dev)": oof_metrics, "fine-tuned, ensemble (test)": test_metrics}).T.round(3)
        """),
        md("## Save predictions and the fold models"),
        code("""
        pred = pd.concat([pd.DataFrame({"uid": np.array(uids)[dev_idx], "y": y_dev, "prob": p_dev, "pred": (p_dev >= THR).astype(int), "split": "dev_oof", "fold": folds}),
                          pd.DataFrame({"uid": np.array(uids)[test_idx], "y": y_test, "prob": p_test, "pred": (p_test >= THR).astype(int), "split": "test", "fold": -1})], ignore_index=True)
        pred.to_csv(OUT / "predictions.csv", index=False); per_fold.to_csv(OUT / "cv_per_fold.csv")
        torch.save({"states": states, "mode": FINETUNE_MODE, "lora_rank": LORA_RANK, "threshold": THR}, OUT / "finetune_folds.pt")
        print("saved", len(pred), "predictions and", len(states), "fold models to", OUT)
        """),
        learned_cell([
            "(fill in after running) accuracy and macro F1 on the test sample with their confidence intervals, next to the frozen reference.",
            "(fill in after running) did fine-tuning beat the frozen embedding, and by how much?",
            "(fill in after running) where did early stopping trigger; did any fold overfit?",
            "(fill in after running) which sub-types are still missed; is clot size the main driver?",
            "(fill in after running) if results are flat: try last_blocks mode, more dev studies, or a lower threshold for higher recall.",
        ]),
    ]
