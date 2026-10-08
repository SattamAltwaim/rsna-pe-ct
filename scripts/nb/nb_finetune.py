"""02_finetune: frozen backbone once, then fine-tune the combiner (+ MLP head) with CV."""

from scripts.nb.tools import code, config_cell, learned_cell, md, plot_note, setup_cell

TITLE = "02_finetune"


def cells():
    return [
        md("""
        # 02 - Fine-tune SPECTRE for PE detection

        **Goal.** Train a PE / no-PE classifier on a few thousand studies by adapting a pretrained
        3D CT foundation model, and report honest numbers: loss, accuracy, macro F1, confusion
        matrix, with early stopping and a cosine learning-rate schedule, on group-stratified folds.

        **What trains and why.** SPECTRE has two parts: a ViT-L *backbone* that turns each
        128 x 128 x 64 crop of the scan into a descriptor, and a small *feature combiner* transformer
        that turns the ~50 crop descriptors of a scan into one scan embedding. The backbone is
        already good and the subset is small, so it stays frozen and runs **once** per study; its
        crop descriptors are cached. Training then touches only the combiner (the last layers of
        the model) and a new MLP head. Two ways to adapt the combiner are available: **LoRA**
        (low-rank adapters on the attention projections, a few hundred thousand parameters, the
        default) or unfreezing its **last blocks** (tens of millions of parameters, more capacity,
        more overfitting risk). A training epoch is seconds, so 5-fold cross-validation is cheap.

        Glossary: **LoRA** = train a small low-rank update `B A x` next to each frozen weight matrix;
        **cosine schedule** = the learning rate warms up, then decays following a cosine to zero;
        **early stopping** = stop when the validation metric has not improved for a few epochs and
        keep the best weights; **StratifiedGroupKFold** = folds with the same class mix, where every
        group (a study = a patient) lands in exactly one fold; **macro F1** = F1 averaged over both
        classes, our headline metric; **AUROC** = ranking quality independent of the threshold.
        """),
        setup_cell(),
        md("## Configuration"),
        config_cell(TITLE, '''FINETUNE_MODE = "lora"         # "lora" | "last_blocks" | "head_only"
    LORA_RANK, N_LAST_BLOCKS = 8, 2
    EPOCHS, BATCH_SIZE, PATIENCE, WARMUP_EPOCHS = 40, 32, 6, 2
    LR_HEAD, LR_ADAPT, WEIGHT_DECAY, CROP_DROPOUT = 1e-3, 2e-4, 1e-2, 0.1
    # subset sizes come from cfg (n_dev / n_test_sample); lower them here for a faster first run'''),
        md("""
        ## Backbone and data

        The model is loaded once in half precision. Labels and splits are built from the label
        file if not found (seeded, identical to the EDA notebook).
        """),
        code("""
        import subprocess, sys
        try:
            import spectre
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "spectre-fm"], check=True)
            import spectre
        print("spectre-fm", spectre.__version__)
        """),
        code("""
        import torch
        from pe_ct import colab, embed

        report = colab.session_report(cfg)
        MODEL_NAME = os.environ.get("PE_CT_MODEL", "spectre-large")   # a random "spectre-small" is used only for local smoke tests
        model = embed.load_model(MODEL_NAME, pretrained=MODEL_NAME == "spectre-large")
        embed.model_info(model)
        """),
        code("""
        import numpy as np
        import pandas as pd
        from tqdm.auto import tqdm
        from pe_ct import labels, pipeline, storage

        studies, split_table = pipeline.ensure_labels_and_splits(cfg)
        slice_index = labels.SliceLabelIndex(pipeline.load_train(cfg))
        locator = pipeline.make_locator(cfg)
        dev_uids = split_table.loc[split_table.in_dev, "study_uid"].tolist()
        test_uids = split_table.loc[split_table.in_test_sample, "study_uid"].tolist()
        print(len(dev_uids), "dev studies |", len(test_uids), "test studies (touched once, at the end)")
        """),
        md("""
        ## Stage A: run the frozen backbone once

        Every dev and test study is read from disk (its files in parallel), oriented, tiled into
        crops and pushed through the frozen backbone; the per-crop descriptors (what the combiner
        consumes, a few hundred KB per study) are kept in small npz files so a dropped session
        resumes where it stopped. Nothing else is written. This is the only slow part of the
        notebook, so the next cell times a few studies first and prints the expected total.
        """),
        code("""
        import time

        by_uid = studies.set_index("study_uid")
        timings = []
        for uid in dev_uids[:3]:
            t0 = time.time()
            image, meta, t = pipeline.process_study(cfg, locator, slice_index, uid, by_uid.loc[uid], cfg.work / "dicom")
            rec = embed.embed_image(model, image, cfg.resample_spacing)
            timings.append({"read+decode_s": t["t_download_s"] + t["t_decode_s"], "gpu_s": rec["t_gpu_s"], "total_s": time.time() - t0, "crops": rec["n_crops"]})
        timings = pd.DataFrame(timings); per_study = timings.total_s.mean()
        print(timings.round(2).to_string())
        print(f"about {per_study:.1f} s per study -> roughly {per_study * (len(dev_uids) + len(test_uids)) / 60:.0f} min for {len(dev_uids) + len(test_uids)} studies (reads overlap the GPU in the real loop, so usually less)")
        """),
        code("""
        air_path = cfg.embeddings_dir / "air_descriptor.npy"
        if not air_path.exists():
            storage.atomic_write_npy(air_path, embed.air_descriptor(model).astype(np.float32))
        failure_log = storage.FailureLog(cfg.failures_csv(NOTEBOOK))
        stats = pipeline.run_embedding_extraction(cfg, model, studies, dev_uids + test_uids, locator, slice_index, failure_log, prefetch=3, progress=tqdm)
        """),
        code("""
        done = storage.done_uids(cfg.embeddings_dir, "npz")
        print(f"this run: {stats['done']} done, {stats['failed']} failed | stored in total: {len(done)} of {len(dev_uids) + len(test_uids)}")
        if stats["seconds"]:
            print(f"seconds per study: mean {np.mean(stats['seconds']):.1f} | GPU {np.mean(stats['t_gpu']):.1f} | CPU read+decode {np.mean(stats['t_cpu']):.1f}")
        failure_log.df.tail()
        """),
        md("""
        ## Load the cached descriptors

        The backbone is no longer needed and is dropped from GPU memory; only the combiner stays.
        """),
        code("""
        from pe_ct import finetune

        records = storage.load_all_records(cfg.embeddings_dir, progress=tqdm)
        air = np.load(air_path)
        uids = [u for u in dev_uids + test_uids if u in records]
        dataset = finetune.DescriptorSet(records, uids)
        pos = {u: i for i, u in enumerate(uids)}
        dev_idx = np.array([pos[u] for u in dev_uids if u in pos]); test_idx = np.array([pos[u] for u in test_uids if u in pos])
        del model.backbone; torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print(len(dataset), "studies |", f"{dataset.y[dev_idx].mean():.1%} PE in dev |", "descriptor dim", dataset.desc[0].shape[1])
        """),
        plot_note(
            "Crop grids per scan",
            "How many scans have each crop grid (R x A x S boxes). A scan's crops are what the combiner sees.",
            "The combiner can only batch scans with the same grid, and scans with very few crops along the "
            "head-to-feet axis have had their top and bottom slices cut off by the tiling.",
            "One or two dominant grids; rare grids form tiny batches, which is fine.",
        ),
        code("""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 3.5))
        viz.bar_counts(ax, dataset.grid_counts(), title="crop grid (R x A x S) per scan")
        viz.save_fig(fig, FIG, NOTEBOOK, "crop_grids");
        """),
        md("""
        ## Group-stratified folds

        Five folds over the dev subset, stratified on the label, with every study in exactly one
        fold. Each fold takes a turn as the validation set; the test sample is not part of this.
        """),
        code("""
        folds = finetune.group_stratified_folds(dataset.y[dev_idx], np.array(uids)[dev_idx], n_splits=cfg.n_folds, seed=cfg.seed)
        pd.DataFrame({"n": pd.Series(folds).value_counts().sort_index(), "pe_frac": pd.Series(dataset.y[dev_idx]).groupby(folds).mean().round(3)})
        """),
        md("""
        ## Reference: frozen embedding + linear probe

        One number to beat: a linear classifier on the frozen scan embedding (fold 0 as validation).
        If fine-tuning cannot beat this, the extra capacity is not being used.
        """),
        code("""
        from pe_ct import probe

        tr, va = dev_idx[folds != 0], dev_idx[folds == 0]
        lin, _ = probe.train_probe(dataset.cls[tr], dataset.y[tr], dataset.cls[va], dataset.y[va], epochs=cfg.probe_epochs, lr=cfg.probe_lr, weight_decay=cfg.probe_weight_decay, patience=cfg.probe_patience, seed=cfg.seed)
        p_lin = probe.predict_proba(lin, dataset.cls[va])
        frozen = probe.compute_metrics(dataset.y[va], p_lin, 0.5)
        print({k: round(frozen[k], 3) for k in ["accuracy", "macro_f1", "f1_pe", "auroc", "auprc"]})
        """),
        md("""
        ## What trains

        A fresh copy of the pretrained combiner per fold, adapted as configured, plus a small MLP
        on its CLS token. The table says how many parameters move.
        """),
        code("""
        build_kwargs = dict(mode=FINETUNE_MODE, lora_rank=LORA_RANK, n_last_blocks=N_LAST_BLOCKS)
        train_kwargs = dict(epochs=EPOCHS, batch_size=BATCH_SIZE, lr_head=LR_HEAD, lr_adapt=LR_ADAPT, weight_decay=WEIGHT_DECAY,
                            warmup_epochs=WARMUP_EPOCHS, patience=PATIENCE, crop_dropout=CROP_DROPOUT, air=air, seed=cfg.seed)
        _, info = finetune.build_classifier(model, **build_kwargs)
        pd.Series(info)
        """),
        md("""
        ## Cross-validated training

        Per fold: AdamW, cosine schedule with warm-up, mixed precision, early stopping on validation
        macro F1 (best weights kept). The out-of-fold probabilities are collected for the error
        analysis below, and the five fold models are kept for the test-time ensemble.
        """),
        code("""
        cv = finetune.cross_validate(model, dataset, dev_idx, folds, build_kwargs, train_kwargs, progress=tqdm)
        per_fold = cv["per_fold"]
        per_fold
        """),
        code("""
        finetune.cv_summary(per_fold)
        """),
        plot_note(
            "Training curves",
            "Left: training (dashed) and validation (solid) loss per epoch. Middle: validation accuracy per epoch. "
            "Right: validation macro F1 per epoch. One colour per fold; curves end where early stopping triggered.",
            "Shows whether the adapter under- or over-fits and whether the schedule length is right.",
            "Validation loss flattening or rising while training loss keeps falling = overfitting (lower the learning "
            "rate, raise crop dropout or weight decay). All folds behaving alike = a stable recipe.",
        ),
        code("""
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        finetune.plot_histories(cv["histories"], axes)
        viz.save_fig(fig, FIG, NOTEBOOK, "training_curves");
        """),
        md("""
        ## Out-of-fold results on the dev subset

        Every dev study was predicted by the fold model that never saw it. The operating threshold
        is the one maximising macro F1 on these out-of-fold probabilities.
        """),
        code("""
        oof = cv["oof"][dev_idx]; y_dev = dataset.y[dev_idx]
        THR = probe.best_threshold(y_dev, oof, "macro_f1")
        oof_metrics = probe.compute_metrics(y_dev, oof, THR)
        print(f"threshold {THR:.2f} |", {k: round(oof_metrics[k], 3) for k in ["accuracy", "macro_f1", "f1_pe", "recall_pe", "precision_pe", "auroc", "auprc"]})
        """),
        plot_note(
            "Confusion matrix (out-of-fold, dev)",
            "Left: counts, rows are the true class, columns the predicted class. Right: row-normalized, so the diagonal "
            "is the recall of each class.",
            "The two kinds of mistakes have different costs: a missed PE (bottom-left) is dangerous, a false alarm "
            "(top-right) costs a radiologist's time.",
            "How much of the PE row lands in the PE column, and whether the healthy row stays mostly on its diagonal.",
        ),
        code("""
        fig = viz.confusion_matrix_fig(probe.confusion(y_dev, oof, THR), title=f"dev, out-of-fold, threshold {THR:.2f}")
        viz.save_fig(fig, FIG, NOTEBOOK, "confusion_dev_oof");
        """),
        plot_note(
            "Score distribution per class (out-of-fold)",
            "Histograms of the predicted PE probability: negative studies in blue, PE studies in red; the dashed line "
            "is the chosen threshold.",
            "The cleanest picture of separability; everything else is a summary of these two humps.",
            "Two humps far apart = easy; heavy overlap = hard. PE scans piling up just below the threshold are the "
            "near misses worth looking at.",
        ),
        code("""
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.hist(oof[y_dev == 0], bins=30, range=(0, 1), alpha=0.6, color=viz.COLOR_NEG, label="negative")
        ax.hist(oof[y_dev == 1], bins=30, range=(0, 1), alpha=0.6, color=viz.COLOR_PE, label="PE")
        ax.axvline(THR, color="k", ls="--", lw=1); ax.set_xlabel("predicted probability of PE"); ax.set_ylabel("dev studies"); ax.legend()
        viz.save_fig(fig, FIG, NOTEBOOK, "score_distribution_oof");
        """),
        plot_note(
            "Recall by PE sub-type (out-of-fold)",
            "Among PE studies: the fraction caught, split by sub-type and by number of positive slices; the count of "
            "studies in each group is written on the bar.",
            "Shows *which* PEs the model catches. Expected: big, central, heart-straining clots are easy, small "
            "peripheral ones hard.",
            "A gradient from 51+ slices down to 1-10 slices; groups with few studies have noisy bars.",
        ),
        code("""
        dev_df = studies.set_index("study_uid").loc[np.array(uids)[dev_idx]].copy()
        dev_df["pred"] = (oof >= THR).astype(int)
        rb = probe.recall_by_subgroup(dev_df)
        fig, ax = plt.subplots(figsize=(10, 4))
        viz.bar_counts(ax, rb["recall"].round(2), title="recall among PE studies, by sub-type", annotate=False); ax.set_ylim(0, 1)
        for i, (n, r) in enumerate(zip(rb["n"], rb["recall"])):
            ax.text(i, r if np.isfinite(r) else 0, f"n={n}\\n{r:.2f}" if np.isfinite(r) else f"n={n}", ha="center", va="bottom", fontsize=8)
        ax.tick_params(axis="x", rotation=30)
        viz.save_fig(fig, FIG, NOTEBOOK, "recall_by_subtype"); rb
        """),
        md("""
        ## Test sample (run once)

        The five fold models vote (average probability) on the frozen test sample, at the threshold
        chosen above. Bootstrap 95 % confidence intervals say how much to trust each number.
        """),
        code("""
        p_test = finetune.ensemble_predict(model, dataset, test_idx, cv["states"], build_kwargs)
        y_test = dataset.y[test_idx]
        test_metrics = probe.compute_metrics(y_test, p_test, THR)
        probe.bootstrap_metrics(y_test, p_test, THR, n_boot=1000, seed=cfg.seed, keys=("accuracy", "macro_f1", "f1_pe", "recall_pe", "precision_pe", "auroc", "auprc"))
        """),
        code("""
        fig = viz.confusion_matrix_fig(probe.confusion(y_test, p_test, THR), title=f"test sample, 5-model ensemble, threshold {THR:.2f}")
        viz.save_fig(fig, FIG, NOTEBOOK, "confusion_test");
        """),
        md("## Frozen probe vs fine-tuned, side by side"),
        code("""
        rows = {"frozen + linear probe (fold 0 val)": frozen, "fine-tuned, out-of-fold (dev)": oof_metrics, "fine-tuned, ensemble (test)": test_metrics}
        pd.DataFrame(rows).loc[["accuracy", "macro_f1", "f1_pe", "recall_pe", "precision_pe", "auroc", "auprc"]].T.round(3)
        """),
        md("## Save predictions and the fold models"),
        code("""
        pred = pd.concat([
            pd.DataFrame({"uid": np.array(uids)[dev_idx], "y": y_dev, "prob": oof, "pred": (oof >= THR).astype(int), "split": "dev_oof", "fold": folds}),
            pd.DataFrame({"uid": np.array(uids)[test_idx], "y": y_test, "prob": p_test, "pred": (p_test >= THR).astype(int), "split": "test", "fold": -1}),
        ], ignore_index=True)
        storage.atomic_write_parquet(pred, cfg.results_dir / "predictions_finetune.parquet")
        tmp = cfg.results_dir / "finetune_folds.pt.tmp"
        torch.save({"states": cv["states"], "build_kwargs": build_kwargs, "train_kwargs": {k: v for k, v in train_kwargs.items() if k != "air"}, "threshold": THR}, tmp)
        tmp.replace(cfg.results_dir / "finetune_folds.pt")
        storage.atomic_write_parquet(per_fold.reset_index(), cfg.results_dir / "finetune_cv_per_fold.parquet")
        print("saved", len(pred), "predictions and", len(cv["states"]), "fold models")
        """),
        learned_cell([
            "(fill in after running) headline numbers: accuracy and macro F1 on the test sample with their confidence intervals, next to the frozen-probe reference.",
            "(fill in after running) did fine-tuning beat the frozen probe, and by how much?",
            "(fill in after running) where did early stopping trigger, and did any fold overfit?",
            "(fill in after running) which sub-types are still missed; is clot size the main driver?",
            "(fill in after running) if results are flat: try last_blocks mode, more dev studies, or a lower threshold for higher recall.",
        ]),
    ]
