"""04_linear_probe: how much PE information is linearly readable from the frozen embedding."""

from scripts.nb.tools import code, config_cell, learned_cell, md, plot_note, setup_cell

TITLE = "04_linear_probe"


def cells():
    return [
        md("""
        # 04 - Linear probe

        **Goal.** Measure how much PE information is *linearly* readable from the frozen scan
        embedding: one `nn.Linear(d, 1)` on standardized embeddings, nothing else.

        Training details: binary cross-entropy with a positive-class weight equal to the
        negative/positive ratio (handles the imbalance), AdamW with weight decay, early stopping on
        validation AUPRC. Feature standardization statistics are fitted on the training fold only.
        A balanced logistic regression from scikit-learn is fitted as a sanity baseline; the two
        should roughly agree.

        Glossary: **AUROC** = area under the ROC curve (ranking quality, insensitive to prevalence);
        **AUPRC** = area under the precision-recall curve (ranking quality *for the positive class*,
        honest under imbalance: a random model scores the prevalence, not 0.5); **macro F1** = the
        F1 score averaged over both classes, our headline metric; **operating threshold** = the
        probability above which we call a scan positive, chosen on validation data, never on test.

        The test sample is touched exactly once, in the final evaluation cell.
        """),
        setup_cell(),
        md("## Configuration"),
        config_cell(TITLE),
        md("## Load embeddings, labels, folds"),
        code("""
        import numpy as np
        import pandas as pd
        from tqdm.auto import tqdm
        from pe_ct import storage, probe

        records = storage.load_all_records(cfg.embeddings_dir, progress=tqdm)
        split_table = pd.read_parquet(cfg.splits_path).set_index("study_uid")
        studies = pd.read_parquet(cfg.study_labels_path).set_index("study_uid")
        uids = [u for u in records if u in split_table.index]
        X = np.stack([records[u]["cls"] for u in uids]).astype(np.float32)
        y = np.array([records[u]["y"] for u in uids])
        meta = split_table.loc[uids].join(studies.drop(columns=["y"]), how="left")
        print(X.shape, "embeddings |", f"{y.mean():.1%} PE overall")
        """),
        code("""
        is_dev = meta.in_dev.to_numpy()
        is_test = meta.in_test_sample.to_numpy()
        folds = meta.fold.to_numpy()
        assert not (is_dev & is_test).any(), "a study is in both dev and test"
        X_dev, y_dev, f_dev = X[is_dev], y[is_dev], folds[is_dev]
        X_test, y_test = X[is_test], y[is_test]
        print(f"dev {len(y_dev)} ({y_dev.mean():.1%} PE) | test sample {len(y_test)} ({y_test.mean():.1%} PE, not looked at until the end)")
        print("per fold:", {f: f"{(f_dev == f).sum()} / {y_dev[f_dev == f].mean():.1%}" for f in range(cfg.n_folds)})
        """),
        md("""
        ## Train on fold 0

        Validation = fold 0, training = the other folds.
        """),
        code("""
        train_kw = dict(epochs=cfg.probe_epochs, lr=cfg.probe_lr, weight_decay=cfg.probe_weight_decay, patience=cfg.probe_patience, seed=cfg.seed)
        VAL = 0
        tr, va = f_dev != VAL, f_dev == VAL
        model0, hist0 = probe.train_probe(X_dev[tr], y_dev[tr], X_dev[va], y_dev[va], **train_kw)
        p_tr0, p_va0 = probe.predict_proba(model0, X_dev[tr]), probe.predict_proba(model0, X_dev[va])
        print("best epoch", hist0.attrs["best_epoch"], "| val AUROC", round(probe.compute_metrics(y_dev[va], p_va0)["auroc"], 3))
        print("sklearn baseline val AUROC", round(probe.compute_metrics(y_dev[va], probe.sklearn_baseline(X_dev[tr], y_dev[tr], X_dev[va]))["auroc"], 3))
        """),
        plot_note(
            "Training vs validation curves",
            "Left: loss per epoch on the training folds (blue) and the validation fold (orange). Right: AUPRC per "
            "epoch for both. The dashed line is the epoch whose weights were kept.",
            "Shows whether the probe under- or over-fits. Validation loss rising while training loss keeps falling "
            "means the probe is memorizing training studies; both still falling means more epochs would help.",
            "Curves that flatten together. A growing gap between train and val is the classic overfitting signature; "
            "with only d+1 parameters on hundreds of studies it should be mild.",
        ),
        code("""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        hist0.plot(x="epoch", y=["train_loss", "val_loss"], ax=axes[0], title="loss")
        hist0.plot(x="epoch", y=["train_auprc", "val_auprc"], ax=axes[1], title="AUPRC")
        for ax in axes: ax.axvline(hist0.attrs["best_epoch"], color="k", ls="--", lw=1)
        viz.save_fig(fig, FIG, NOTEBOOK, "train_val_curves");
        """),
        plot_note(
            "Score distribution per class",
            "Histograms of the predicted PE probability on the validation fold: negative studies in blue, PE studies in red.",
            "The cleanest picture of separability. Everything else (ROC, PR, thresholds) is a summary of these two humps.",
            "Two humps far apart = easy; heavily overlapping = hard. Note where the PE hump sits: a model that is "
            "unsure puts PE scans in the middle rather than at 1.",
        ),
        code("""
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.hist(p_va0[y_dev[va] == 0], bins=30, range=(0, 1), alpha=0.6, color=viz.COLOR_NEG, label="negative")
        ax.hist(p_va0[y_dev[va] == 1], bins=30, range=(0, 1), alpha=0.6, color=viz.COLOR_PE, label="PE")
        ax.set_xlabel("predicted probability of PE"); ax.set_ylabel("studies (val fold)"); ax.legend()
        viz.save_fig(fig, FIG, NOTEBOOK, "score_distribution");
        """),
        plot_note(
            "ROC and precision-recall curves",
            "Left: ROC (true-positive rate vs false-positive rate). Right: precision vs recall. The dashed lines are "
            "the chance level: the diagonal for ROC, a horizontal line at the PE prevalence for PR.",
            "ROC looks flattering under imbalance because the many negatives make the false-positive *rate* small even "
            "when the false positives outnumber the true ones. PR shows directly how many of the scans we flag are "
            "really PE, which is what a radiologist triaging a worklist cares about.",
            "How far the PR curve sits above the prevalence line, and whether high recall is reachable without "
            "precision collapsing.",
        ),
        code("""
        cv = probe.curves(y_dev[va], p_va0)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        axes[0].plot(cv["fpr"], cv["tpr"], color=viz.PALETTE[0]); axes[0].plot([0, 1], [0, 1], "k--", lw=1)
        axes[0].set_xlabel("false positive rate"); axes[0].set_ylabel("true positive rate"); axes[0].set_title(f"ROC  (AUROC {probe.compute_metrics(y_dev[va], p_va0)['auroc']:.3f})")
        axes[1].plot(cv["recall"], cv["precision"], color=viz.PALETTE[3]); axes[1].axhline(y_dev[va].mean(), color="k", ls="--", lw=1, label="random = prevalence")
        axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision"); axes[1].set_title(f"PR  (AUPRC {probe.compute_metrics(y_dev[va], p_va0)['auprc']:.3f})"); axes[1].legend()
        viz.save_fig(fig, FIG, NOTEBOOK, "roc_pr_curves");
        """),
        plot_note(
            "Threshold sweep",
            "Precision, recall, F1 of the PE class and macro F1 as a function of the decision threshold on the "
            "validation fold. The vertical line marks the threshold with the best PE F1.",
            "A probability of 0.5 is an arbitrary cut-off, especially with a class-weighted loss. This picks a sensible "
            "operating point on validation data and is the threshold used for every confusion matrix from here on.",
            "Where recall and precision cross, and how flat the F1 curve is around its maximum (flat = the choice is robust).",
        ),
        code("""
        sweep = probe.threshold_sweep(y_dev[va], p_va0)
        THR = probe.best_threshold(y_dev[va], p_va0, "f1")
        fig, ax = plt.subplots(figsize=(8, 4))
        sweep.plot(x="threshold", y=["precision", "recall", "f1", "macro_f1"], ax=ax)
        ax.axvline(THR, color="k", ls="--", lw=1, label=f"best F1 threshold = {THR:.2f}"); ax.legend(); ax.set_ylim(0, 1)
        viz.save_fig(fig, FIG, NOTEBOOK, "threshold_sweep");
        """),
        plot_note(
            "Confusion matrix at the chosen threshold",
            "Left: counts (rows = true class, columns = predicted). Right: the same normalized by row, i.e. recall "
            "per class on the diagonal.",
            "Puts numbers on the two kinds of mistakes: a missed PE (bottom-left) is dangerous for the patient; a "
            "false alarm (top-right) costs a radiologist's time.",
            "The bottom-right cell (caught PEs) vs bottom-left (missed PEs) is the recall trade-off chosen above.",
        ),
        code("""
        cm = probe.confusion(y_dev[va], p_va0, THR)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
        for ax, mat, title, fmt in [(axes[0], cm, "counts", "d"), (axes[1], cm / cm.sum(1, keepdims=True), "row-normalized", ".2f")]:
            ax.imshow(mat, cmap="Blues"); ax.set_title(title)
            ax.set_xticks([0, 1]); ax.set_xticklabels(["pred no PE", "pred PE"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["true no PE", "true PE"])
            for i in range(2):
                for j in range(2): ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center", color="w" if mat[i, j] > mat.max() / 2 else "k")
        viz.save_fig(fig, FIG, NOTEBOOK, "confusion_matrix");
        """),
        md("""
        ## Per-class precision, recall and F1

        Train and validation side by side at the chosen threshold. Macro F1 is the headline number.
        """),
        code("""
        print("val:", {k: round(v, 3) for k, v in probe.compute_metrics(y_dev[va], p_va0, THR).items() if k in ["macro_f1", "accuracy", "auroc", "auprc", "recall_pe", "precision_pe"]})
        probe.report_table(y_dev[tr], p_tr0, y_dev[va], p_va0, THR)
        """),
        md("""
        ## 5-fold cross-validation

        One split can be lucky. Every fold takes a turn as the validation set; the per-fold
        threshold is chosen on that fold. The out-of-fold probabilities are kept for the sub-type
        analysis and for the error notebook.
        """),
        code("""
        per_fold, oof = probe.cross_validate(X_dev, y_dev, f_dev, n_folds=cfg.n_folds, **train_kw)
        probe.cv_summary(per_fold)
        """),
        plot_note(
            "Learning curve",
            "Validation metrics (fold 0) when the probe is trained on growing stratified subsets of the training folds.",
            "Tells us whether downloading and embedding more studies would help: a curve still rising at the right "
            "edge says yes; a flat one says the frozen embedding, not the amount of data, is the limit.",
            "The slope at the largest training size. Compare AUROC (ranking) and macro F1 (decisions): they may saturate differently.",
        ),
        code("""
        sizes = [s for s in [250, 500, 1000, 1600] if s <= (f_dev != VAL).sum()] or [max(10, (f_dev != VAL).sum() // 2), (f_dev != VAL).sum()]
        lc = probe.learning_curve(X_dev, y_dev, f_dev, sizes, val_fold=VAL, **train_kw)
        fig, ax = plt.subplots(figsize=(7, 4))
        lc.plot(x="n_train", y=["auroc", "auprc", "macro_f1"], marker="o", ax=ax); ax.set_ylim(0, 1); ax.set_xlabel("training studies")
        viz.save_fig(fig, FIG, NOTEBOOK, "learning_curve"); lc
        """),
        plot_note(
            "Recall by PE sub-type",
            "Bars: the fraction of PE studies that were caught (out-of-fold predictions at each fold's threshold), split "
            "by sub-type and by the number of positive slices; the count of studies in each group is written on the bar.",
            "Shows *which* PEs the model catches. The expectation for a whole-scan embedding is that big, central, "
            "heart-straining clots are easy and small peripheral ones are hard; this is the quantitative version of that.",
            "A steep gradient from 51+ slices down to 1-10 slices, and central above non-central. Groups with few "
            "studies have noisy bars.",
        ),
        code("""
        dev_df = meta[is_dev].copy()
        dev_df["prob"] = oof
        dev_df["pred"] = 0
        for f in range(cfg.n_folds):
            m = dev_df.fold == f
            dev_df.loc[m, "pred"] = (dev_df.loc[m, "prob"] >= per_fold.loc[f, "threshold"]).astype(int)
        rb = probe.recall_by_subgroup(dev_df)
        fig, ax = plt.subplots(figsize=(10, 4))
        viz.bar_counts(ax, rb["recall"].round(2), title="recall among PE studies, by sub-type", annotate=False); ax.set_ylim(0, 1)
        for i, (n, r) in enumerate(zip(rb["n"], rb["recall"])):
            ax.text(i, r if np.isfinite(r) else 0, f"n={n}\\n{r:.2f}" if np.isfinite(r) else f"n={n}", ha="center", va="bottom", fontsize=8)
        ax.tick_params(axis="x", rotation=30)
        viz.save_fig(fig, FIG, NOTEBOOK, "recall_by_subtype"); rb
        """),
        md("""
        ## Final test evaluation (run once)

        Retrain on the whole dev subset for the median best epoch found in cross-validation, apply
        the validation-chosen threshold, and report every metric on the frozen test sample with
        bootstrap 95 % confidence intervals.
        """),
        code("""
        n_epochs = int(per_fold.best_epoch.median()) + 1
        THR_FINAL = float(np.median([probe.best_threshold(y_dev[f_dev == f], oof[f_dev == f]) for f in range(cfg.n_folds)]))
        final_kw = dict(train_kw, epochs=n_epochs, patience=n_epochs + 1)
        model_final, _ = probe.train_probe(X_dev, y_dev, X_dev, y_dev, **final_kw)
        p_test = probe.predict_proba(model_final, X_test)
        print(f"trained {n_epochs} epochs on {len(y_dev)} dev studies | threshold {THR_FINAL:.2f}")
        probe.bootstrap_metrics(y_test, p_test, THR_FINAL, n_boot=1000, seed=cfg.seed)
        """),
        code("""
        print("test sample confusion matrix (rows true, cols pred):"); print(probe.confusion(y_test, p_test, THR_FINAL))
        probe.report_table(y_dev, probe.predict_proba(model_final, X_dev), y_test, p_test, THR_FINAL)
        """),
        md("## Save predictions and the final probe for the error analysis"),
        code("""
        import torch

        pred = pd.concat([
            pd.DataFrame({"uid": dev_df.index, "y": dev_df.y, "prob": dev_df.prob, "pred": dev_df.pred, "split": "dev_oof", "fold": dev_df.fold}),
            pd.DataFrame({"uid": meta.index[is_test], "y": y_test, "prob": p_test, "pred": (p_test >= THR_FINAL).astype(int), "split": "test", "fold": -1}),
        ], ignore_index=True)
        storage.atomic_write_parquet(pred, cfg.results_dir / "predictions.parquet")
        tmp = cfg.results_dir / "probe_final.pt.tmp"
        torch.save({"state_dict": model_final.state_dict(), "threshold": THR_FINAL, "n_epochs": n_epochs, "d": X.shape[1]}, tmp)
        tmp.replace(cfg.results_dir / "probe_final.pt")
        storage.atomic_write_parquet(per_fold.reset_index(), cfg.results_dir / "cv_per_fold.parquet")
        print("saved", len(pred), "predictions")
        """),
        learned_cell([
            "(fill in after running) headline numbers: macro F1, AUROC and AUPRC on the test sample with their confidence intervals.",
            "(fill in after running) does the probe overfit, and does the torch probe agree with the sklearn baseline?",
            "(fill in after running) is the learning curve still rising (more data would help) or flat (the frozen embedding is the limit)?",
            "(fill in after running) which sub-types are caught and which are missed; how strong is the dependence on clot size?",
        ]),
    ]
