"""05_errors_and_heatmaps: which scans the model gets wrong and where it looks."""

from scripts.nb.tools import code, config_cell, learned_cell, md, plot_note, setup_cell

TITLE = "05_errors_and_heatmaps"


def cells():
    return [
        md("""
        # 05 - Errors and heatmaps

        **Goal.** See which scans the probe gets wrong, and where the model "looks" when it calls PE.

        **Method for the heatmaps: crop-level occlusion.** For a study we already have the descriptor
        of every crop that feeds SPECTRE's feature combiner. Occluding crop *i* means replacing its
        descriptor with that of a crop of pure air, re-running only the small combiner plus the
        trained linear layer, and recording how much the PE logit drops:

        > importance(i) = logit(all crops) - logit(crop *i* replaced by air)

        A large drop means the model relied on that region. The backbone is never re-run, so this
        is cheap. Replacing (rather than removing) keeps the crop grid intact, which the combiner's
        position encoding needs.

        **Resolution warning.** One crop is 128 x 128 x 64 voxels, so a heatmap has a few dozen boxes
        per scan. It says *which region*, not *which pixel*. Since the labels only locate clots along
        the head-to-feet axis, "did it look in the right place" can only be judged along that axis too.

        Glossary: **FN** = false negative (a PE study the model called negative); **FP** = false
        positive (a negative study called PE); **logit** = the raw score before the sigmoid.
        """),
        setup_cell(),
        md("## Configuration"),
        config_cell(TITLE, "N_GALLERY = 8   # studies per error gallery"),
        md("## Load predictions, embeddings, labels"),
        code("""
        import numpy as np
        import pandas as pd
        from tqdm.auto import tqdm
        from pe_ct import storage, labels

        pred = pd.read_parquet(cfg.results_dir / "predictions.parquet").set_index("uid")
        records = storage.load_all_records(cfg.embeddings_dir, progress=tqdm)
        studies = pd.read_parquet(cfg.study_labels_path).set_index("study_uid")
        df = pred.join(studies.drop(columns=["y"]), how="left")
        df["manufacturer"] = [records[u]["manufacturer"] if u in records else "unknown" for u in df.index]
        df["error"] = np.select([(df.y == 1) & (df.pred == 0), (df.y == 0) & (df.pred == 1)], ["FN", "FP"], "correct")
        print(df.error.value_counts().to_dict(), "| splits:", df.split.value_counts().to_dict())
        """),
        md("## A. Misclassification analysis"),
        md("""
        ### Error breakdown

        For each flag or sub-type: how often it occurs among false negatives, false positives, and
        correctly classified studies of the same true class. A flag that is much more frequent among
        errors than among correct predictions points at a systematic failure mode.
        """),
        code("""
        flags = ["flow_artifact", "true_filling_defect_not_pe", "central_pe", "rv_lv_ratio_gte_1", "chronic_pe", "acute_and_chronic_pe"]
        df["few_pos_slices"] = (df.n_pos_slices <= 10) & (df.y == 1)
        groups = {"FN": df[df.error == "FN"], "correct PE": df[(df.y == 1) & (df.error == "correct")],
                  "FP": df[df.error == "FP"], "correct negative": df[(df.y == 0) & (df.error == "correct")]}
        rows = {name: g[flags + ["few_pos_slices"]].mean().round(3).to_dict() | {"n": len(g), "mean_prob": round(g.prob.mean(), 3)} for name, g in groups.items()}
        pd.DataFrame(rows)
        """),
        code("""
        man = pd.crosstab(df.manufacturer, df.error, normalize="index").round(3)
        man["n"] = df.manufacturer.value_counts()
        man.sort_values("n", ascending=False)
        """),
        md("""
        ### Fetch the volumes for the galleries

        The most confident false negatives (PE scans with the lowest scores), the most confident false
        positives (negative scans with the highest scores), and two confident true positives for
        reference. Only these few studies are re-downloaded.
        """),
        code("""
        from pe_ct import io as pio, pipeline

        top_fn = df[df.error == "FN"].sort_values("prob").head(N_GALLERY)
        top_fp = df[df.error == "FP"].sort_values("prob", ascending=False).head(N_GALLERY)
        top_tp = df[(df.y == 1) & (df.error == "correct")].sort_values("prob", ascending=False).head(2)
        wanted = list(dict.fromkeys(top_tp.index.tolist() + top_fn.index.tolist() + top_fp.index.tolist()))
        locator = pio.StudyLocator(pd.read_parquet(cfg.zip_index_path))
        slice_index = labels.SliceLabelIndex(labels.load_train_csv(cfg.train_csv))
        vols = pipeline.fetch_volumes_for_uids(cfg, studies, wanted, locator, slice_index, progress=tqdm)
        print(len(vols), "volumes fetched")
        """),
        code("""
        from pe_ct import volume
        import matplotlib.pyplot as plt

        def gallery(uids, title, fname):
            n = len(uids)
            fig, axes = plt.subplots(2, n, figsize=(3.1 * n, 7), squeeze=False)
            for j, u in enumerate(uids):
                v, m = vols[u]; lab = np.asarray(m["slice_labels"]); band = labels.positive_band(lab)
                z = labels.representative_positive_slice(lab) if band else v.shape[0] // 2
                flags = [f for f in ["flow_artifact", "true_filling_defect_not_pe", "central_pe", "rv_lv_ratio_gte_1", "chronic_pe"] if m[f]]
                viz.show_slice(axes[0, j], viz.crop_center(v[z], 0.55), "vessel", f"{u}\\np={df.loc[u, 'prob']:.2f}  pos slices={m['n_pos_slices']}\\n{' '.join(flags) or '-'}")
                viz.show_slice(axes[1, j], volume.coronal_slab(v), "vessel", f"coronal, z={z}", aspect=volume.aspect_coronal(m))
                viz.shade_z_band(axes[1, j], band); axes[1, j].axhline(z, color=viz.PALETTE[1], lw=0.8)
            fig.suptitle(title); fig.tight_layout()
            viz.save_fig(fig, FIG, NOTEBOOK, fname)
        """),
        plot_note(
            "Gallery: most confident false negatives",
            "Top row: the most representative positive slice of each missed PE study (middle of its longest positive run), "
            "central crop, vessel window; the caption gives the predicted probability, the number of positive slices and "
            "the sub-type flags. Bottom row: the coronal view with the labelled positive range shaded green and the shown "
            "slice marked in orange.",
            "These are the dangerous errors. Seeing them side by side is the fastest way to spot what they have in common.",
            "Few positive slices (small clots), peripheral location, no central flag, unusual scanners, or clots sitting "
            "near the top/bottom of the scan where the crop grid may not reach.",
        ),
        code("""
        gallery(top_fn.index.tolist(), "most confident false negatives (PE scored lowest)", "gallery_false_negatives")
        """),
        plot_note(
            "Gallery: most confident false positives",
            "Same layout for negative studies that scored highest. There is no positive range to shade; the shown slice is "
            "mid-chest. Captions show the artifact flags.",
            "False alarms cost radiologist time and trust. If artifact flags are over-represented here, the model reacts to "
            "dye-mixing artifacts or clot look-alikes exactly as a human novice would.",
            "Flow artifacts, true filling defects that are not PE, very bright or very faint contrast, unusual anatomy.",
        ),
        code("""
        gallery(top_fp.index.tolist(), "most confident false positives (negatives scored highest)", "gallery_false_positives")
        """),
        md("## B. Heatmaps: where the model looks"),
        md("""
        ### Load the combiner, the air baseline and the trained probe

        The backbone weights come along with the combiner (one download), but only the combiner runs.
        """),
        code("""
        import torch
        from pe_ct import embed, explain, probe

        MODEL_NAME = os.environ.get("PE_CT_MODEL", "spectre-large")   # must match the model used in notebook 03
        model = embed.load_model(MODEL_NAME, pretrained=MODEL_NAME == "spectre-large")
        air = np.load(cfg.embeddings_dir / "air_descriptor.npy")
        ckpt = torch.load(cfg.results_dir / "probe_final.pt", map_location="cpu")
        lp = probe.LinearProbe(np.zeros(ckpt["d"], np.float32), np.ones(ckpt["d"], np.float32)); lp.load_state_dict(ckpt["state_dict"])
        logit_fn = probe.predict_logit_fn(lp)
        THR = ckpt["threshold"]
        print("combiner params (M):", embed.model_info(model)["n_params_combiner_M"], "| threshold", round(THR, 3))
        """),
        code("""
        def heat_for(uid):
            r = records[uid]
            out = explain.occlusion_importance(model, r["crop_desc"], r["grid"], logit_fn, air)
            out["boxes"] = r["boxes_lpi"]; out["labels"] = r["slice_labels"]; out["shape"] = tuple(int(s) for s in r["shape_zyx"])
            return out

        heats = {u: heat_for(u) for u in tqdm(wanted)}
        """),
        plot_note(
            "Importance heatmaps",
            "For reference true positives, the top false negatives and the top false positives: a coronal view (left) and "
            "the representative axial slice (right) in the vessel window, with each crop's importance painted over its box "
            "(brighter = removing that region lowers the PE score more). The labelled positive slice range is outlined in "
            "green on the coronal view. Boxes outside the crop grid stay unpainted: the model never saw them.",
            "Answers 'what is the model reacting to?' at the resolution the model actually works at. True positives tell us "
            "what a correct call looks like; false negatives whether the model looked at the clot and still said no, or never "
            "looked there; false positives what fooled it.",
            "Does the brightest box overlap the green band? Is importance concentrated around the heart and central arteries "
            "(plausible) or at the edges, the shoulders, the table (shortcut)?",
        ),
        code("""
        def heat_panels(uids, title, fname):
            n = len(uids)
            fig, axes = plt.subplots(2, n, figsize=(3.3 * n, 7.5), squeeze=False)
            for j, u in enumerate(uids):
                v, m = vols[u]; h = heats[u]; lab = np.asarray(m["slice_labels"]); band = labels.positive_band(lab)
                heat = explain.importance_volume(h["importance"], h["boxes"], v.shape)
                z = labels.representative_positive_slice(lab) if band else int(np.nanargmax(explain.z_profile(h["importance"], h["boxes"], v.shape[0])))
                vmax = max(float(np.nanmax(heat)), 1e-6); asp = volume.aspect_coronal(m); y_mid = v.shape[1] // 2
                viz.overlay_heatmap(axes[0, j], volume.coronal_slab(v, y_mid), heat[:, y_mid, :], aspect=asp, vmax=vmax)
                if band:
                    axes[0, j].axhline(band[0] - 0.5, color=viz.COLOR_GT, lw=1.2); axes[0, j].axhline(band[1] - 0.5, color=viz.COLOR_GT, lw=1.2)
                axes[0, j].set_title(f"{u}  y={m['y']}  p={df.loc[u, 'prob']:.2f}\\nlogit {h['logit_full']:.2f}", fontsize=8)
                viz.overlay_heatmap(axes[1, j], v[z], heat[z], vmax=vmax); axes[1, j].set_title(f"axial z={z}", fontsize=8)
            fig.suptitle(title); fig.tight_layout(); viz.save_fig(fig, FIG, NOTEBOOK, fname)

        heat_panels(top_tp.index.tolist(), "true positives (reference)", "heatmaps_true_positives")
        heat_panels(top_fn.index.tolist()[:6], "false negatives", "heatmaps_false_negatives")
        heat_panels(top_fp.index.tolist()[:6], "false positives", "heatmaps_false_positives")
        """),
        md("""
        ### Did it look in the right place?

        For every PE study with an embedding, the importance profile along z is compared with the
        labelled positive slice range. The **z-hit rate** is the fraction of PE studies whose most
        important crop overlaps the positive range; a random-crop baseline says what chance looks
        like. The **band importance fraction** is the share of positive importance falling on crops
        that overlap the band.
        """),
        code("""
        pe_uids = [u for u in df.index if df.loc[u, "y"] == 1 and u in records]
        rows = []
        for u in tqdm(pe_uids):
            h = heat_for(u)
            rows.append({"uid": u, "error": df.loc[u, "error"], "prob": df.loc[u, "prob"], "n_pos_slices": df.loc[u, "n_pos_slices"],
                         "z_hit": explain.z_hit(h["importance"], h["boxes"], h["labels"]),
                         "z_hit_random": explain.random_z_hit_rate(h["boxes"], h["labels"], n_draws=200, seed=cfg.seed),
                         "band_fraction": explain.band_importance_fraction(h["importance"], h["boxes"], h["labels"]),
                         "band_in_grid": explain.covered_z_range(h["boxes"])[0] <= labels.positive_band(h["labels"])[0] and labels.positive_band(h["labels"])[1] <= explain.covered_z_range(h["boxes"])[1]})
        zhit = pd.DataFrame(rows).set_index("uid")
        print(f"z-hit rate: {zhit.z_hit.mean():.2f}  (random-crop baseline {zhit.z_hit_random.mean():.2f}) over {len(zhit)} PE studies")
        print(f"positive band fully inside the crop grid: {zhit.band_in_grid.mean():.0%} of PE studies")
        zhit.groupby("error")[["z_hit", "z_hit_random", "band_fraction"]].mean().round(3)
        """),
        plot_note(
            "Importance along z vs the ground-truth band",
            "For a few true positives and false negatives: the importance profile along the head-to-feet axis (max over "
            "the in-plane crops at each slice) with the labelled positive range shaded green. Grey margins are slices the "
            "crop grid does not cover.",
            "This is the one-dimensional version of the heatmap check and the only localisation the labels allow us to verify.",
            "True positives: the peak inside or next to the green band. False negatives: either a peak in the band (the model "
            "looked and was not convinced) or none (it never looked there), two different failure modes with different fixes.",
        ),
        code("""
        show = top_tp.index.tolist() + top_fn.index.tolist()[:4]
        fig, axes = plt.subplots(1, len(show), figsize=(2.6 * len(show), 4.5), sharey=False)
        for ax, u in zip(np.atleast_1d(axes), show):
            h = heats[u]; prof = explain.z_profile(h["importance"], h["boxes"], h["shape"][0]); z = np.arange(len(prof))
            ax.plot(np.nan_to_num(prof), z, color=viz.PALETTE[1]); ax.invert_yaxis()
            band = labels.positive_band(h["labels"])
            if band: ax.axhspan(band[0], band[1], color=viz.COLOR_GT, alpha=0.3)
            lo, hi = explain.covered_z_range(h["boxes"]); ax.axhspan(0, lo, color="gray", alpha=0.15); ax.axhspan(hi, len(prof), color="gray", alpha=0.15)
            ax.set_title(f"{u}\\n{df.loc[u, 'error']}  p={df.loc[u, 'prob']:.2f}", fontsize=8); ax.set_xlabel("importance")
        np.atleast_1d(axes)[0].set_ylabel("slice (0 = top)"); fig.tight_layout()
        viz.save_fig(fig, FIG, NOTEBOOK, "z_profiles");
        """),
        md("""
        ### Missed-focus examples

        False negatives where the ground-truth band received little importance: the "should have
        looked here but did not" cases.
        """),
        code("""
        missed = zhit[(zhit.error == "FN")].sort_values("band_fraction").head(8)
        missed[["prob", "n_pos_slices", "z_hit", "band_fraction", "band_in_grid"]]
        """),
        code("""
        storage.atomic_write_parquet(zhit.reset_index(), cfg.results_dir / "z_hit_analysis.parquet")
        print("saved", cfg.results_dir / "z_hit_analysis.parquet")
        """),
        md("""
        ## What we learned, and what to do next

        Fill in after running. Suggested structure:

        - **Errors.** Which flags and sub-types are over-represented among FNs and FPs; whether a scanner stands out.
        - **Focus.** z-hit rate vs the random baseline; share of FNs where the model never looked at the band vs looked and declined.
        - **Coverage.** Share of PE studies whose positive band is fully inside the crop grid (clots in the uncovered margin are undetectable by construction).
        - **Next steps.** If recall on small clots is poor and the model does look at the right region: an attention / MIL head
          over the crop tokens (E7). If the model does not look there: resampling to a fixed spacing or finer crops. If the
          learning curve was still rising: more embeddings (E8). If everything plateaus: light fine-tuning (E9).
        """),
    ]
