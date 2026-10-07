"""03_extract_embeddings: frozen SPECTRE on dev + test studies, streaming."""

from scripts.nb.tools import code, config_cell, learned_cell, md, plot_note, setup_cell

TITLE = "03_extract_embeddings"


def cells():
    return [
        md("""
        # 03 - Extract frozen embeddings with SPECTRE

        **Goal.** Run every dev and test-sample study through a frozen 3D CT foundation model and
        store one scan-level embedding plus the per-crop descriptors needed later for heatmaps.

        **The backbone.** SPECTRE (`cclaess/SPECTRE-Large`) is a 3D vision transformer pretrained on
        chest and abdomen CT. It tiles the scan into fixed-size 3D crops, runs a ViT on each crop
        (the *backbone*), pools each crop to one descriptor, and runs a small transformer over the
        crop descriptors (the *feature combiner*) to produce one scan embedding plus one token per crop.
        License: CC BY-NC-SA (research use only).

        **Preprocessing is the model's own.** The library orients the scan, windows HU to the
        range it was trained with and tiles it into its crop grid; we only hand it a volume in
        Hounsfield units in the orientation it expects. Resampling to a fixed voxel spacing is
        optional in the library and is controlled by one config value, recorded with every record.

        Glossary: **token** = one vector the transformer works with (here: one per crop);
        **CLS** = the extra token that summarises the whole input, used as the scan embedding;
        **crop grid** = the arrangement of non-overlapping boxes the scan is cut into.
        """),
        setup_cell(),
        md("## Configuration"),
        config_cell(TITLE, "cfg.resample_spacing = None   # None = native spacing (the library default); e.g. (0.75, 0.75, 1.5)"),
        md("""
        ## Install and load the backbone

        Half precision on the GPU; the model is loaded once in eval mode.
        """),
        code("""
        import subprocess, sys
        try:
            import spectre
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "spectre-fm[inference]", "umap-learn"], check=True)
            import spectre
        print("spectre-fm", spectre.__version__)
        """),
        code("""
        import torch
        from pe_ct import embed, colab

        report = colab.session_report(cfg)
        MODEL_NAME = os.environ.get("PE_CT_MODEL", "spectre-large")   # a random "spectre-small" is used only for local smoke tests
        model = embed.load_model(MODEL_NAME, pretrained=MODEL_NAME == "spectre-large")
        info = embed.model_info(model)
        info
        """),
        md("## Load splits, labels and the lookup objects"),
        code("""
        import pandas as pd
        from pe_ct import io as pio, labels, storage

        studies = pd.read_parquet(cfg.study_labels_path)
        split_table = pd.read_parquet(cfg.splits_path)
        zip_index = pd.read_parquet(cfg.zip_index_path)
        locator = pio.StudyLocator(zip_index)
        slice_index = labels.SliceLabelIndex(labels.load_train_csv(cfg.train_csv))
        dev_uids = split_table.loc[split_table.in_dev, "study_uid"].tolist()
        test_uids = split_table.loc[split_table.in_test_sample, "study_uid"].tolist()
        print(len(dev_uids), "dev |", len(test_uids), "test sample")
        """),
        md("""
        ## What the model sees

        One EDA volume (already on Drive) is pushed through the model's preprocessing and the crop
        grid is drawn on top.
        """),
        code("""
        from tqdm.auto import tqdm
        from pe_ct import volume

        storage.prepare_local(cfg.volumes_eda_dir, cfg.work / "volumes_eda", progress=tqdm)
        store = storage.VolumeStore(cfg.work / "volumes_eda" / "cache")
        demo_uid = next(u for u in store.uids() if store.load_meta(u)["y"] == 1)
        demo_vol, demo_meta = store.load_volume(demo_uid)
        demo_img = volume.array_to_image(demo_vol, demo_meta)
        tensor, prep = embed.prepare_input(demo_img, cfg.resample_spacing)
        layout = embed.crop_layout(prep["ras_shape"], info["crop_size"])
        boxes = embed.crop_boxes_lpi(layout)
        print("input tensor (1, R, A, S):", tuple(tensor.shape), "| crop grid (R, A, S):", layout["grid"], "| crops:", len(boxes))
        """),
        plot_note(
            "What the model sees",
            "Left: a coronal view of the stored volume in the vessel window, head at the top. Right: the same view "
            "after the model's own preprocessing (its wide HU window), with the crop grid drawn in orange: every box "
            "is one 3D crop that becomes one token. Green marks the PE-positive slice range.",
            "The model does not see pixels, it sees a handful of boxes. The heatmaps in the last notebook will have "
            "exactly this box resolution, and slices outside the grid (grey bands at the top/bottom) are never seen at all.",
            "A few boxes across and a few boxes down. Check whether the positive band overlaps a box edge or the "
            "unseen margin; clots in the margin can not be detected at all by this pipeline.",
        ),
        code("""
        import matplotlib.pyplot as plt

        band = labels.positive_band(demo_meta["slice_labels"])
        fig, axes = plt.subplots(1, 2, figsize=(11, 6))
        asp = volume.aspect_coronal(demo_meta)
        viz.show_slice(axes[0], volume.coronal_slab(demo_vol), "vessel", "stored volume, vessel window", aspect=asp)
        viz.show_slice(axes[1], volume.coronal_slab(demo_vol), (0, 2000), "model input window with crop grid", aspect=asp)
        for (z0, z1), (y0, y1), (x0, x1) in boxes:
            axes[1].add_patch(plt.Rectangle((x0 - 0.5, z0 - 0.5), x1 - x0, z1 - z0, fill=False, ec=viz.PALETTE[1], lw=1))
        for ax in axes:
            viz.shade_z_band(ax, band)
        viz.save_fig(fig, FIG, NOTEBOOK, "what_the_model_sees");
        """),
        md("""
        ## Three studies end to end

        Timing is split into GPU time (backbone + combiner) and CPU time (fetch + decode), because
        the CPU side is expected to be the bottleneck.
        """),
        code("""
        import time
        from pe_ct import pipeline

        by_uid = studies.set_index("study_uid")
        for uid in dev_uids[:3]:
            t0 = time.time()
            image, meta, timings = pipeline.process_study(cfg, locator, slice_index, uid, by_uid.loc[uid], cfg.work / "dicom")
            rec = embed.embed_image(model, image, cfg.resample_spacing)
            print(f"{uid}: cls {rec['cls'].shape}, crops {rec['n_crops']} grid {tuple(rec['grid'])}, "
                  f"gpu {rec['t_gpu_s']:.1f}s, download {timings['t_download_s']:.1f}s, decode {timings['t_decode_s']:.1f}s, total {time.time() - t0:.1f}s")
        """),
        md("""
        ## Main loop (resumable, streaming)

        Fetch -> volume -> embed -> append to the current shard -> delete the DICOMs. While the GPU
        embeds one study, background threads fetch and decode the next ones. Finished shards are
        copied to Drive with their marker; failures are logged with the reason and skipped on re-runs.
        The occlusion baseline (the descriptor of a crop of pure air) is saved once alongside.
        """),
        code("""
        import numpy as np

        air_path = cfg.embeddings_dir / "air_descriptor.npy"
        if not air_path.exists():
            storage.atomic_write_npy(air_path, embed.air_descriptor(model).astype(np.float32))
        failure_log = storage.FailureLog(cfg.failures_csv(NOTEBOOK))
        stats = pipeline.run_embedding_extraction(cfg, model, studies, dev_uids + test_uids, locator, slice_index, failure_log, prefetch=2, progress=tqdm)
        """),
        md("## Summary"),
        code("""
        done = storage.done_uids(cfg.embeddings_dir, "npz")
        print(f"this run: {stats['done']} done, {stats['failed']} failed | stored in total: {len(done)} of {len(dev_uids) + len(test_uids)}")
        if stats["seconds"]:
            print(f"seconds per study: mean {np.mean(stats['seconds']):.1f} | GPU {np.mean(stats['t_gpu']):.1f} | CPU fetch+decode {np.mean(stats['t_cpu']):.1f}")
        failure_log.df.tail()
        """),
        md("""
        ## Embedding sanity check

        All stored scan embeddings are projected to 2D with UMAP (a neighbourhood-preserving
        projection) for a first look at what the embedding encodes.
        """),
        code("""
        records = storage.load_all_records(cfg.embeddings_dir, progress=tqdm)
        X = np.stack([r["cls"] for r in records.values()])
        y = np.array([r["y"] for r in records.values()])
        man = np.array([r.get("scanner", r["manufacturer"]) for r in records.values()])
        print(X.shape, "embeddings |", f"{y.mean():.1%} PE")
        """),
        plot_note(
            "UMAP of scan embeddings",
            "Each point is one study's scan embedding projected to 2D. Left: coloured by PE label. Right: coloured by "
            "scanner (reconstruction kernel as a proxy, since the manufacturer tag is missing in this dataset).",
            "If points cluster strongly by scanner, the embedding encodes the scanner more than the anatomy, a "
            "shortcut risk for the classifier. Some separation by PE label would be a very good sign; none is "
            "expected for a subtle disease and is not a failure, since a linear probe can use directions UMAP discards.",
            "Left: whether PE points concentrate anywhere. Right: whether scanners form separate islands "
            "(bad) or mix (good).",
        ),
        code("""
        import umap

        emb2 = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=cfg.seed).fit_transform((X - X.mean(0)) / (X.std(0) + 1e-6))
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        for v, c, name in [(0, viz.COLOR_NEG, "negative"), (1, viz.COLOR_PE, "PE")]:
            axes[0].scatter(*emb2[y == v].T, s=6, color=c, label=name, alpha=0.7)
        axes[0].legend(); axes[0].set_title("coloured by PE label")
        for i, m in enumerate(pd.Series(man).value_counts().index):
            axes[1].scatter(*emb2[man == m].T, s=6, color=viz.PALETTE[i % 10], label=f"{m} ({(man == m).sum()})", alpha=0.7)
        axes[1].legend(fontsize=7); axes[1].set_title("coloured by scanner")
        for ax in axes: ax.set_xticks([]); ax.set_yticks([])
        viz.save_fig(fig, FIG, NOTEBOOK, "umap_embeddings");
        """),
        learned_cell([
            "(fill in after running) seconds per study, and whether GPU or CPU dominated.",
            "(fill in after running) typical crop grid per scan and how many slices fall outside the grid.",
            "(fill in after running) does the UMAP show scanner clusters, PE clusters, neither?",
            "The per-crop descriptors are stored alongside the scan embedding, so heatmaps later need only the small combiner, not the backbone.",
        ]),
    ]
