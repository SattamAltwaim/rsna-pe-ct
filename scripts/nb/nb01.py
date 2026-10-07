"""01_download_subset: DICOMs -> oriented HU volumes with aligned slice labels, in tar shards."""

from scripts.nb.tools import code, config_cell, learned_cell, md, plot_note, setup_cell

TITLE = "01_download_subset"


def cells():
    return [
        md("""
        # 01 - Download the EDA subset

        **Goal.** Turn zipped DICOM slices into clean 3D volumes in Hounsfield units, with the
        slice labels aligned to the volume's z-axis, and store them on Drive in shards.

        Glossary for this notebook:

        - **Slice**: one 2D frame of the 3D volume (one DICOM file).
        - **Axial / coronal / sagittal**: the three viewing planes. Think of the body as a loaf of
          bread standing upright: *axial* slices are the normal bread slices seen from above,
          *coronal* cuts it front-to-back (a view from the front), *sagittal* cuts it left-to-right
          (a view from the side).
        - **HU**: Hounsfield units, the calibrated density scale (air -1000, water 0, bone about +1000).
        - **Series reader**: a library routine that sorts the slice files by physical position,
          applies the scanner's intensity calibration and assembles the 3D geometry.
        - **z-axis**: the head-to-feet axis. In our stored volumes index 0 is the top of the scan.
        """),
        setup_cell(),
        md("## Configuration"),
        config_cell(TITLE),
        code("""
        from pe_ct import colab
        report = colab.session_report(cfg)
        """),
        md("""
        ## Load the split table and select the EDA subset

        We also build two lookup objects: one that maps a study to its byte offsets in the zip,
        and one that maps a study to its per-slice labels.
        """),
        code("""
        import pandas as pd
        from pe_ct import io as pio, labels

        split_table = pd.read_parquet(cfg.splits_path)
        studies = pd.read_parquet(cfg.study_labels_path)
        eda_uids = split_table.loc[split_table.in_eda, "study_uid"].tolist()
        print(len(eda_uids), "EDA studies |", int(split_table.loc[split_table.in_eda, "y"].sum()), "PE positive")
        """),
        code("""
        zip_index = pd.read_parquet(cfg.zip_index_path)
        locator = pio.StudyLocator(zip_index)
        slice_index = labels.SliceLabelIndex(labels.load_train_csv(cfg.train_csv))
        print(len(locator.study_uids), "studies indexed")
        """),
        md("""
        ## One study through the pipeline

        Fetch, decode, orient, align labels. The printout checks the three things that silently
        break everything if wrong: the shape, the voxel spacing, and the HU range (air must sit
        near -1000; contrast-filled vessels and bone must reach several hundred).
        """),
        code("""
        from pe_ct import pipeline, volume

        uid = eda_uids[0]
        image, meta, timings = pipeline.process_study(cfg, locator, slice_index, studies.set_index("study_uid").loc[uid], cfg.work / "dicom")
        vol = volume.image_to_hu_array(image)
        print("study", uid, "| shape (z, y, x)", vol.shape, "| spacing (x, y, z) mm", [round(s, 3) for s in meta["spacing_xyz_mm"]])
        print("timings", timings, "| scanner", meta["manufacturer"], "| slice thickness", meta["slice_thickness_mm"], "mm")
        print("HU sanity", meta["hu_sanity"])
        print("positive slices:", labels.positive_runs(meta["slice_labels"]), "of", len(meta["slice_labels"]))
        """),
        plot_note(
            "Three orthogonal views of one study",
            "Left: an axial slice (seen from the feet, the standard radiology view: the patient's front is at the "
            "top, their left is on the right of the screen). Middle: a coronal cut (seen from the front). Right: a "
            "sagittal cut (seen from the side). Orange lines show where the other two planes cut. The thin strip on "
            "the far right marks PE-positive slices in red along the z-axis. Vessel window.",
            "This is the visual proof that the volume was assembled correctly: slices in the right order, the head "
            "at the top, no left-right mirroring, and voxel spacing applied so the body is not squashed.",
            "A chest that looks like a chest: lungs dark, heart and large vessels bright, spine at the back. "
            "The red band should fall at the level of the heart, where the lung arteries are.",
        ),
        code("""
        z = labels.representative_positive_slice(meta["slice_labels"]) or vol.shape[0] // 2
        fig = viz.orthoviews(vol, meta, window="vessel", z=z, labels=meta["slice_labels"], title=f"study {uid}")
        viz.save_fig(fig, FIG, NOTEBOOK, "orthoviews_one_study");
        """),
        md("""
        ## Build the EDA volumes (resumable)

        Each study is fetched with parallel range requests to local disk, decoded into a volume,
        appended to the current shard and its DICOM files deleted. A shard is copied to Drive and
        gets its `.done.json` marker only when complete, so a disconnected session loses at most
        one partial shard. Studies that fail are logged with the reason and skipped on re-runs.
        """),
        code("""
        from tqdm.auto import tqdm
        from pe_ct import storage

        failure_log = storage.FailureLog(cfg.failures_csv(NOTEBOOK))
        stats = pipeline.run_volume_download(cfg, studies, eda_uids, locator, slice_index, failure_log, progress=tqdm)
        """),
        md("""
        ## Summary

        The measured seconds per study is the number that sets the budget for the embedding
        notebook, which streams many more studies through the same fetch-and-decode path.
        """),
        code("""
        import numpy as np

        done_uids = storage.done_uids(cfg.volumes_eda_dir, "tar")
        secs = np.array(stats["seconds"]) if stats["seconds"] else np.array([np.nan])
        print(f"done this run: {stats['done']} | failed: {stats['failed']} | stored in total: {len(done_uids)} of {len(eda_uids)}")
        print(f"seconds per study: median {np.nanmedian(secs):.1f} | mean {np.nanmean(secs):.1f}")
        print(f"GB written this run: {stats['bytes'] / 2**30:.2f} | Drive free: {colab.free_space_gb(cfg.root)} GB")
        failure_log.df.tail()
        """),
        md("""
        ## Local copy of the shards

        Reading many files from Drive is slow, so the finished shards are copied to local disk
        and extracted once. The EDA notebook does the same at its start.
        """),
        code("""
        fresh = storage.prepare_local(cfg.volumes_eda_dir, cfg.work / "volumes_eda", progress=tqdm)
        store = storage.VolumeStore(cfg.work / "volumes_eda" / "cache")
        print(len(fresh), "shards extracted now |", len(store.uids()), "volumes available locally")
        """),
        plot_note(
            "Coronal views of random volumes",
            "A grid of coronal (front) views through the middle of randomly chosen stored volumes, vessel window, "
            "head at the top. The caption gives the study id, the label and the scanner when known.",
            "A quick visual check across different scanners: every panel must look like a chest. One wrong-way-up "
            "or mirrored volume here would mean the orientation code fails on some scanner's conventions.",
            "Head at the top, lungs as two dark lobes, the bright heart between them, the spine running down the middle. "
            "Aspect ratios should look natural (slices are thicker than in-plane pixels, which the plot corrects for).",
        ),
        code("""
        rng = np.random.default_rng(cfg.seed)
        sample = rng.choice(store.uids(), size=min(5, len(store.uids())), replace=False)
        images, titles, aspects = [], [], []
        for u in sample:
            v, m = store.load_volume(u)
            images.append(volume.coronal_slab(v)); aspects.append(volume.aspect_coronal(m))
            titles.append(f"{u}  y={m['y']}  {m['manufacturer'][:12]}")
        fig = viz.image_grid(images, titles, ncols=5, window="vessel", aspects=aspects)
        viz.save_fig(fig, FIG, NOTEBOOK, "coronal_random_volumes");
        """),
        learned_cell([
            "(fill in after running) seconds per study and where the time goes (network vs decode).",
            "(fill in after running) how many studies failed and why (compressed pixel data, missing slices, odd geometry).",
            "(fill in after running) whether any scanner needed special handling for orientation.",
            "Volumes are stored head-first in LPI orientation, so slice 0 is the top of the scan in every notebook.",
        ]),
    ]
