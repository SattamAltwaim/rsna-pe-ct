"""00_setup_and_splits: labels, zip index, study table, frozen splits."""

from scripts.nb.tools import code, config_cell, learned_cell, md, plot_note, setup_cell

TITLE = "00_setup_and_splits"


def cells():
    return [
        md("""
        # 00 - Setup and splits

        **Goal.** Load the label file, collapse the slice labels into one row per CT study, and
        freeze leak-free train / validation / test splits *before looking at a single image*.

        Everything downstream (volumes, embeddings, the linear probe, the error analysis) reads
        the split table written here. The splits are seeded, so later notebooks rebuild the
        identical table if they cannot find it (useful on Kaggle, where each notebook has its
        own output folder).

        **Data source.** On Colab the DICOMs are streamed from the public S3 zip with byte-range
        requests and outputs go to Google Drive. On Kaggle the competition dataset is mounted
        read-only under `/kaggle/input` and outputs go to `/kaggle/working` (persisted as the
        notebook's output when you *Save Version*). The code detects which one it is on.
        """),
        md("""
        ## CT scans for ML engineers (read once)

        - **A CT scan is a 3D grayscale volume**, shape roughly `(slices, 512, 512)`. The closest
          computer-vision analogy is a short video clip whose frames are cross-sections of the body
          from the neck down to the belly. One frame is called a **slice** (also an *axial* slice).
        - **Pixel values are Hounsfield Units (HU)**, a calibrated physical density scale. Air is about
          -1000, fat about -100, water 0, soft tissue about +40, contrast-filled blood +200 to +400,
          bone about +1000. The same number means the same tissue on any scanner, which is unusual
          for images and very convenient.
        - **Windowing** means clipping to an HU range and rescaling to [0, 1] for display or for a
          model. Different windows reveal different tissue (a "lung window" vs a "vessel window").
        - **Voxel spacing** is the physical size of one voxel in mm. It varies by scanner, so models
          usually resample to a fixed spacing: the 3D version of resizing.
        - **DICOM** is the medical image file format: one file per slice with a metadata header.
          Hierarchy: **Study** (one exam = one sample here) -> **Series** (one 3D reconstruction;
          exactly one per study in this dataset) -> **SOP Instance** (one slice file).
        - **Pulmonary embolism (PE)** is a blood clot blocking arteries in the lungs. Dangerous and urgent.
        - **CTPA** (CT pulmonary angiogram) is a chest CT taken right after injecting contrast dye,
          so blood inside the lung arteries appears **bright**.
        - **A clot looks like a dark spot inside a bright vessel** (a *filling defect*): the clot blocks
          the dye. In CV terms: small, low-contrast dark objects inside thin bright tubes, somewhere in
          a large 3D volume.
        - **Label limitation.** Labels say *which slices* contain a clot, not *where in the slice*.
          There are no boxes or masks, so clots can only be localised along the head-to-feet axis.
        """),
        setup_cell(),
        md("## Configuration"),
        config_cell(TITLE),
        md("""
        ## Session and storage

        Mount Drive, create the folder layout and print how much space is free. Raw DICOM
        files never touch Drive (they go to Colab's local disk and are deleted after use);
        the only large thing we store is the EDA volume subset.
        """),
        code("""
        from pe_ct import colab
        report = colab.session_report(cfg)
        """),
        md("""
        ## Labels: `train.csv`

        One row per slice. On Kaggle it is copied from the mounted competition dataset; on Colab
        just that member is fetched from the dataset zip with an HTTP range request.
        """),
        code("""
        from pe_ct import labels, pipeline, storage

        train = pipeline.load_train(cfg)
        print(train.shape, "slice rows |", train.StudyInstanceUID.nunique(), "studies")
        train.head(3)
        """),
        md("""
        ## Zip index (S3 source only)

        When the DICOMs are streamed from the public zip, each file is fetched with a byte-range
        request, which needs the member offsets from the zip's central directory. Reading it
        (millions of entries) takes a minute, so it is done once and saved as a parquet table.
        On Kaggle the files are already on disk and this step is skipped.
        """),
        code("""
        locator = pipeline.make_locator(cfg)
        if locator is not None:
            import pandas as pd
            zip_index = pd.read_parquet(cfg.zip_index_path)
            print(len(zip_index), "entries |", zip_index.is_dcm.sum(), "DICOM files")
            print("compression types (0 = stored, 8 = deflate):", zip_index.compress_type.value_counts().to_dict())
        else:
            print("Kaggle source: DICOMs are read from", cfg.kaggle_input)
        """),
        md("""
        ## One row per study

        All study-level columns are constant within a study, and every study has exactly one
        series; both facts are asserted while collapsing. The three mutually exclusive study
        groups are *negative*, *PE* and *indeterminate* (poor-quality scans that are neither).
        """),
        code("""
        studies = labels.study_table(train)
        storage.atomic_write_parquet(studies, cfg.study_labels_path)
        print(studies.shape)
        studies.head()
        """),
        code("""
        print("slices per study: median", studies.n_slices.median(), "| min", studies.n_slices.min(), "| max", studies.n_slices.max())
        pe = studies[studies.group == "pe"]
        print("positive slices per PE study: median", pe.n_pos_slices.median(), "| range", pe.n_pos_slices.min(), "-", pe.n_pos_slices.max())
        labels.group_counts(studies)
        """),
        md("""
        ## Drop indeterminate studies

        Indeterminate studies carry a quality flag (motion or poor contrast) and are neither
        positive nor negative. They are excluded from the binary task.
        """),
        code("""
        usable = labels.usable_studies(studies)
        print(len(usable), "usable studies |", int(usable.y.sum()), "PE positive |", f"{usable.y.mean():.1%} prevalence")
        print("indeterminate quality flags:", studies.loc[studies.group == "indeterminate", ["qa_motion", "qa_contrast"]].sum().to_dict())
        """),
        md("""
        ## Freeze the splits

        A stratified test pool is carved out first and not touched again until the final cell of
        the linear-probe notebook. From the remaining dev pool we sample the working subsets:
        a class-balanced EDA subset, a dev subset with stratified folds for cross-validation, and
        a test sample drawn from the frozen pool. Stratification uses the label together with two
        clinically important sub-types so rare kinds of PE are represented everywhere.
        """),
        code("""
        from pe_ct import splits

        split_table = splits.make_splits(usable, cfg)
        splits.check_no_leakage(split_table)
        summary = splits.split_summary(split_table)
        summary
        """),
        plot_note(
            "Class balance per split",
            "One group of bars per subset; bar height is the share of PE-positive studies (left panel) "
            "and the share of central or heart-straining PE among the positives (right panel).",
            "If stratification worked, every subset has the same mix, so a model's score on the dev folds "
            "transfers to the test sample. A mismatch here would make every later comparison unfair.",
            "All bars within a panel at the same height. Differences larger than a couple of percent "
            "would mean the sampling went wrong.",
        ),
        code("""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        summary["pe_frac"].plot.bar(ax=axes[0], color=viz.COLOR_PE, title="PE prevalence per subset")
        summary[["central_pe_frac_of_pe", "rv_lv_gte_1_frac_of_pe"]].plot.bar(ax=axes[1], title="sub-type share among PE studies")
        for ax in axes:
            ax.set_ylim(0, 1); ax.set_xlabel("")
        fig.tight_layout()
        viz.save_fig(fig, FIG, NOTEBOOK, "class_balance_per_split");
        """),
        md("## Save the split table"),
        code("""
        storage.atomic_write_parquet(split_table, cfg.splits_path)
        print("saved", cfg.splits_path)
        split_table.head()
        """),
        learned_cell([
            "The usable dataset is a few thousand studies with roughly a 30 % PE prevalence; the class imbalance is moderate, not extreme.",
            "A typical study has a couple of hundred slices and a PE study has a few dozen positive slices, but the range is huge (from a single slice to almost the whole scan).",
            "On the S3 source every zip member is deflate-compressed, so each DICOM is inflated after its range request (about half the raw size travels over the network); on Kaggle the files are read directly.",
            "The test pool is frozen here and must not be opened again until the last cell of the linear-probe notebook.",
            "(fill in after running) anything surprising in the split summary table.",
        ]),
    ]
