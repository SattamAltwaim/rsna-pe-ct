"""02_eda: what the data, the scans and the clots look like."""

from scripts.nb.tools import code, config_cell, learned_cell, md, plot_note, setup_cell

TITLE = "02_eda"


def cells():
    return [
        md("""
        # 02 - Exploratory data analysis

        **Goal.** A first-time CT user should finish this notebook understanding what the data looks
        like, what a clot looks like, and where clots occur along the body.

        Glossary for this notebook:

        - **Slice**: one 2D frame of the 3D volume, like one frame of a video. **Axial** slices are
          the native frames (seen from the feet); **coronal** and **sagittal** views are re-slices
          from the front and from the side.
        - **HU** (Hounsfield units): calibrated density. Air -1000, fat -100, water 0, soft tissue
          +40, contrast-filled blood +200 to +400, bone +1000.
        - **Window**: an HU range mapped to black..white for display. The lung window shows air-filled
          detail, the soft-tissue window organs, the vessel window the contrast-filled arteries.
        - **CTPA**: chest CT taken right after injecting contrast dye, so blood in the lung arteries is bright.
        - **Filling defect**: a dark spot inside a bright vessel where the dye could not flow because a
          clot is in the way. This is what a PE looks like.
        - **Sub-labels** (PE studies only): *central* (clot in the big arteries near the heart) vs
          peripheral; *RV/LV ratio >= 1* (the right heart chamber is enlarged, a sign of strain from a
          big clot); *chronic* vs acute (old vs fresh clot); *left-/right-sided*.
        - **Look-alikes**: *flow artifact* (dye mixing unevenly) and *true filling defect, not PE*
          (something else blocking the dye). They fool humans and models alike.
        """),
        setup_cell(),
        md("## Configuration"),
        config_cell(TITLE),
        md("""
        ## Load labels, splits and the EDA volumes

        The finished volume shards are copied from Drive to local disk once (fast random access
        afterwards). Volumes are read through a small store object, so no cell handles tar files.
        """),
        code("""
        import numpy as np
        import pandas as pd
        from tqdm.auto import tqdm
        from pe_ct import labels, storage, volume

        studies = pd.read_parquet(cfg.study_labels_path)
        split_table = pd.read_parquet(cfg.splits_path)
        storage.prepare_local(cfg.volumes_eda_dir, cfg.work / "volumes_eda", progress=tqdm)
        store = storage.VolumeStore(cfg.work / "volumes_eda" / "cache")
        eda_uids = store.uids()
        print(len(studies), "labelled studies |", len(eda_uids), "EDA volumes on local disk")
        """),
        code("""
        metas = {u: store.load_meta(u) for u in eda_uids}
        acq = pd.DataFrame([{
            "study_uid": u, "y": m["y"], "n_slices": m["shape_zyx"][0],
            "slice_spacing_mm": m["slice_spacing_median_mm"], "inplane_mm": m["spacing_xyz_mm"][0],
            "manufacturer": m["manufacturer"], "kvp": m["kvp"], "n_missing": m["n_missing_slices_est"],
        } for u, m in metas.items()]).set_index("study_uid")
        pe_uids = [u for u in eda_uids if metas[u]["y"] == 1]
        neg_uids = [u for u in eda_uids if metas[u]["y"] == 0]
        print(len(pe_uids), "PE |", len(neg_uids), "negative in the EDA subset")
        """),
        md("## A. The labels (all studies)"),
        plot_note(
            "Study groups",
            "Three bars: the number of negative, PE-positive and indeterminate studies in the whole label file.",
            "This is the class imbalance the classifier has to live with, and the reason we report precision-recall "
            "and macro F1 rather than plain accuracy.",
            "Roughly two negatives per positive; a small indeterminate group that we exclude.",
        ),
        code("""
        import matplotlib.pyplot as plt

        counts = labels.group_counts(studies)["n_studies"]
        fig, ax = plt.subplots(figsize=(6, 3.5))
        viz.bar_counts(ax, counts, colors=[viz.COLOR_NEG, viz.COLOR_PE, viz.COLOR_IND], title="study groups")
        viz.save_fig(fig, FIG, NOTEBOOK, "study_groups");
        """),
        plot_note(
            "Sub-label prevalence among PE studies",
            "Horizontal bars: the fraction of PE-positive studies carrying each sub-label.",
            "Tells us which kinds of PE are common and which are rare; rare kinds need stratified sampling "
            "and get their own recall numbers later.",
            "Acute, one-sided clots dominate; central clots and heart strain are a minority; chronic clots are rare.",
        ),
        code("""
        prev = labels.sublabel_prevalence(studies)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        viz.bar_counts(ax, prev.round(3), horizontal=True, title="share of PE studies with each sub-label")
        ax.set_xlim(0, 1)
        viz.save_fig(fig, FIG, NOTEBOOK, "sublabel_prevalence");
        """),
        plot_note(
            "Sub-label co-occurrence",
            "A matrix: cell (row, column) is the probability that a PE study with the row label also carries the "
            "column label. The diagonal is 1 by construction.",
            "The labels are not independent: a clot that is central tends to strain the heart, and a two-sided clot "
            "is both left- and right-sided. Any analysis by sub-type has to keep this in mind.",
            "Bright off-diagonal cells, especially central -> RV/LV >= 1. Mutually exclusive pairs (RV/LV >= 1 vs < 1) are 0.",
        ),
        code("""
        co = labels.sublabel_cooccurrence(studies)
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(co.values, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(co))); ax.set_xticklabels(co.columns, rotation=60, ha="right")
        ax.set_yticks(range(len(co))); ax.set_yticklabels(co.index)
        for i in range(len(co)):
            for j in range(len(co)):
                ax.text(j, i, f"{co.values[i, j]:.2f}", ha="center", va="center", fontsize=8, color="w" if co.values[i, j] > 0.6 else "k")
        ax.set_title("P(column | row) among PE studies"); fig.colorbar(im, fraction=0.046)
        viz.save_fig(fig, FIG, NOTEBOOK, "sublabel_cooccurrence");
        """),
        plot_note(
            "Positive slices per PE study",
            "Histogram of how many slices are labelled positive in each PE study, on a log x-axis.",
            "A clot visible on one slice is a tiny object in a volume of hundreds of slices; one visible on two "
            "hundred slices is huge. This spread predicts which studies will be hard for a whole-scan embedding.",
            "A wide, roughly log-normal spread: the median is a few dozen slices, with a long tail in both directions.",
        ),
        code("""
        pe_all = studies[studies.group == "pe"]
        fig, ax = plt.subplots(figsize=(7, 3.5))
        bins = np.logspace(0, np.log10(pe_all.n_pos_slices.max() + 1), 30)
        ax.hist(pe_all.n_pos_slices, bins=bins, color=viz.COLOR_PE)
        ax.set_xscale("log"); ax.set_xlabel("positive slices per PE study"); ax.set_ylabel("studies")
        ax.axvline(pe_all.n_pos_slices.median(), color="k", ls="--", lw=1, label=f"median = {pe_all.n_pos_slices.median():.0f}")
        ax.legend()
        viz.save_fig(fig, FIG, NOTEBOOK, "positive_slices_per_study");
        """),
        md("## B. The scans"),
        md("""
        ### Interactive slice browser

        Pick a study, drag the slider to move along the z-axis (top of the scan at 0), change the
        window. The thin strip on the right marks the PE-positive slices in red and the orange line
        is the slice you are looking at. This is the single most useful way to "get" a CT:
        it really is a stack of frames.
        """),
        code("""
        viz.slice_scroller(store.load_volume, pe_uids[:20] + neg_uids[:10], labels_lookup=lambda u, m: m["slice_labels"])
        """),
        plot_note(
            "The same slice in four windows",
            "One axial slice through the lung arteries of a PE study (a slice labelled positive), shown with the full "
            "HU range and three clinical windows. Level L is the HU value mapped to mid-gray; width W the HU range "
            "spanned from black to white.",
            "A model's input preprocessing is exactly such a window. The clot is invisible in some windows and "
            "visible in others, which is why the preprocessing must match what the backbone was trained with.",
            "Full range: almost everything is mid-gray. Lung window: fine airway detail, vessels saturated white. "
            "Soft tissue and vessel windows: bright branching arteries in the center with (possibly) a darker spot inside.",
        ),
        code("""
        demo_uid = max(pe_uids, key=lambda u: metas[u]["n_pos_slices"])   # the clearest clot: most positive slices
        demo_vol, demo_meta = store.load_volume(demo_uid)
        demo_z = labels.representative_positive_slice(demo_meta["slice_labels"])
        fig = viz.window_comparison(demo_vol[demo_z], title=f"study {demo_uid}, slice {demo_z} (PE positive)")
        viz.save_fig(fig, FIG, NOTEBOOK, "windowing_comparison");
        """),
        plot_note(
            "HU histograms",
            "Voxel-value histograms (log y) of a few volumes, with dashed lines at the textbook HU of air, fat, water, "
            "soft tissue, contrast-filled blood and bone.",
            "Checks that the intensity calibration is right: if the peaks did not line up with the physical tissues, "
            "every window and the backbone's preprocessing would be off.",
            "A huge peak at -1000 (air inside and around the body), a broad bump around 0 to +60 (soft tissue), a "
            "shoulder in the +200 to +400 range (contrast in vessels) and a long tail to +1000 and beyond (bone).",
        ),
        code("""
        sample_uids = pe_uids[:2] + neg_uids[:2]
        vols = [store.load_volume(u)[0] for u in sample_uids]
        fig, ax = plt.subplots(figsize=(9, 4))
        viz.hu_histogram(ax, vols, labels=[f"{u} (y={metas[u]['y']})" for u in sample_uids])
        viz.save_fig(fig, FIG, NOTEBOOK, "hu_histograms");
        """),
        plot_note(
            "Acquisition variability",
            "Four panels over the EDA volumes: number of slices, slice spacing (mm), in-plane pixel size (mm), and "
            "the scanner manufacturer with the PE rate per manufacturer written above each bar.",
            "Raw scans are inconsistent in resolution, which motivates resampling or a model robust to it. The last "
            "panel is a shortcut check: if one manufacturer contributed mostly PE cases, a model could learn to "
            "recognise the scanner instead of the disease.",
            "Slice spacing clustering at a few standard values; in-plane size varying with patient size. PE rates "
            "per manufacturer close to the overall prevalence; a manufacturer far from it is a confounder to watch.",
        ),
        code("""
        fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
        axes[0].hist(acq.n_slices, bins=30, color=viz.PALETTE[0]); axes[0].set_title("slices per study")
        axes[1].hist(acq.slice_spacing_mm, bins=30, color=viz.PALETTE[1]); axes[1].set_title("slice spacing (mm)")
        axes[2].hist(acq.inplane_mm, bins=30, color=viz.PALETTE[2]); axes[2].set_title("in-plane pixel size (mm)")
        man = acq.groupby("manufacturer").agg(n=("y", "size"), pe_rate=("y", "mean")).sort_values("n", ascending=False)
        viz.bar_counts(axes[3], man["n"], title="scanner manufacturer (PE rate above bars)", annotate=False)
        for i, (n, r) in enumerate(zip(man["n"], man["pe_rate"])):
            axes[3].text(i, n, f"{r:.0%}", ha="center", va="bottom", fontsize=8)
        axes[3].tick_params(axis="x", rotation=45)
        fig.tight_layout(); viz.save_fig(fig, FIG, NOTEBOOK, "acquisition_variability");
        """),
        md("## C. What a clot looks like"),
        plot_note(
            "Healthy vs PE, side by side",
            "Top row: PE studies at a positive slice (the middle of their longest positive run). Bottom row: negative "
            "studies at the same relative height in the body. Vessel window.",
            "This is the whole task in one picture: a human (or a model) has to tell these two rows apart.",
            "Look inside the bright branching vessels near the center of the chest, next to the heart, for darker "
            "gray spots or a vessel that looks cut off. Negative studies show uniformly bright vessels. Do not be "
            "discouraged if you cannot see the difference: clots can be a few pixels wide.",
        ),
        code("""
        n_pairs = min(4, len(pe_uids), len(neg_uids))
        pe_show = sorted(pe_uids, key=lambda u: -metas[u]["n_pos_slices"])[:n_pairs]
        images, titles = [], []
        rel = []
        for u in pe_show:
            v, m = store.load_volume(u); z = labels.representative_positive_slice(m["slice_labels"])
            images.append(v[z]); titles.append(f"PE  {u}  z={z}"); rel.append(z / v.shape[0])
        for u, r in zip(neg_uids[:n_pairs], rel):
            v, m = store.load_volume(u); z = int(r * (v.shape[0] - 1))
            images.append(v[z]); titles.append(f"negative  {u}  z={z}")
        fig = viz.image_grid(images, titles, ncols=n_pairs, window="vessel")
        viz.save_fig(fig, FIG, NOTEBOOK, "healthy_vs_pe");
        """),
        plot_note(
            "Zoom on the central chest",
            "The same PE slices as above, cropped to the central region where the main lung arteries branch off "
            "from the heart, shown larger. Vessel window.",
            "Clots are small; a full 512-pixel slice hides them. At this zoom the dark spot inside a bright artery is "
            "what a radiologist calls a filling defect.",
            "Bright tubes (arteries full of dye) with a gray or dark patch inside, or a bright tube that ends abruptly. "
            "Compare against the uniformly bright vessels in the healthy row above.",
        ),
        code("""
        crops = [viz.crop_center(img, frac=0.45) for img in images[:n_pairs]]
        fig = viz.image_grid(crops, titles[:n_pairs], ncols=n_pairs, window="vessel", figsize_per=4.2)
        viz.save_fig(fig, FIG, NOTEBOOK, "pe_zoomed_center");
        """),
        plot_note(
            "A clot entering and leaving the frame",
            "Consecutive axial slices of one PE study, from a few slices above the labelled clot to a few below. "
            "Titles mark which slices are labelled positive. Central crop, vessel window.",
            "Shows that a clot is a 3D object: it appears, grows, shrinks and disappears along the z-axis exactly like "
            "an object entering and leaving a video. It also visually confirms that the slice labels were aligned to "
            "the volume correctly, which the heatmap analysis later depends on.",
            "The dark spot inside the artery should be present on the slices marked positive and absent just outside "
            "that range. If the labelled range and the visible clot disagree, the label alignment is wrong.",
        ),
        code("""
        band = labels.positive_band(demo_meta["slice_labels"])
        lo, hi = max(band[0] - 3, 0), min(band[1] + 3, demo_vol.shape[0])
        zs = np.unique(np.linspace(lo, hi - 1, min(8, hi - lo)).astype(int))
        strip = [viz.crop_center(demo_vol[z], frac=0.45) for z in zs]
        titles = [f"z={z}  {'PE' if demo_meta['slice_labels'][z] else '-'}" for z in zs]
        fig = viz.image_grid(strip, titles, ncols=len(zs), window="vessel", figsize_per=2.6, suptitle=f"study {demo_uid}")
        viz.save_fig(fig, FIG, NOTEBOOK, "clot_sequence");
        """),
        plot_note(
            "Central vs peripheral PE",
            "Left: a study labelled *central PE* (clot in the main arteries right next to the heart). Right: a PE study "
            "without the central flag (clot further out in smaller branches). Central crop at a positive slice, vessel window.",
            "These are the easy and the hard end of the problem. Central clots are big and sit in big vessels; peripheral "
            "clots are small objects in thin vessels and are the ones a coarse whole-scan model is expected to miss.",
            "Left: a large dark region inside a wide bright artery. Right: at most a small gray dot in a thin vessel, "
            "possibly not visible at all at this resolution.",
        ),
        code("""
        central = [u for u in pe_uids if metas[u]["central_pe"] == 1]
        peripheral = [u for u in pe_uids if metas[u]["central_pe"] == 0]
        pair, titles = [], []
        for name, group in [("central PE", central), ("peripheral PE", peripheral)]:
            if group:
                u = max(group, key=lambda u: metas[u]["n_pos_slices"]); v, m = store.load_volume(u)
                z = labels.representative_positive_slice(m["slice_labels"])
                pair.append(viz.crop_center(v[z], frac=0.45)); titles.append(f"{name}  {u}  z={z}")
        fig = viz.image_grid(pair, titles, ncols=2, window="vessel", figsize_per=5)
        viz.save_fig(fig, FIG, NOTEBOOK, "central_vs_peripheral");
        """),
        plot_note(
            "Look-alikes",
            "Studies in the EDA subset flagged *flow artifact* or *true filling defect, not PE*, shown at a mid-chest slice, "
            "central crop, vessel window. The caption says which flag applies and whether the study is PE-positive.",
            "These are the cases that fool humans: dye mixing unevenly, or something that blocks the dye but is not a clot. "
            "A model that keys on 'dark spot in bright vessel' will produce false positives on exactly these. They are "
            "the natural hard negatives for a later training stage.",
            "Dark patches inside vessels that are not clots. If the subset has no such study, the cell says so.",
        ),
        code("""
        flagged = [u for u in eda_uids if metas[u]["flow_artifact"] or metas[u]["true_filling_defect_not_pe"]]
        if flagged:
            imgs, titles = [], []
            for u in flagged[:8]:
                v, m = store.load_volume(u); z = labels.representative_positive_slice(m["slice_labels"]) or v.shape[0] // 2
                flags = [f for f in ["flow_artifact", "true_filling_defect_not_pe"] if m[f]]
                imgs.append(viz.crop_center(v[z], frac=0.45)); titles.append(f"{u} y={m['y']}\\n{', '.join(flags)}")
            fig = viz.image_grid(imgs, titles, ncols=4, window="vessel")
            viz.save_fig(fig, FIG, NOTEBOOK, "look_alikes")
        else:
            print("no look-alike flags in the EDA subset")
        """),
        md("""
        ## D. Where along the body clots appear

        The slice labels tell us which slices are positive, but the label file's row order does not
        follow slice position (checked on real studies), so positions can only be computed for
        studies whose volumes we have assembled: the EDA subset. The embedding notebook records the
        aligned labels for every study it processes, so this analysis can be repeated on thousands
        of studies later.
        """),
        plot_note(
            "Where clots sit along the head-to-feet axis",
            "Histogram of the position of every PE-positive slice, normalised so 0 is the top of the scan and 1 the "
            "bottom, for the EDA PE studies; central and non-central studies overlaid.",
            "Tells us which part of the volume the model must attend to, and gives the ground truth for the "
            "'did it look in the right place' check in the heatmap notebook.",
            "A single hump somewhere in the middle (the lung arteries sit around the heart), central clots more "
            "concentrated than peripheral ones, which spread out towards the lung bases.",
        ),
        code("""
        pos_central, pos_other = [], []
        for u in pe_uids:
            lab = np.asarray(metas[u]["slice_labels"]); z = np.where(lab == 1)[0]
            (pos_central if metas[u]["central_pe"] else pos_other).extend(volume.normalized_z(z, len(lab)).tolist())
        fig, ax = plt.subplots(figsize=(8, 3.8))
        ax.hist(pos_other, bins=40, range=(0, 1), alpha=0.6, color=viz.PALETTE[0], label=f"non-central PE ({len(pos_other)} slices)", density=True)
        ax.hist(pos_central, bins=40, range=(0, 1), alpha=0.6, color=viz.PALETTE[3], label=f"central PE ({len(pos_central)} slices)", density=True)
        ax.set_xlabel("position along the scan (0 = top, 1 = bottom)"); ax.set_ylabel("density"); ax.legend()
        viz.save_fig(fig, FIG, NOTEBOOK, "clot_position_distribution");
        """),
        plot_note(
            "Barcode of clot extent",
            "One row per EDA PE study (sorted by number of positive slices), columns are the normalised position along "
            "the scan; a cell is red where that slice is positive.",
            "One picture showing both how much of each scan is affected and where. It is the visual form of the "
            "'positive slices per study' histogram, with position added.",
            "Mostly contiguous red blocks (one clot region) of very different lengths, all roughly at the same height; "
            "scattered or multiple blocks indicate several clots.",
        ),
        code("""
        n_bins = 200
        order = sorted(pe_uids, key=lambda u: metas[u]["n_pos_slices"])
        barcode = np.zeros((len(order), n_bins))
        for i, u in enumerate(order):
            lab = np.asarray(metas[u]["slice_labels"]); idx = (volume.normalized_z(np.arange(len(lab)), len(lab)) * (n_bins - 1)).astype(int)
            np.maximum.at(barcode[i], idx, lab)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(barcode, aspect="auto", cmap=plt.matplotlib.colors.ListedColormap(["white", viz.COLOR_PE]), interpolation="nearest")
        ax.set_xlabel("position along the scan (0 = top, 1 = bottom)"); ax.set_ylabel("PE studies, sorted by clot extent")
        ax.set_xticks([0, n_bins // 2, n_bins - 1]); ax.set_xticklabels(["0", "0.5", "1"])
        viz.save_fig(fig, FIG, NOTEBOOK, "clot_barcode");
        """),
        learned_cell([
            "(fill in after running) how visible the clots were to you at full resolution vs zoomed in.",
            "(fill in after running) whether the labelled positive range matched the visible clot in the sequence strip (label alignment check).",
            "(fill in after running) the spread of slice spacing and in-plane resolution, and whether any manufacturer is a PE-rate outlier.",
            "(fill in after running) where along the scan the clots concentrate and how central differs from peripheral.",
            "The label file's row order does not follow slice position, so z-position analyses need assembled volumes.",
        ]),
    ]
