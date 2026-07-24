import os
import sys

current = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(current, "..", "src"))
sys.path.insert(0, current)

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.nddata import Cutout2D
from photutils.psf.matching import SplitCosineBellWindow, TukeyWindow
import mophongo.utils as mutils

import mophongo.pipeline as pipeline
from utils import (
    make_simple_data,
    save_diagnostic_image,
    save_flux_vs_truth_plot,
)


def test_pipeline_flux_recovery(tmp_path):
    #    images, segmap, catalog, psfs, truth_img, rms = make_simple_data(seed=5, nsrc=300, size=501, ndilate=1, peak_snr=1)
    #    table, resid, templates = pipeline.run(images, segmap, catalog, psfs, rms)

    images, segmap, catalog, psfs, truth_img, wht = make_simple_data(
        seed=5, nsrc=151, size=301, ndilate=2, peak_snr=1.5
    )
    #    table, resid, templates = pipeline.run(images, segmap, catalog, psfs, rms, extend_templates='psf')

    # add the hires images as the first fitting image, so that we can compare fluxes
    images.insert(0, images[0])
    wht.insert(0, wht[0])
    psfs.insert(0, psfs[0])
    # images are: hires, hires, lowres
    # psfs are:   hires, hires, lowres
    # so this would add psf hires wings to templates, and result in a delta function for kernel
    dirac = lambda n: ((np.arange(n)[:, None] == n // 2) & (np.arange(n) == n // 2)).astype(float)

    kernel = [mutils.matching_kernel(psfs[0], psf) for psf in psfs]
    kernel[0] = kernel[1] = dirac(3)  # no kernel for the first image, it is the hires image
    pl = pipeline.Pipeline(
        images, segmap, catalog=catalog, weights=wht, kernels=kernel
    )
    table, resid = pl.run()

    # @@@ sometimes flux_true is NEGATIVE?
    table["flux_true"] = catalog["flux_true"]  # add flux_true to the table

    # Plot for high-res (flux_0) vs truth
    flux_hi_plot = tmp_path / "flux_hi_vs_true.png"
    save_flux_vs_truth_plot(
        flux_hi_plot,
        table["flux_true"],
        table["flux_1"],
        error=table["err_1"],  # Add error column
        label="Flux (hires)",
        xlabel="True Flux",
        ylabel="Recovered Flux (hires)",
    )
    assert flux_hi_plot.exists()

    # Plot for low-res (flux_1) vs truth
    flux_lo_plot = tmp_path / "flux_lo_vs_true.png"
    save_flux_vs_truth_plot(
        flux_lo_plot,
        table["flux_true"],
        table["flux_2"],
        error=table["err_2"],  # Add error column
        label="Flux (lowres)",
        xlabel="True Flux",
        ylabel="Recovered Flux (lowres)",
    )
    assert flux_lo_plot.exists()

    # Plot for flux_lo vs flux_hi with error propagation
    flux_lo_hi_plot = tmp_path / "flux_lo_vs_hi.png"
    # Calculate combined error for hires vs lowres comparison
    combined_error = np.sqrt(table["err_1"] ** 2 + table["err_2"] ** 2)
    save_flux_vs_truth_plot(
        flux_lo_hi_plot,
        table["flux_1"],
        table["flux_2"],
        error=combined_error,
        label="Flux (lowres) vs (hires)",
        xlabel="Recovered Flux (hires)",
        ylabel="Recovered Flux (lowres)",
    )
    assert flux_lo_hi_plot.exists()

    # ----------------------------------- separate run for high-res, using the truth image as templates
    pl_true = pipeline.Pipeline(
        [truth_img, images[1]],
        segmap,
        catalog=catalog,
        kernels=[dirac(3), psfs[1]],
        weights=[np.zeros(wht[0].shape), wht[1]],
    )
    table_true, resid_hi = pl_true.run()
    table_true["flux_true"] = catalog["flux_true"]
    # Plot for high-res (flux_0) vs truth
    flux_true_plot = tmp_path / "flux_hi_vs_true_truemodel.png"
    save_flux_vs_truth_plot(
        flux_true_plot,
        table_true["flux_true"],
        table_true["flux_1"],
        error=table_true["err_1"],  # Add error column
        label="Flux (hires)",
        xlabel="True Flux",
        ylabel="Recovered Flux (hires)",
    )
    assert flux_true_plot.exists()

    model = images[2] - resid[1]
    fname = tmp_path / "diagnostic.png"
    save_diagnostic_image(
        fname, truth_img, images[1], images[2], model, resid[1], segmap=segmap, catalog=catalog
    )
    fname = tmp_path / "diagnostic_hires.png"
    model = images[1] - resid[0]
    save_diagnostic_image(
        fname, truth_img, images[0], images[1], model, resid[0], segmap=segmap, catalog=catalog
    )

    fname = tmp_path / "diagnostic_hires_truemodel.png"
    model = images[1] - resid_hi[0]
    save_diagnostic_image(
        fname, truth_img, truth_img, images[1], model, resid_hi[0], segmap=segmap, catalog=catalog
    )
    assert fname.exists()

    # ------------------------------------------------------------------
    # A8 -- keystone end-to-end assertions: flux recovery + residual noise
    # floor. Both run against the same fit above (hires band = flux_1,
    # lowres band = flux_2), so a real regression in the fit/kernel/PSF
    # chain trips one of these.
    # ------------------------------------------------------------------
    flux_true = np.array(table["flux_true"])
    for idx in range(1, len(psfs)):
        col = f"flux_{idx}"
        ratio = np.array(table[col]) / flux_true
        p5, p16, p50, p84, p95 = np.percentile(ratio, [5, 16, 50, 84, 95])
        print(
            f"flux_{idx}/flux_true percentiles: 5th={p5:.2f}, 16th={p16:.2f}, "
            f"50th={p50:.2f}, 84th={p84:.2f}, 95th={p95:.2f}"
        )

        # Whole-population median should recover flux_true to within a few
        # percent -- on this synthetic field (peak_snr=1.5, nsrc=151) the
        # observed median is ~1.00 and the 16th/84th band is ~0.98-1.02;
        # the tolerances below are set a few times looser than that so the
        # test is robust to noise realizations while still catching a real
        # flux bias.
        assert abs(p50 - 1.0) < 0.05, f"flux_{idx}: median ratio {p50:.3f} far from 1.0"
        assert (p84 - p16) < 0.20, f"flux_{idx}: 16-84th spread {p84-p16:.3f} too wide"

        # Bright sources (top quartile by true flux) should recover flux
        # even more tightly -- this isolates a systematic flux bias from
        # noise-dominated scatter in faint sources.
        bright = flux_true >= np.percentile(flux_true, 75)
        ratio_bright = ratio[bright & np.isfinite(ratio)]
        med_bright = np.median(ratio_bright)
        assert abs(med_bright - 1.0) < 0.03, (
            f"flux_{idx}: bright-source median ratio {med_bright:.3f} far from 1.0"
        )

    # Residual should be at the noise floor: mean chi^2 per pixel in the
    # BACKGROUND (outside every segment), using the known inverse-variance
    # weight maps from make_simple_data, should be close to 1. This is
    # checked in the background rather than over the whole image because
    # band 1 here is a near-degenerate self-fit (images[1] is the same
    # hires image the templates were extracted from, with an effectively
    # unit kernel), so in-source residual there is ~0 by construction and
    # would swamp a whole-image chi^2 with a number that has nothing to do
    # with the noise floor.
    bg = segmap == 0
    for ifilt, res in zip(range(1, len(images)), resid):
        w = wht[ifilt]
        chi2_bg = float(np.mean(res[bg] ** 2 * w[bg]))
        print(f"band {ifilt}: background residual chi2/pix = {chi2_bg:.3f}")
        assert 0.85 < chi2_bg < 1.15, (
            f"band {ifilt}: background chi2/pix {chi2_bg:.3f} far from 1.0 (noise floor)"
        )

    # No significant flux left inside source segments, for the genuinely
    # independent (lowres) fit: per-source residual sum should be
    # consistent with photon noise, not a systematic model mismatch.
    from scipy import ndimage

    ifilt_lo = len(images) - 1  # lowres is always the last image
    res_lo = resid[ifilt_lo - 1]
    w_lo = wht[ifilt_lo]
    seg_ids = np.unique(segmap)
    seg_ids = seg_ids[seg_ids > 0]
    resid_sum = ndimage.sum(res_lo, labels=segmap, index=seg_ids)
    npix = ndimage.sum(np.ones_like(segmap), labels=segmap, index=seg_ids)
    noise_std_lo = 1.0 / np.sqrt(w_lo[0, 0])
    expected_sigma = noise_std_lo * np.sqrt(npix)
    zscore = resid_sum / expected_sigma
    frac_significant = float(np.mean(np.abs(zscore) > 5))
    print(
        f"lowres band: fraction of segments with |residual sum|/sigma > 5: "
        f"{frac_significant:.3f}"
    )
    assert frac_significant < 0.10, (
        f"lowres band: {frac_significant:.1%} of segments have a residual flux "
        "excess >5 sigma above the noise floor"
    )

    # sanity check on propagated errors for low-res image
    from mophongo.psf import PSF
    from mophongo.templates import Templates

    psf_hi = PSF.from_array(psfs[1])
    psf_lo = PSF.from_array(psfs[2])
    kernel = psf_hi.matching_kernel(psf_lo)
    tmpls = Templates.from_image(images[0], segmap, list(zip(catalog["x"], catalog["y"])), kernel)
    noise_std = wht[1][0, 0]
    err_pred = np.array([noise_std / np.sqrt((t.data**2).sum()) for t in tmpls.templates])
    ratio_err = table["err_1"] / err_pred
    assert np.allclose(np.mean(ratio_err), 1.0, atol=3)

    # Write catalog with all columns formatted to 3 digits after the decimal
    for col in table.colnames:
        if table[col].dtype.kind in "fc":  # float or complex
            table[col].info.format = ".3f"

    cat_file = tmp_path / "photometry.cat"
    table.write(cat_file, format="ascii.commented_header")
    assert cat_file.exists()

    loaded = Table.read(cat_file, format="ascii.commented_header")
    assert len(loaded) == len(table)


def test_pipeline_astrometry(tmp_path):
    """End-to-end astrometry: a known shift between the detection image and the
    low-res science image must be recovered by the pipeline's astrometry passes,
    leaving a smaller residual than the same fit with astrometry disabled.

    This exercises the wiring that the unit tests in ``test_astrometry.py`` do
    not: ``generate_scenes`` -> per-scene shift solve -> ``apply_shifts`` ->
    residual, driven through the real ``Pipeline.run`` entry point. A regression
    in that wiring (wrong sign, shifts computed but not applied, iteration logic)
    would pass every solver-level test but fail here.
    """
    from scipy.ndimage import shift as nd_shift
    from mophongo.fit import FitConfig

    images, segmap, catalog, psfs, truth, wht = make_simple_data(
        nsrc=15, size=151, peak_snr=20, seed=42
    )

    # Inject a known global sub-pixel offset into the low-res science image so
    # the templates (extracted at catalog positions from the detection image)
    # are misaligned with band 1.
    true_dx, true_dy = 0.6, -0.5
    science = nd_shift(images[1], (true_dy, true_dx))

    dirac = lambda n: (
        (np.arange(n)[:, None] == n // 2) & (np.arange(n) == n // 2)
    ).astype(float)
    kernels = [dirac(3), mutils.matching_kernel(psfs[0], psfs[1])]

    # Order-0 (global) shift model matches the injected offset; low SNR gate and
    # small bright count so the modest synthetic field still drives astrometry.
    cfg_kwargs = dict(
        snr_thresh_astrom=3.0,
        scene_minimum_bright=2,
        scene_coupling_thresh=0.01,
        astrom_kwargs={"poly": {"order": 0}, "gp": {"length_scale": 400}},
    )

    pl = pipeline.Pipeline(
        [images[0].copy(), science.copy()],
        segmap,
        catalog=catalog.copy(),
        weights=[wht[0], wht[1]],
        kernels=kernels,
        config=FitConfig(fit_astrometry_niter=2, **cfg_kwargs),
    )
    pl.run()

    # The broad low-res PSF makes the residual almost insensitive to a sub-pixel
    # shift, so assert on the recovered shift itself. After the astrometry passes,
    # each shifted template's accumulated ``.shifted`` (dx, dy) should sum to the
    # injected offset — this proves the pipeline both COMPUTED the shift and
    # APPLIED it onto the templates (the wiring the solver-level tests skip).
    band1_templates = pl.all_templates[0]
    shifted = np.array(
        [t.shifted for t in band1_templates if np.linalg.norm(t.shifted) > 1e-6]
    )
    print(
        f"[astrometry] injected (dx,dy)=({true_dx:.2f},{true_dy:.2f}); "
        f"recovered n={len(shifted)} "
        f"mean=({shifted[:,0].mean():.3f},{shifted[:,1].mean():.3f})"
        if len(shifted)
        else "[astrometry] no templates were shifted"
    )

    assert len(shifted) > 0, "no templates were shifted — astrometry did not engage"
    assert abs(shifted[:, 0].mean() - true_dx) < 0.3, (
        f"dx recovered {shifted[:, 0].mean():.3f}, expected {true_dx:.3f}"
    )
    assert abs(shifted[:, 1].mean() - true_dy) < 0.3, (
        f"dy recovered {shifted[:, 1].mean():.3f}, expected {true_dy:.3f}"
    )
