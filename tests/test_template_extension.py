"""Tests for PSF-wing extension of segmap-truncated templates.

Step 1 covers the encircled-energy helpers in ``utils``. Later steps add tests
for the extension routine, metadata propagation, and the Mode-B regression.
"""

import numpy as np
import pytest

from mophongo.utils import psf_ee_radius_pix, psf_ee_area_pix


def _gaussian_psf(n: int = 81, sigma: float = 3.0) -> np.ndarray:
    """Normalised, centred 2-D isotropic Gaussian PSF."""
    y, x = np.mgrid[0:n, 0:n]
    cy = cx = (n - 1) / 2.0
    psf = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma**2))
    return psf / psf.sum()


@pytest.mark.parametrize("sigma", [2.0, 3.0, 5.0])
@pytest.mark.parametrize("ee", [0.5, 0.8, 0.95])
def test_ee_radius_matches_analytic_gaussian(sigma, ee):
    """For a 2-D Gaussian, EE(r) = 1 - exp(-r^2 / 2 sigma^2)."""
    psf = _gaussian_psf(sigma=sigma)
    r = psf_ee_radius_pix(psf, ee_fraction=ee)
    r_analytic = sigma * np.sqrt(-2.0 * np.log(1.0 - ee))
    assert r == pytest.approx(r_analytic, rel=0.05)


def test_ee_area_is_pi_r_squared():
    psf = _gaussian_psf(sigma=3.0)
    r = psf_ee_radius_pix(psf, 0.95)
    assert psf_ee_area_pix(psf, 0.95) == int(np.ceil(np.pi * r * r))


def test_ee_radius_scales_with_psf_width():
    """Band-independence: the threshold tracks PSF width, not a hard constant."""
    r_narrow = psf_ee_radius_pix(_gaussian_psf(sigma=2.0), 0.95)
    r_wide = psf_ee_radius_pix(_gaussian_psf(sigma=4.0), 0.95)
    assert r_wide == pytest.approx(2.0 * r_narrow, rel=0.05)


def test_ee_radius_uses_signed_psf_and_warns_on_negative(caplog):
    """Curve of growth is computed as-given; negative rings trigger a warning."""
    psf = _gaussian_psf(sigma=3.0)
    psf[0, 0] = -0.5 * psf.max()  # inject a significant negative pixel
    with caplog.at_level("WARNING"):
        psf_ee_radius_pix(psf, 0.95)
    assert any("negative" in rec.message for rec in caplog.records)


def test_ee_radius_rejects_nonpositive_total():
    with pytest.raises(ValueError):
        psf_ee_radius_pix(np.zeros((21, 21)), 0.95)


# --- Step 3: template metadata (n_pix, flag bits, propagation) -------------

from mophongo.templates import Template, Templates


def test_extension_flag_bits_distinct():
    bits = [
        Template.FLAG_VALID,
        Template.FLAG_CONVOLVED,
        Template.FLAG_SUM_ZERO,
        Template.FLAG_HAS_NAN,
        Template.FLAG_OUTSIDE_WEIGHT,
        Template.FLAG_SHIFTED,
        Template.FLAG_PSF_EXTENDED,
        Template.FLAG_EXTEND_FAILED,
    ]
    # each a single distinct bit, no collisions
    assert all(b & (b - 1) == 0 for b in bits)
    assert len(set(bits)) == len(bits)


def test_extract_records_n_pix():
    """n_pix equals the segmap pixel count, independent of cutout size."""
    image = np.zeros((40, 40))
    segmap = np.zeros((40, 40), dtype=int)
    segmap[18:22, 18:21] = 1  # 4 x 3 = 12 segmap pixels
    image[segmap == 1] = 5.0

    tmpls = Templates()
    tmpls.extract_templates(image, segmap, [(19.0, 19.5)])
    assert tmpls._templates[0].n_pix == 12


def test_convolve_propagates_n_pix_and_flags_even_when_sum_zero():
    """n_pix and extension flags survive convolution, including FLAG_SUM_ZERO."""
    tmpl = Template(np.ones((9, 9)), (4, 4), (9, 9), label=1)
    tmpl.n_pix = 7
    tmpl.flag |= Template.FLAG_PSF_EXTENDED

    kernel = np.zeros((5, 5))
    kernel[2, 2] = 1.0
    conv = tmpl.convolve_cutout(kernel)
    assert conv.n_pix == 7
    assert conv.flag & Template.FLAG_PSF_EXTENDED

    # A zero-sum template still carries its area/provenance metadata.
    zero = Template(np.zeros((9, 9)), (4, 4), (9, 9), label=2)
    zero.n_pix = 3
    zero.flag |= Template.FLAG_EXTEND_FAILED
    conv_zero = zero.convolve_cutout(kernel)
    assert conv_zero.flag & Template.FLAG_SUM_ZERO
    assert conv_zero.n_pix == 3
    assert conv_zero.flag & Template.FLAG_EXTEND_FAILED


# --- Step 4: aperture-derived min_size --------------------------------------

from astropy.wcs import WCS


def _simple_wcs(pscale_arcsec: float = 0.04, n: int = 101) -> WCS:
    w = WCS(naxis=2)
    w.wcs.crpix = [n / 2, n / 2]
    w.wcs.cdelt = [-pscale_arcsec / 3600.0, pscale_arcsec / 3600.0]
    w.wcs.crval = [150.0, 2.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


def test_min_size_is_instance_attribute():
    assert Templates().min_size == 8  # default preserves prior behaviour
    assert Templates(min_size=20).min_size == 20


def test_min_size_from_aperture_even_pixel_count():
    w = _simple_wcs(pscale_arcsec=0.04)
    # 0.4" / 0.04"/pix = 10 pix; *1.5 margin = 15 -> rounded up to even = 16
    assert Templates.min_size_from_aperture(0.4, w, margin=1.5) == 16
    # scales with aperture diameter
    assert Templates.min_size_from_aperture(0.8, w, margin=1.5) == 30


# --- Step 8: Pipeline.run integration ---------------------------------------

from utils import make_simple_data  # tests/utils.py helper
from mophongo.pipeline import Pipeline
from mophongo.fit import FitConfig


def _pipeline_inputs():
    images, segmap, catalog, psfs, _truth, _rms = make_simple_data(
        seed=5, nsrc=20, size=81, ndilate=1, peak_snr=5
    )
    wcs = [_simple_wcs(pscale_arcsec=0.1, n=81), _simple_wcs(pscale_arcsec=0.1, n=81)]
    weights = [np.ones_like(images[0]), np.ones_like(images[1])]
    return images, segmap, catalog, psfs, wcs, weights


def test_pipeline_run_sizes_cutouts_when_mode_set():
    """Phase 0: selecting an extension mode caches the detection-PSF growth
    curve and enlarges min_size to hold the EE-cap disk + aperture (the actual
    wing fill is wired in Phase 2)."""
    images, segmap, catalog, psfs, wcs, weights = _pipeline_inputs()
    cfg = FitConfig(
        template_extend_mode="psf",
        aperture_diam=0.3,
        aperture_units="arcsec",
        fit_astrometry_niter=0,
        run_scene_solver=False,
    )
    pl = Pipeline(
        images, segmap, catalog=catalog, psfs=psfs, wcs=wcs,
        kernels=[None, None], weights=weights, config=cfg,
    )
    cat, _resid = pl.run()
    assert len(cat) > 0
    # The PSF/aperture floor bumped min_size above the default 8.
    assert pl.tmpls.min_size > 8
    # Growth curve cached with the standard EE fractions.
    assert pl.detection_psf is not None
    assert set(pl.ee_radii_pix) >= {0.5, 0.95, 0.99}
    assert pl.ee_radii_pix[0.5] < pl.ee_radii_pix[0.95] < pl.ee_radii_pix[0.99]
    # Templates remain unit-sum (the apcor invariant the Estimator-3 code needs).
    for t in pl.tmpls._templates:
        if t.data.sum() != 0:
            assert t.data.sum() == pytest.approx(1.0, rel=1e-6)


def test_pipeline_run_no_extension_when_mode_none():
    """With template_extend_mode='none' (default), min_size stays default and
    nothing is extended."""
    images, segmap, catalog, psfs, wcs, weights = _pipeline_inputs()
    cfg = FitConfig(
        template_extend_mode="none",
        fit_astrometry_niter=0,
        run_scene_solver=False,
    )
    pl = Pipeline(
        images, segmap, catalog=catalog, psfs=psfs, wcs=wcs,
        kernels=[None, None], weights=weights, config=cfg,
    )
    pl.run()
    assert pl.tmpls.min_size == 8
    assert not any(t.flag & Template.FLAG_PSF_EXTENDED for t in pl.tmpls._templates)


# --- Extraction-time template extension (data / psf / hybrid) ----------------


def _point_source_scene(total_flux=1000.0, sigma=3.0, n=121, seg_frac=0.5):
    """Gaussian point source with a core-only segmap (truncated-template case)."""
    psf = _gaussian_psf(n=n, sigma=sigma)
    image = total_flux * psf
    segmap = (image > seg_frac * image.max()).astype(int)
    pos = ((n - 1) / 2.0, (n - 1) / 2.0)
    return image, segmap, psf, pos


def _footprint(tm, shape):
    fp = np.zeros(shape, bool)
    fp[tm.slices_original] = tm.data[tm.slices_cutout] != 0
    return fp


def test_data_mode_extends_beyond_segment_and_is_unit_sum():
    """data mode fills real pixels beyond the segment; template stays unit-sum and
    template_norm captures the extended composite (invariant template_norm*H=composite)."""
    image, segmap, psf, pos = _point_source_scene(total_flux=1000.0, sigma=3.0, n=121)
    trunc = Templates(min_size=61)
    trunc.extract_templates(image, segmap, [pos])
    n_trunc = int((trunc._templates[0].data != 0).sum())

    ext = Templates(min_size=61)
    ext.extract_templates(image, segmap, [pos], extend_mode="data",
                          detection_psf=psf, max_radius_pix=20.0)
    t = ext._templates[0]
    assert int((t.data != 0).sum()) > n_trunc          # genuinely extended
    # NOTE: FLAG_PSF_EXTENDED now means "the PSF model actually contributed"
    # (docs/aperture_corrections.md Sec 5.1), not "extension was attempted".
    # This scene is a noiseless point source, so data alone reproduces the PSF
    # shape everywhere (w_core == w_k == 1 throughout) and the flag correctly
    # stays clear; extension is still verified via the pixel-count growth above.
    assert t.data.sum() == pytest.approx(1.0, rel=1e-6)  # unit-sum invariant
    # template_norm should exceed the segmap-only flux (now holds the wings)
    assert t.template_norm > trunc._templates[0].template_norm


def test_psf_mode_wings_follow_the_psf():
    """psf mode: the (unit-sum) extended template is proportional to the PSF."""
    image, segmap, psf, pos = _point_source_scene(total_flux=1000.0, sigma=3.0, n=121)
    ext = Templates(min_size=61)
    ext.extract_templates(image, segmap, [pos], extend_mode="psf",
                          detection_psf=psf, max_radius_pix=20.0)
    t = ext._templates[0]
    # NOTE: FLAG_PSF_EXTENDED means "the PSF model actually contributed"; on
    # this noiseless point source data == PSF shape everywhere, so w == 1 and
    # the flag correctly stays clear (see test_data_mode_... above).
    assert t.data.sum() == pytest.approx(1.0, rel=1e-6)
    # On a noiseless point source the composite must track the PSF: high
    # correlation between the filled template and the PSF over its footprint.
    fp = t.data[t.slices_cutout] != 0
    psf_cut = psf[t.slices_original]
    a = t.data[t.slices_cutout][fp].ravel()
    b = psf_cut[fp].ravel()
    assert np.corrcoef(a, b)[0, 1] > 0.99


def test_ownership_is_self_consistent_tiny_next_to_big():
    """A 1-px segment keeps its own pixel even beside a much larger segment."""
    seg = np.zeros((60, 60), int)
    seg[20:40, 20:40] = 2          # big segment
    seg[30, 19] = 1                # 1-px segment adjacent
    owner = Templates._build_ownership(seg, radius=8.0)
    assert owner[30, 19] == 1      # self-ownership preserved
    assert owner[25, 25] == 2


def test_extended_neighbours_are_disjoint():
    """Two close sources' extended templates never share a pixel (data mode)."""
    n = 81
    yy, xx = np.mgrid[0:n, 0:n]
    def g(cx, cy, s, a): return a * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s ** 2))
    img = g(30, 40, 5.0, 200.0) + g(50, 40, 2.0, 60.0)
    seg = np.zeros((n, n), int)
    seg[(g(30, 40, 5.0, 200.0) > 60) & (xx < 41)] = 1
    seg[(g(50, 40, 2.0, 60.0) > 18) & (xx >= 41)] = 2
    psf = g(40, 40, 3.0, 1.0); psf /= psf.sum()
    t = Templates(min_size=41)
    tmpls = t.extract_templates(img, seg, [(30, 40), (50, 40)], extend_mode="data",
                               detection_psf=psf, max_radius_pix=12.0)
    foot = np.zeros((n, n), int)
    for tm in tmpls:
        fp = _footprint(tm, (n, n))
        assert int((foot & fp).sum()) == 0   # disjoint
        foot |= fp


def test_fill_reaches_max_radius():
    """For an isolated source the fill extends out to ~max_radius_pix."""
    image, segmap, psf, pos = _point_source_scene(total_flux=1000.0, sigma=2.0, n=121)
    R = 18.0
    ext = Templates(min_size=2 * int(np.ceil(R)) + 4)
    ext.extract_templates(image, segmap, [pos], extend_mode="data",
                          detection_psf=psf, max_radius_pix=R)
    t = ext._templates[0]
    fp = t.data[t.slices_cutout] != 0
    ys, xs = np.where(fp)
    cx = t.input_position_cutout[0] - t.slices_cutout[1].start
    cy = t.input_position_cutout[1] - t.slices_cutout[0].start
    rmax = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2).max()
    assert rmax >= R - 1.5   # reaches the requested radius (isolated -> no owner cut)


def test_halo_blend_tracks_data_or_converges_to_psf():
    """Unified blend (docs/aperture_corrections.md Sec 5.1): a real extended
    halo carries enough per-annulus SNR that H tracks the real data bit-exactly;
    a noise-only halo (same bright core) has no SNR to support it, so H
    converges to the (here, negligible at this radius) PSF model instead of
    the noise pattern -- checked via the far annulus' pixel-to-pixel scatter,
    which collapses relative to the raw noise once the PSF model dominates.
    Replaces the old hard ``wings_snr_psf`` branch-forcing test (the three-
    branch decision tree no longer exists)."""
    n = 121
    yy, xx = np.mgrid[0:n, 0:n]
    def g(cx, cy, s, a): return a * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s ** 2))
    psf = g(60, 60, 2.0, 1.0); psf /= psf.sum()
    core = g(60, 60, 2.0, 400.0)
    seg = np.zeros((n, n), int); seg[core > 0.5 * core.max()] = 1
    pos = (60.0, 60.0)
    ivar = np.full((n, n), 1.0)  # sigma = 1
    kw = dict(extend_mode="auto", detection_psf=psf, detection_weight=ivar,
              max_radius_pix=22.0, fit_snrlo_psf=10.0, wings_snr_psf=3.0)

    # Far annulus: well outside both the tiny (sigma=2) segment and the PSF's
    # own reach, so any signal there in the "noise" scene is pure noise.
    r2 = (xx - 60) ** 2 + (yy - 60) ** 2
    annulus = (r2 >= 15 ** 2) & (r2 <= 20 ** 2)

    # Real broad halo present -> high per-annulus SNR -> H tracks data exactly.
    rng = np.random.default_rng(0)
    img_halo = core + g(60, 60, 9.0, 40.0) + rng.normal(0, 1.0, (n, n))
    th = Templates(min_size=61)
    th.extract_templates(img_halo, seg, [pos], **kw)
    t = th._templates[0]
    ann_local = annulus[t.slices_original]
    H = t.data[t.slices_cutout] * t.template_norm
    np.testing.assert_allclose(
        H[ann_local], img_halo[t.slices_original][ann_local], atol=1e-6
    )

    # No real halo: beyond the segment is pure noise -> low per-annulus SNR ->
    # H collapses toward the PSF model (negligible at this radius for a
    # sigma=2 PSF), so its scatter there is far below the raw noise's.
    img_noise = core * (seg == 1) + rng.normal(0, 1.0, (n, n))
    tp = Templates(min_size=61)
    tp.extract_templates(img_noise, seg, [pos], **kw)
    tn = tp._templates[0]
    ann_local_n = annulus[tn.slices_original]
    Hn = tn.data[tn.slices_cutout] * tn.template_norm
    assert np.std(Hn[ann_local_n]) < 0.3 * np.std(img_noise[tn.slices_original][ann_local_n])


# --- Unified-blend weight semantics and robustness ---------------------------

from mophongo.templates import blend_weight


def test_blend_weight_semantics():
    """The config SNR is the ONSET of PSF blending: weight is exactly 1 at and
    above the threshold, rolls off smoothly below, and non-finite SNR defers
    fully to the PSF (never the accidental full-data-trust of unguarded
    min/max ordering)."""
    assert blend_weight(3.0, 3.0, 2.0) == 1.0           # exact threshold -> pure data
    assert blend_weight(10.0, 3.0, 2.0) == 1.0          # above -> pure data
    assert blend_weight(1.5, 3.0, 2.0) == pytest.approx(0.25)  # (1.5/3)^2
    assert blend_weight(0.0, 3.0, 2.0) == 0.0
    assert blend_weight(-5.0, 3.0, 2.0) == 0.0          # negative clamps to 0
    assert blend_weight(float("nan"), 3.0, 2.0) == 0.0  # no measurement -> PSF
    assert blend_weight(float("inf"), 3.0, 2.0) == 1.0
    assert blend_weight(1e300, 3.0, 2.0) == 1.0         # no overflow warning path
    assert blend_weight(1.0, 0.0, 2.0) == 1.0           # disabled threshold -> data
    # monotone non-decreasing in snr
    ws = [blend_weight(s, 3.0, 2.0) for s in np.linspace(0, 6, 25)]
    assert all(b >= a for a, b in zip(ws, ws[1:]))


def _blend_scene(n=121, core_amp=400.0):
    """Noiseless scene scaffold (ivar supplies the noise level, so annulus
    SNRs are deterministic): sigma=2 point core with a core-only segmap."""
    yy, xx = np.mgrid[0:n, 0:n]

    def g(cx, cy, s, a):
        return a * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s ** 2))

    psf = g(60, 60, 2.0, 1.0)
    psf /= psf.sum()
    core = g(60, 60, 2.0, core_amp)
    seg = np.zeros((n, n), int)
    seg[core > 0.5 * core.max()] = 1
    r2 = (xx - 60) ** 2 + (yy - 60) ** 2
    kw = dict(extend_mode="auto", detection_psf=psf,
              detection_weight=np.ones((n, n)),
              max_radius_pix=22.0, fit_snrlo_psf=10.0, wings_snr_psf=3.0)
    return core, seg, r2, kw, g


def _composite(tmpls, full_mask):
    t = tmpls._templates[0]
    H = t.data[t.slices_cutout] * t.template_norm
    return H, full_mask[t.slices_original]


def test_monotone_halo_weights_capped_by_inner_annuli():
    """A high-SNR ring OUTSIDE noise-only annuli cannot re-inflate to data:
    the cumulative-min weight is set by the interior (data trust never
    increases with radius). Standalone, that ring's SNR would give w == 1."""
    core, seg, r2, kw, g = _blend_scene()
    img = core.copy()
    ring = (r2 >= 13 ** 2) & (r2 <= 15 ** 2)
    img[ring] += 50.0   # far outside the sigma=2 core; annuli in between are 0

    tm = Templates(min_size=61)
    tm.extract_templates(img, seg, [(60.0, 60.0)], **kw)
    H, ring_local = _composite(tm, ring)
    # Interior annuli have zero flux -> w == 0 there -> the ring is capped at 0:
    # H on the ring is the (negligible at r~14) PSF model, not the 50-count data.
    assert np.abs(H[ring_local]).max() < 0.5


def test_halo_weight_never_exceeds_core_weight():
    """The halo cumulative min is seeded at w_core: a faint core (snr_seg
    below the onset) caps even an arbitrarily high-SNR halo at w_core. The
    bright halo lives strictly OUTSIDE the segment so it cannot inflate
    snr_seg, and it covers every halo annulus so no zero-flux ring can cap
    the weight below w_core first."""
    core, seg, r2, kw, g = _blend_scene(core_amp=2.75)  # faint core: snr_seg ~ 8
    img = core.copy()
    img[seg == 0] += 40.0   # uniform bright halo: every annulus SNR >> onset

    tm = Templates(min_size=61)
    tm.extract_templates(img, seg, [(60.0, 60.0)], **kw)
    t = tm._templates[0]

    # expected w_core from the actual in-segment sums (ivar == 1)
    snr_seg = t.snr_seg
    assert 0.0 < snr_seg < 15.0
    w_core = min(1.0, (snr_seg / 15.0) ** 2)

    annulus = (r2 >= 10 ** 2) & (r2 <= 14 ** 2)   # PSF model negligible here
    H, ann_local = _composite(tm, annulus)
    data = img[t.slices_original][ann_local]
    ratio = np.median(H[ann_local] / data)
    # halo tracks w_core * data (annulus SNR alone would give w == 1)
    assert ratio == pytest.approx(w_core, rel=0.02)
    assert ratio < 0.9


def test_negative_halo_annulus_blends_to_psf():
    """Positive-pixel clamp: a net-negative annulus has snr == 0 -> w == 0,
    so H there is the PSF model, never the negative data."""
    core, seg, r2, kw, g = _blend_scene()
    img = core.copy()
    ring = (r2 >= 10 ** 2) & (r2 <= 13 ** 2)
    img[ring] = -50.0

    tm = Templates(min_size=61)
    tm.extract_templates(img, seg, [(60.0, 60.0)], **kw)
    H, ring_local = _composite(tm, ring)
    assert np.abs(H[ring_local]).max() < 0.5   # model, not the -50 data
    assert H[ring_local].min() > -1e-6         # nothing negative leaks in


def test_nan_pixels_take_psf_model_and_template_stays_finite():
    """One NaN pixel (core or halo) must neither poison its annulus nor NaN
    the normalized template: excluded from all SNR statistics, and the pixel
    itself takes the PSF model. Regression: pre-guard, 0*NaN == NaN made the
    unit-sum normalization turn the ENTIRE template NaN."""
    core, seg, r2, kw, g = _blend_scene()
    img = core + g(60, 60, 9.0, 40.0)          # bright core + real halo
    img[60, 62] = np.nan                        # in-segment pixel
    img[60, 72] = np.nan                        # halo pixel (r = 12)

    tm = Templates(min_size=61)
    tm.extract_templates(img, seg, [(60.0, 60.0)], **kw)
    t = tm._templates[0]
    assert np.all(np.isfinite(t.data))

    H = t.data[t.slices_cutout] * t.template_norm
    loc = np.zeros_like(img, dtype=bool)
    loc[60, 62] = True
    nan_core_local = loc[t.slices_original]
    # the NaN core pixel took the (positive) PSF model...
    assert H[nan_core_local][0] > 0
    # ...while the rest of the (high-SNR, w_core == 1) segment is exact data
    seg_local = (seg == 1)[t.slices_original] & ~nan_core_local
    np.testing.assert_allclose(H[seg_local], img[t.slices_original][seg_local],
                               atol=1e-5)


def test_detection_psf_region_lookup_uses_source_sky_position():
    """Regression: cut.wcs is the CRPIX-shifted cutout-frame WCS, so region
    lookups (PSF shape AND containment) must feed it input_position_cutout.
    Feeding position_original lands ~(cutout corner)*pscale away on the sky —
    here that puts the source in the wrong region, whose PSF is 2.5x wider
    and whose containment is 1.0 instead of 0.8."""
    import geopandas as gpd
    import shapely.geometry as sgeom
    from mophongo.psf_map import PSFRegionMap

    n = 101
    w = _simple_wcs(pscale_arcsec=0.04, n=n)
    src_xy = (75.0, 75.0)   # far from CRPIX so the frame-mixing offset is large
    ra0, dec0 = w.wcs_pix2world(*src_xy, 0)

    def _gauss_n(nn, sigma):
        y, x = np.mgrid[0:nn, 0:nn]
        c = (nn - 1) / 2.0
        p = np.exp(-((x - c) ** 2 + (y - c) ** 2) / (2 * sigma ** 2))
        return p / p.sum()

    # Region A: 1-arcsec box on the source's TRUE sky position (narrow PSF,
    # containment 0.8). Region B: everything else (wide PSF, containment 1.0).
    d = 1.0 / 3600.0
    box_a = sgeom.box(ra0 - d, dec0 - d, ra0 + d, dec0 + d)
    box_b = sgeom.box(ra0 - 1.0, dec0 - 1.0, ra0 + 1.0, dec0 + 1.0).difference(box_a)
    regions = gpd.GeoDataFrame({"psf_key": [0, 1]}, geometry=[box_a, box_b], crs=None)
    prm = PSFRegionMap(
        regions=regions,
        psfs=np.array([_gauss_n(31, 2.0), _gauss_n(31, 5.0)]),
        containment=np.array([0.8, 1.0]),
    )

    # Faint scene (net-negative segment -> w == 0 -> H is exactly the PSF
    # model of whichever region gets resolved).
    image = np.zeros((n, n))
    segmap = np.zeros((n, n), dtype=int)
    segmap[74:77, 74:77] = 1
    image[74:77, 74:77] = -1.0
    image[75, 75] = 3.0

    tmpls = Templates(min_size=41)
    tmpls.extract_templates(
        image, segmap, [src_xy], wcs=w, extend_mode="auto",
        detection_psf=prm, detection_weight=np.ones((n, n)),
        max_radius_pix=18.0, psf_ee_radius_pix=18.0,
        fit_snrlo_psf=10.0, wings_snr_psf=3.0,
    )
    t = tmpls._templates[0]

    # Shape check (_lookup_detection_psf site): flux-weighted <r^2> is ~2*sigma^2
    # = 8 for the true region's sigma=2 PSF vs ~50 for the wrong sigma=5 one.
    H = t.data[t.slices_cutout] * t.template_norm
    hh, ww_ = H.shape
    cy = t.input_position_cutout[1] - t.slices_cutout[0].start
    cx = t.input_position_cutout[0] - t.slices_cutout[1].start
    yy, xx = np.mgrid[0:hh, 0:ww_]
    rr2 = (xx - cx) ** 2 + (yy - cy) ** 2
    r2_mean = float((H * rr2).sum() / H.sum())
    assert r2_mean < 20.0   # sigma=2 region PSF (wrong region would give ~50)

    # Containment check (flux_beyond_stamp site): with c_det = 0.8 and the
    # reach covering essentially all of the PSF (f_cut ~ 1), beyond/norm =
    # (1/0.8 - f_cut)/f_cut ~ 0.25; the wrong region's c = 1.0 gives ~ 0.
    ratio = t.flux_beyond_stamp / t.template_norm
    assert ratio == pytest.approx(0.25, rel=0.10)


def test_extend_none_leaves_truncated_templates():
    image, segmap, psf, pos = _point_source_scene(sigma=3.0, n=121)
    a = Templates(min_size=61)
    a.extract_templates(image, segmap, [pos], extend_mode="none")
    b = Templates(min_size=61)
    b.extract_templates(image, segmap, [pos])  # default extend handled by Pipeline, not here
    np.testing.assert_array_equal(a._templates[0].data, b._templates[0].data)
    assert not (a._templates[0].flag & Template.FLAG_PSF_EXTENDED)


def test_from_image_passes_extend_kwargs():
    """extend kwargs reach ``_extended_composite`` through ``from_image`` --
    checked via a real side effect (``snr_seg`` is only computed when the
    composite is built), since FLAG_PSF_EXTENDED now means "the PSF model
    actually contributed" rather than "extension was attempted"."""
    image, segmap, psf, pos = _point_source_scene(total_flux=1000.0, sigma=3.0, n=121)
    tmpls = Templates.from_image(image, segmap, [pos], min_size=61,
                                 extend_mode="data", detection_psf=psf, max_radius_pix=20.0)
    assert np.isfinite(tmpls._templates[0].snr_seg)
    plain = Templates.from_image(image, segmap, [pos], min_size=61)
    assert not np.isfinite(plain._templates[0].snr_seg)


