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
    assert t.flag & Template.FLAG_PSF_EXTENDED
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
    assert t.flag & Template.FLAG_PSF_EXTENDED
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


def test_auto_routes_by_wing_signal():
    """auto tree: a real extended halo -> data wings; a noise-only halo -> psf wings.

    The branch is forced for the reference via ``wings_snr_psf`` (negative => any
    wing SNR counts as extended -> data; huge => no wing SNR qualifies -> psf), and
    the auto choice (``wings_snr_psf=3``) must match the corresponding reference.
    """
    n = 121
    yy, xx = np.mgrid[0:n, 0:n]
    def g(cx, cy, s, a): return a * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s ** 2))
    psf = g(60, 60, 2.0, 1.0); psf /= psf.sum()
    core = g(60, 60, 2.0, 400.0)
    seg = np.zeros((n, n), int); seg[core > 0.5 * core.max()] = 1
    pos = (60.0, 60.0)
    ivar = np.full((n, n), 1.0)  # sigma = 1
    kw = dict(extend_mode="auto", detection_psf=psf, detection_weight=ivar, max_radius_pix=22.0)

    # Real broad halo present -> high wing SNR -> data extension.
    rng = np.random.default_rng(0)
    img_halo = core + g(60, 60, 9.0, 40.0) + rng.normal(0, 1.0, (n, n))
    th = Templates(min_size=61)
    th.extract_templates(img_halo, seg, [pos], wings_snr_psf=3.0, **kw)
    td = Templates(min_size=61)  # force data branch
    td.extract_templates(img_halo, seg, [pos], wings_snr_psf=-1.0, **kw)
    assert np.allclose(th._templates[0].data, td._templates[0].data)  # chose data

    # No real halo: beyond the segment is pure noise -> low wing SNR -> psf wings.
    img_noise = core * (seg == 1) + rng.normal(0, 1.0, (n, n))
    th2 = Templates(min_size=61)
    th2.extract_templates(img_noise, seg, [pos], wings_snr_psf=3.0, **kw)
    tp = Templates(min_size=61)  # force psf branch
    tp.extract_templates(img_noise, seg, [pos], wings_snr_psf=1e9, **kw)
    assert np.allclose(th2._templates[0].data, tp._templates[0].data)  # chose psf


def test_extend_none_leaves_truncated_templates():
    image, segmap, psf, pos = _point_source_scene(sigma=3.0, n=121)
    a = Templates(min_size=61)
    a.extract_templates(image, segmap, [pos], extend_mode="none")
    b = Templates(min_size=61)
    b.extract_templates(image, segmap, [pos])  # default extend handled by Pipeline, not here
    np.testing.assert_array_equal(a._templates[0].data, b._templates[0].data)
    assert not (a._templates[0].flag & Template.FLAG_PSF_EXTENDED)


def test_from_image_passes_extend_kwargs():
    image, segmap, psf, pos = _point_source_scene(total_flux=1000.0, sigma=3.0, n=121)
    tmpls = Templates.from_image(image, segmap, [pos], min_size=61,
                                 extend_mode="data", detection_psf=psf, max_radius_pix=20.0)
    assert tmpls._templates[0].flag & Template.FLAG_PSF_EXTENDED
    plain = Templates.from_image(image, segmap, [pos], min_size=61)
    assert not (plain._templates[0].flag & Template.FLAG_PSF_EXTENDED)


