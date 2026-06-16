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


def test_pipeline_run_extends_when_enabled():
    """Step 8: extension is wired into Pipeline.run and sizes the cutouts."""
    images, segmap, catalog, psfs, wcs, weights = _pipeline_inputs()
    cfg = FitConfig(
        extend_template_segmap=True,
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
    # The PSF floor bumped min_size above the default 8.
    assert pl.tmpls.min_size > 8
    for t in pl.tmpls._templates:
        # Every flagged template is consistent (extended xor failed, never both).
        assert not (
            (t.flag & Template.FLAG_PSF_EXTENDED)
            and (t.flag & Template.FLAG_EXTEND_FAILED)
        )
        # Invariant the apcor code relies on: hires templates stay unit-sum even
        # after wing extension (the raised flux_f444w carries the scale).
        if t.data.sum() != 0:
            assert t.data.sum() == pytest.approx(1.0, rel=1e-6)


def test_pipeline_run_no_extension_when_disabled():
    """With the switch off, nothing is extended and min_size stays default."""
    images, segmap, catalog, psfs, wcs, weights = _pipeline_inputs()
    cfg = FitConfig(
        extend_template_segmap=False,
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


# --- Step 5: PSF-wing extension ---------------------------------------------


def _point_source_scene(total_flux=1000.0, sigma=3.0, n=121, seg_frac=0.5):
    """Image of a single Gaussian point source and a core-only segmap.

    The segmap thresholds the PSF at ``seg_frac`` of the peak, so it captures
    only the bright core (small n_pix), reproducing the truncated-template
    failure mode.
    """
    psf = _gaussian_psf(n=n, sigma=sigma)  # sums to 1
    image = total_flux * psf
    segmap = (image > seg_frac * image.max()).astype(int)  # label 1 core
    pos = ((n - 1) / 2.0, (n - 1) / 2.0)
    return image, segmap, psf, pos


def test_extension_raises_flux_f444w_to_inferred_total():
    """flux_f444w should relax to ~the true total after extension (Blocker B2)."""
    total_flux = 1000.0
    image, segmap, psf, pos = _point_source_scene(total_flux=total_flux, sigma=3.0)

    tmpls = Templates(min_size=40)  # room for the wings
    tmpls.extract_templates(image, segmap, [pos])
    t = tmpls._templates[0]
    seg_flux = t.flux_f444w
    assert seg_flux < 0.8 * total_flux  # truncated: segmap holds only the core

    tmpls.extend_with_psf_wings(psf, target_ee=0.95, inplace=True)
    assert t.flag & Template.FLAG_PSF_EXTENDED
    # inferred total ~ the true source flux
    assert t.flux_f444w == pytest.approx(total_flux, rel=0.05)
    # flux was actually pasted outside the original segment
    assert (t.data[~(t.data == 0)]).size > 0
    # Invariant: the template stays unit-sum (aperture code reads aper(T) as a
    # *fraction*); the raised flux_f444w carries the absolute scale.
    assert t.data.sum() == pytest.approx(1.0, rel=1e-6)


def test_extension_makes_template_psf_shaped():
    """After extension the (unit-summed) core matches a clean PSF cutout."""
    image, segmap, psf, pos = _point_source_scene(sigma=3.0, n=121)
    tmpls = Templates(min_size=40)
    tmpls.extract_templates(image, segmap, [pos])
    t = tmpls._templates[0]

    before_outside = float(t.data[t.data == 0].size)
    tmpls.extend_with_psf_wings(psf, target_ee=0.95, inplace=True)
    # fewer zero pixels: wings now fill part of the previously-empty region
    after_zeros = float((t.data == 0).sum())
    assert after_zeros < before_outside


def test_extension_skips_adequate_segmap():
    """A source whose segmap already exceeds ee_area is left untouched."""
    # Low threshold -> segmap captures most of the PSF (large n_pix >= ee_area)
    image, segmap, psf, pos = _point_source_scene(sigma=3.0, n=121, seg_frac=0.001)
    tmpls = Templates(min_size=40)
    tmpls.extract_templates(image, segmap, [pos])
    t = tmpls._templates[0]
    ee_area = psf_ee_area_pix(psf, 0.95)
    assert t.n_pix >= ee_area

    before = t.data.copy()
    flux_before = t.flux_f444w
    tmpls.extend_with_psf_wings(psf, target_ee=0.95, inplace=True)
    assert not (t.flag & Template.FLAG_PSF_EXTENDED)
    assert t.flux_f444w == flux_before
    np.testing.assert_array_equal(t.data, before)


def test_extension_skips_empty_template():
    """n_pix == 0 (e.g. FLAG_SUM_ZERO) is never extended."""
    t = Template(np.zeros((40, 40)), (20, 20), (40, 40), label=1)
    t.n_pix = 0
    tmpls = Templates()
    tmpls._templates = [t]
    tmpls.extend_with_psf_wings(_gaussian_psf(sigma=3.0), inplace=True)
    assert not (t.flag & Template.FLAG_PSF_EXTENDED)


def test_extension_zero_overlap_guard_sets_failed_flag():
    """If the PSF has negligible overlap with the segment, flag and skip."""
    n = 41
    data = np.zeros((n, n))
    data[0:3, 0:3] = 1.0  # segment in the corner

    # Source position at the centre, far from the corner segment; a narrow PSF
    # sampled at the centre is ~0 over the corner -> f_seg ~ 0.
    t = Template(data, (20, 20), (n, n), label=1)
    t.n_pix = 9
    tmpls = Templates()
    tmpls._templates = [t]

    narrow_psf = _gaussian_psf(n=41, sigma=1.0)
    ee_area = psf_ee_area_pix(narrow_psf, 0.95)
    assert 0 < t.n_pix < ee_area  # passes the size gate, so the guard is exercised

    tmpls.extend_with_psf_wings(narrow_psf, target_ee=0.95, inplace=True)
    assert t.flag & Template.FLAG_EXTEND_FAILED
    assert not (t.flag & Template.FLAG_PSF_EXTENDED)


def test_extension_is_idempotent():
    """A second call must not re-inflate flux_f444w (guarded by the flag)."""
    image, segmap, psf, pos = _point_source_scene(total_flux=1000.0, sigma=3.0)
    tmpls = Templates(min_size=40)
    tmpls.extract_templates(image, segmap, [pos])
    t = tmpls._templates[0]

    tmpls.extend_with_psf_wings(psf, target_ee=0.95, inplace=True)
    flux_once = t.flux_f444w
    data_once = t.data.copy()
    assert t.flag & Template.FLAG_PSF_EXTENDED

    tmpls.extend_with_psf_wings(psf, target_ee=0.95, inplace=True)
    assert t.flux_f444w == flux_once
    np.testing.assert_array_equal(t.data, data_once)


def test_extension_fails_when_cutout_too_small_for_wings():
    """If the EE disk does not fit in the cutout, fail instead of clipping."""
    # Default min_size=8 -> ~8px cutout, far smaller than the sigma=3 EE radius.
    image, segmap, psf, pos = _point_source_scene(total_flux=1000.0, sigma=3.0)
    tmpls = Templates()  # min_size=8
    tmpls.extract_templates(image, segmap, [pos])
    t = tmpls._templates[0]
    ee_area = psf_ee_area_pix(psf, 0.95)
    assert 0 < t.n_pix < ee_area  # passes the size gate
    flux_before = t.flux_f444w

    tmpls.extend_with_psf_wings(psf, target_ee=0.95, inplace=True)
    assert t.flag & Template.FLAG_EXTEND_FAILED
    assert not (t.flag & Template.FLAG_PSF_EXTENDED)
    assert t.flux_f444w == flux_before  # denominator untouched on failure


def test_from_image_wires_extension():
    """Step 6: from_image(extension=psf, ...) extends truncated templates."""
    image, segmap, psf, pos = _point_source_scene(total_flux=1000.0, sigma=3.0)
    tmpls = Templates.from_image(
        image, segmap, [pos], extension=psf, target_ee=0.95, min_size=40
    )
    t = tmpls._templates[0]
    assert t.flag & Template.FLAG_PSF_EXTENDED
    assert t.flux_f444w == pytest.approx(1000.0, rel=0.05)
    # Without extension the dead parameter path stays a no-op
    plain = Templates.from_image(image, segmap, [pos], min_size=40)
    assert not (plain._templates[0].flag & Template.FLAG_PSF_EXTENDED)


def test_extension_inplace_false_preserves_originals():
    image, segmap, psf, pos = _point_source_scene(sigma=3.0, n=121)
    tmpls = Templates(min_size=40)
    tmpls.extract_templates(image, segmap, [pos])
    orig = tmpls._templates[0]
    orig_flux = orig.flux_f444w
    orig_data = orig.data.copy()

    out = tmpls.extend_with_psf_wings(psf, target_ee=0.95, inplace=False)
    assert out[0] is not orig
    # originals untouched
    assert orig.flux_f444w == orig_flux
    np.testing.assert_array_equal(orig.data, orig_data)
    # the returned copy was extended
    assert out[0].flag & Template.FLAG_PSF_EXTENDED
