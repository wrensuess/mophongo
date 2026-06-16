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
