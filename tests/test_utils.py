"""Tests for the cross-cutting numerical helpers in ``mophongo.utils``.

Covers the encircled-energy round trip, the WCS pixel-scale binning-factor
detection, and PSF-stamp containment (docs/test_suite_cleanup_plan.md, A0).
"""

from __future__ import annotations

import numpy as np
import pytest
from astropy.wcs import WCS

from mophongo.utils import (
    bin_factor_from_wcs,
    gaussian,
    moffat,
    psf_ee_at_radius,
    psf_ee_radius_pix,
    psf_stamp_containment,
    rebin_wcs,
)


def _make_wcs(pscale_arcsec: float) -> WCS:
    """Return a minimal tangent-plane WCS with a known pixel scale."""
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [150.0, 2.0]
    w.wcs.crpix = [50, 50]
    w.wcs.cdelt = [-pscale_arcsec / 3600.0, pscale_arcsec / 3600.0]
    return w


@pytest.mark.parametrize("fraction", [0.5, 0.8, 0.95])
@pytest.mark.parametrize(
    "psf",
    [
        gaussian(101, fwhm=6.0, flux=1.0),
        moffat(101, 6.0, 6.0, beta=3.0),
    ],
    ids=["gaussian", "moffat"],
)
def test_psf_ee_round_trip(psf: np.ndarray, fraction: float) -> None:
    """``psf_ee_at_radius(psf, psf_ee_radius_pix(psf, f)) == f``.

    Both functions normalize by the true PSF total (``psf.sum()``), not the
    stamp sum within some aperture, so they must stay mutually consistent --
    everything downstream that talks about enclosed energy depends on it.
    """
    radius = psf_ee_radius_pix(psf, fraction)
    recovered = psf_ee_at_radius(psf, radius)
    assert recovered == pytest.approx(fraction, abs=1e-6)


@pytest.mark.parametrize(
    "s_det,s_img,expected_k",
    [
        (0.02, 0.04, 2),
        (0.02, 0.08, 4),
    ],
)
def test_bin_factor_from_wcs_integer_ratio(s_det, s_img, expected_k) -> None:
    w_det = _make_wcs(s_det)
    w_img = _make_wcs(s_img)
    assert bin_factor_from_wcs(w_det, w_img) == expected_k


def test_bin_factor_from_wcs_raises_on_non_integer_ratio() -> None:
    w_det = _make_wcs(0.02)
    w_img = _make_wcs(0.03)  # ratio = 1.5, not an integer
    with pytest.raises(ValueError):
        bin_factor_from_wcs(w_det, w_img)


def test_psf_stamp_containment_in_unit_interval() -> None:
    psf = gaussian(101, fwhm=6.0, flux=1.0)
    # A stamp much narrower than the PSF support: partial containment.
    frac = psf_stamp_containment(psf, parent_pscale=0.02, stamp_width_arcsec=0.1)
    assert 0.0 < frac <= 1.0

    # A stamp as wide as the full PSF grid: (near) complete containment.
    frac_full = psf_stamp_containment(psf, parent_pscale=0.02, stamp_width_arcsec=1.0)
    assert 0.0 < frac_full <= 1.0
    assert frac_full > frac


@pytest.mark.xfail(
    reason=(
        "rebin_wcs (src/mophongo/utils.py) references an undefined name 'n' "
        "(`factor = 2**n`) instead of its `factor` parameter -- a latent "
        "NameError on any call. Documented here, not fixed (tests-only cleanup)."
    ),
    strict=False,
)
def test_rebin_wcs_latent_name_error() -> None:
    w = _make_wcs(0.02)
    rebin_wcs(w, 2)
