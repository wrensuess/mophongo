"""Flux conservation under downsampling (docs/test_suite_cleanup_plan.md, A4).

Replaces the dead ``tests/test_downsample.py``, which targeted a
``mophongo.utils.bin2d_mean`` helper that does not exist and was therefore
never actually run. This exercises the real downsampling API instead:
``Template.downsample`` (templates.py) and ``mophongo.utils.downsample_psf``.
"""

from __future__ import annotations

import numpy as np
import pytest

from mophongo.templates import Template
from mophongo.utils import downsample_psf, gaussian


@pytest.mark.parametrize("k", [2, 3, 4])
def test_template_downsample_conserves_flux(k: int) -> None:
    """A k-fold block reduction must conserve total flux exactly."""
    rng = np.random.default_rng(0)
    ny = nx = 24  # divisible by 2, 3, and 4
    parent = rng.random((ny, nx)).astype(np.float64)

    cy, cx = (ny - 1) / 2, (nx - 1) / 2
    tmpl = Template(parent, (cx, cy), (ny, nx), label=1)

    # The cutout spans the whole parent array, so its origin is (0, 0) and
    # therefore trivially aligned to any k -- no edge pixels get trimmed.
    x0, y0 = map(int, tmpl._origin_original_true)
    assert x0 % k == 0 and y0 % k == 0

    total_hi = float(tmpl.data.sum(dtype=np.float64))
    lo = tmpl.downsample(k)
    total_lo = float(lo.data.sum(dtype=np.float64))

    assert total_lo == pytest.approx(total_hi, rel=1e-10)


@pytest.mark.parametrize("k", [2, 3, 4])
def test_downsample_psf_conserves_flux(k: int) -> None:
    """``downsample_psf`` must conserve the PSF's total (unit) flux."""
    psf = gaussian(41, fwhm=4.0, flux=1.0)
    total_hi = float(psf.sum(dtype=np.float64))

    psf_lo = downsample_psf(psf, k)
    total_lo = float(psf_lo.sum(dtype=np.float64))

    assert total_lo == pytest.approx(total_hi, rel=1e-6, abs=1e-9)
