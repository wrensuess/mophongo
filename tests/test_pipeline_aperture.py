import numpy as np
import pytest
from astropy.table import Table
from mophongo.pipeline import Pipeline
from mophongo.templates import Template


def _make_tmpl(flux_f444w=25.0):
    """Flat 5×5 template; flux_f444w mimics pre-normalisation F444W total."""
    tmpl = Template(np.ones((5, 5)), (2, 2), (10, 10), label=1)
    tmpl.flux_f444w = flux_f444w
    return tmpl


def test_aperture_photometry_estimator3():
    """Without f444w_totals: apcor = ap_F_real / ap_B_real, res_sum = 0."""
    pl = Pipeline([np.zeros((10, 10))], np.zeros((10, 10)))
    pl.psfs = [np.ones((5, 5))]
    cat = Table({"id": [1]})
    conv_tmpl = _make_tmpl()
    orig_tmpl = _make_tmpl()
    residual = np.zeros((10, 10))

    pl._add_aperture_photometry(
        cat, [conv_tmpl], np.array([1.0]), residual, 1,
        r_orig_pix=1.5,
        orig_templates=[orig_tmpl],
    )

    assert "ap_model_1" in cat.colnames
    assert "apcor_1" in cat.colnames
    assert "res_sum_1" in cat.colnames
    assert "ap_flux_corr_1" in cat.colnames
    assert np.isfinite(cat["apcor_1"][0])
    assert np.isfinite(cat["ap_flux_corr_1"][0])
    assert cat["res_sum_1"][0] == pytest.approx(0.0)
    # ap_flux_corr = ap_model * apcor = fl * ap_B_frac * (ap_F_real / ap_B_real)
    ap_model = cat["ap_model_1"][0]
    apcor = cat["apcor_1"][0]
    assert cat["ap_flux_corr_1"][0] == pytest.approx(ap_model * apcor)


def test_aperture_photometry_with_tcor():
    """With f444w_totals: apcor = f444w_total / ap_B_real; ap_flux_corr = ap_model * apcor."""
    flux_f444w_i = 25.0
    fl = 2.0
    f444w_total = 42.0

    pl = Pipeline([np.zeros((10, 10))], np.zeros((10, 10)))
    pl.psfs = [np.ones((5, 5))]
    cat = Table({"id": [1]})
    conv_tmpl = _make_tmpl(flux_f444w_i)
    orig_tmpl = _make_tmpl(flux_f444w_i)
    residual = np.zeros((10, 10))

    pl._add_aperture_photometry(
        cat, [conv_tmpl], np.array([fl]), residual, 1,
        r_orig_pix=1.5,
        orig_templates=[orig_tmpl],
        f444w_totals={1: f444w_total},
    )

    ap_model = cat["ap_model_1"][0]
    apcor = cat["apcor_1"][0]
    # apcor = f444w_total / ap_B_real; ap_flux_corr = ap_model * apcor + res_seg(=0)
    assert cat["ap_flux_corr_1"][0] == pytest.approx(ap_model * apcor)
    # ap_model * apcor = fl * ap_B_frac * f444w_total / (flux_f444w_i * ap_B_frac)
    #                  = fl * f444w_total / flux_f444w_i
    assert cat["ap_flux_corr_1"][0] == pytest.approx(fl * f444w_total / flux_f444w_i)


def test_residual_segmap_sum_same_res():
    """k=1: sum residual only over (segmap == source_id), ignore flux outside."""
    segmap = np.zeros((10, 10), dtype=int)
    segmap[3:6, 3:6] = 1                  # 3×3 segmap region for source 1
    residual = np.zeros((10, 10))
    residual[3:6, 3:6] = 0.5              # in-segmap residual (9 × 0.5 = 4.5)
    residual[7, 7] = 100.0                # outside-segmap: must NOT be counted

    pl = Pipeline([np.zeros((10, 10))], segmap)
    # orig_t whose slices_original covers the segmap region within the parent image
    orig_t = Template(np.zeros((10, 10)), (4, 4), (6, 6), label=1)

    result = pl._residual_segmap_sum(residual, source_id=1, orig_t=orig_t, k=1)
    assert result == pytest.approx(0.5 * 9)
    assert result < 100.0                  # explicitly: outside-segmap pixel excluded


def test_residual_segmap_sum_multi_res():
    """k=2: high-res segmap mask is binned down to match low-res residual."""
    segmap = np.zeros((10, 10), dtype=int)
    segmap[2:6, 2:6] = 1                  # 4×4 high-res region; aligned to k=2 blocks
    # Low-res residual: high-res [2:6) maps to low-res [1:3) under k=2
    residual_lo = np.zeros((5, 5))
    residual_lo[1:3, 1:3] = 0.5           # 4 low-res pixels in the binned segmap
    residual_lo[4, 4] = 100.0             # well outside segmap

    pl = Pipeline([np.zeros((5, 5))], segmap)
    orig_t = Template(np.zeros((10, 10)), (4, 4), (6, 6), label=1)

    result = pl._residual_segmap_sum(residual_lo, source_id=1, orig_t=orig_t, k=2)
    assert result == pytest.approx(0.5 * 4)
    assert result < 100.0


def test_aperture_photometry_uses_segmap_residual():
    """ap_flux_corr adds the segmap-extent residual, not the in-aperture residual."""
    flux_f444w_i = 25.0
    fl = 2.0
    f444w_total = 42.0

    segmap = np.zeros((10, 10), dtype=int)
    segmap[1:4, 1:4] = 1                  # 9 pixels in segmap
    residual = np.zeros((10, 10))
    residual[1:4, 1:4] = 0.3              # 9 × 0.3 = 2.7 over segmap

    pl = Pipeline([np.zeros((10, 10))], segmap)
    pl.psfs = [np.ones((5, 5))]
    cat = Table({"id": [1]})
    conv_tmpl = _make_tmpl(flux_f444w_i)
    orig_tmpl = _make_tmpl(flux_f444w_i)

    pl._add_aperture_photometry(
        cat, [conv_tmpl], np.array([fl]), residual, 1,
        r_orig_pix=1.5,
        orig_templates=[orig_tmpl],
        f444w_totals={1: f444w_total},
    )

    expected_res_seg = 0.3 * 9
    assert cat["res_seg_1"][0] == pytest.approx(expected_res_seg)

    ap_model = cat["ap_model_1"][0]
    apcor = cat["apcor_1"][0]
    # ap_flux_corr uses res_seg, NOT res_sum
    assert cat["ap_flux_corr_1"][0] == pytest.approx(ap_model * apcor + expected_res_seg)
