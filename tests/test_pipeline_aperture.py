import numpy as np
import pytest
from astropy.table import Table
from mophongo.pipeline import Pipeline
from mophongo.templates import Template


def _make_tmpl(template_norm=25.0):
    """Flat 5×5 template; template_norm mimics pre-normalisation F444W total."""
    tmpl = Template(np.ones((5, 5)), (2, 2), (10, 10), label=1)
    tmpl.template_norm = template_norm
    return tmpl


def test_aperture_photometry_estimator3():
    """Without f444w_totals: tcor_H = 1, so apcor = apcor1 = ap_f/ap_b; res_sum = 0."""
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

    for col in ("ap_model_1", "apcor_1", "apcor1_1", "tcor_1", "res_sum_1", "ap_flux_corr_1"):
        assert col in cat.colnames
    assert np.isfinite(cat["apcor_1"][0])
    assert np.isfinite(cat["ap_flux_corr_1"][0])
    assert cat["res_sum_1"][0] == pytest.approx(0.0)
    # No catalog total -> tcor_H = 1, combined apcor == apcor1 (factored, not collapsed)
    assert cat["tcor_1"][0] == pytest.approx(1.0)
    assert cat["apcor_1"][0] == pytest.approx(cat["apcor1_1"][0])
    # ap_flux_corr = ap_model * apcor1 * tcor_H + res_sum(=0)
    ap_model = cat["ap_model_1"][0]
    assert cat["ap_flux_corr_1"][0] == pytest.approx(
        ap_model * cat["apcor1_1"][0] * cat["tcor_1"][0]
    )


def test_aperture_photometry_with_tcor():
    """With f444w_totals: tcor_H = ftot/ap_f, apcor = apcor1*tcor_H (kept factored).

    ap_model * apcor1 * tcor_H collapses (only because residual=0 and ftot given)
    to fl * ftot / template_norm — verified as a value check, but the code keeps
    apcor1 and tcor_H as separate factors.
    """
    template_norm_i = 25.0
    fl = 2.0
    f444w_total = 42.0

    conv_tmpl = _make_tmpl(template_norm_i)
    orig_tmpl = _make_tmpl(template_norm_i)
    # F444W image holds this source's own model, so the neighbour-subtracted data
    # aperture flux (the new tcor_H denominator) equals the template aperture flux
    # (residual = 0). With all-zero data it would correctly be 0 (no real source).
    img = np.zeros((10, 10))
    img[orig_tmpl.slices_original] += orig_tmpl.data[orig_tmpl.slices_cutout] * template_norm_i
    pl = Pipeline([img], np.zeros((10, 10)))
    pl.psfs = [np.ones((5, 5))]
    cat = Table({"id": [1]})
    residual = np.zeros((10, 10))

    pl._add_aperture_photometry(
        cat, [conv_tmpl], np.array([fl]), residual, 1,
        r_orig_pix=1.5,
        orig_templates=[orig_tmpl],
        f444w_totals={1: f444w_total},
    )

    # Factored correction stored separately and consistently.
    assert cat["tcor_1"][0] != pytest.approx(1.0)
    assert cat["apf_data_1"][0] > 0   # neighbour-subtracted F444W aperture flux (tcor_H denom)
    assert cat["apcor_1"][0] == pytest.approx(cat["apcor1_1"][0] * cat["tcor_1"][0])
    ap_model = cat["ap_model_1"][0]
    assert cat["ap_flux_corr_1"][0] == pytest.approx(
        ap_model * cat["apcor1_1"][0] * cat["tcor_1"][0]
    )
    # value identity (res=0, ftot given): ap_model*apcor1*tcor_H = fl*ftot/template_norm
    assert cat["ap_flux_corr_1"][0] == pytest.approx(fl * f444w_total / template_norm_i)


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


def test_estimator3_uses_aperture_residual_not_segmap():
    """Estimator 3 adds the UNSCALED residual over the measurement aperture disk
    (res_sum), not the segmap-extent residual (res_seg).

    Two residual blobs: one at the template centre (inside the aperture) and one
    in a far corner that is inside the segmap but outside the aperture. res_seg
    sees both; res_sum sees only the central one, and ap_flux_corr must use
    res_sum.
    """
    from mophongo.fit import FitConfig

    template_norm_i = 25.0
    fl = 2.0
    f444w_total = 42.0

    n = 21
    c = n // 2  # 10
    segmap = np.zeros((n, n), dtype=int)
    segmap[:] = 0
    segmap[c - 6:c + 7, c - 6:c + 7] = 1   # large segment spanning the cutout
    residual = np.zeros((n, n))
    residual[c, c] = 0.3                    # at centre, inside the r=3 aperture
    residual[c + 6, c + 6] = 5.0            # dist ~8.5 px: inside segmap, outside aperture

    # Fixed 6-pixel-diameter (r=3) aperture so the disk is smaller than the segment.
    cfg = FitConfig(aperture_diam=6.0, aperture_units="pix")
    conv_tmpl = Template(np.ones((n, n)), (c, c), (n, n), label=1)
    conv_tmpl.template_norm = template_norm_i
    orig_tmpl = Template(np.ones((n, n)), (c, c), (n, n), label=1)
    orig_tmpl.template_norm = template_norm_i
    # F444W image = source model, so the neighbour-subtracted data aperture flux
    # (tcor_H denominator) equals the template ap_f (F444W residual = 0).
    img = np.zeros((n, n))
    img[orig_tmpl.slices_original] += orig_tmpl.data[orig_tmpl.slices_cutout] * template_norm_i
    pl = Pipeline([img], segmap, config=cfg)
    pl.psfs = [np.ones((5, 5))]
    cat = Table({"id": [1]})

    pl._add_aperture_photometry(
        cat, [conv_tmpl], np.array([fl]), residual, 1,
        r_orig_pix=3.0,
        orig_templates=[orig_tmpl],
        f444w_totals={1: f444w_total},
    )

    # res_seg sees both blobs; res_sum sees only the in-aperture one -> they differ.
    assert cat["res_seg_1"][0] == pytest.approx(0.3 + 5.0)
    assert cat["res_sum_1"][0] == pytest.approx(0.3)

    ap_model = cat["ap_model_1"][0]
    # Estimator 3 uses res_sum (in-aperture), NOT res_seg
    assert cat["ap_flux_corr_1"][0] == pytest.approx(
        ap_model * cat["apcor1_1"][0] * cat["tcor_1"][0] + cat["res_sum_1"][0]
    )
    assert cat["ap_flux_corr_1"][0] != pytest.approx(
        ap_model * cat["apcor1_1"][0] * cat["tcor_1"][0] + cat["res_seg_1"][0]
    )
