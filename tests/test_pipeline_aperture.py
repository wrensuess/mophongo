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


# --- low-SNR catalog-anchored tcor_H denominator -----------------------------

def _gauss(n, sigma):
    c = (n - 1) / 2.0
    y, x = np.mgrid[0:n, 0:n]
    p = np.exp(-((x - c) ** 2 + (y - c) ** 2) / (2 * sigma ** 2))
    return p / p.sum()


def _simple_wcs(pscale_arcsec=0.04, n=25):
    from astropy.wcs import WCS
    w = WCS(naxis=2)
    w.wcs.crpix = [n / 2.0, n / 2.0]
    w.wcs.crval = [150.0, 2.0]
    w.wcs.cdelt = [-pscale_arcsec / 3600.0, pscale_arcsec / 3600.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


def _lowsnr_setup(snr_seg=1.0, enable=True, template_norm=25.0, ftot=42.0,
                  catalog_cols=False, tot_cor=3.0, use_aper=0.2):
    """Pipeline whose F444W image == the source's own model (residual=0), so
    ap_f_data == template ap_f == template_norm * ap_F_frac exactly.

    catalog_cols=True attaches tot_cor/use_aper to self.catalog and a WCS,
    enabling the Rung-1 catalog-anchored prediction; otherwise Rung 2 applies
    (Fap_pred = ftot * ap_F_frac).
    """
    from mophongo.fit import FitConfig
    n, tn = 25, template_norm
    c = n // 2
    prof = _gauss(n, 2.5)  # peaked profile -> non-trivial growth ratio
    conv = Template(prof.copy(), (c, c), (n, n), label=1)
    conv.template_norm = tn
    orig = Template(prof.copy(), (c, c), (n, n), label=1)
    orig.template_norm = tn
    orig.snr_seg = snr_seg
    img = np.zeros((n, n))
    img[orig.slices_original] += orig.data[orig.slices_cutout] * tn
    cfg_kw = dict(tcor_lowsnr_psf=enable, tcor_blend_center=1.5, fit_snrlo_psf=10.0)
    pl_kw = {}
    if catalog_cols:
        cfg_kw.update(f444w_totcor_col="tot_cor", f444w_aper_col="use_aper")
        pl_kw["catalog"] = Table({"id": [1], "x": [c], "y": [c],
                                  "tot_cor": [tot_cor], "use_aper": [use_aper]})
        pl_kw["wcs"] = [_simple_wcs(0.04, n)]
    pl = Pipeline([img], np.zeros((n, n)), config=FitConfig(**cfg_kw), **pl_kw)
    pl.psfs = [np.ones((5, 5))]
    cat = Table({"id": [1]})
    pl._add_aperture_photometry(
        cat, [conv], np.array([2.0]), np.zeros((n, n)), 1,
        r_orig_pix=5.0, orig_templates=[orig], f444w_totals={1: ftot},
    )
    return cat


def test_tcor_lowsnr_disabled_by_default():
    """tcor_lowsnr_psf=False: blend inactive, w=1, denom == measured ap_f_data."""
    cat = _lowsnr_setup(snr_seg=1.0, enable=False)
    assert cat["tcor_w_1"][0] == pytest.approx(1.0)
    assert cat["aper_rphi_1"][0] == pytest.approx(cat["apf_data_1"][0])
    assert cat["tcor_1"][0] == pytest.approx(42.0 / cat["apf_data_1"][0])


def test_tcor_lowsnr_high_snr_identity():
    """Enabled but high snr_seg -> w rounds to 1.0 -> denom bit-identical to ap_f_data."""
    cat = _lowsnr_setup(snr_seg=1000.0, enable=True)
    assert cat["tcor_w_1"][0] == pytest.approx(1.0)
    assert cat["aper_rphi_1"][0] == pytest.approx(cat["apf_data_1"][0])


def test_tcor_lowsnr_nan_snr_no_op():
    """NaN snr_seg (non-extended template) -> w=1 even when enabled."""
    cat = _lowsnr_setup(snr_seg=float("nan"), enable=True)
    assert cat["tcor_w_1"][0] == pytest.approx(1.0)
    assert cat["aper_rphi_1"][0] == pytest.approx(cat["apf_data_1"][0])


def test_tcor_lowsnr_rung2_predicts_from_total():
    """No catalog columns -> Rung 2: Fap_pred = ftot * ap_F_frac. With residual=0,
    ap_F_frac = ap_f_data / template_norm, so aper_pred = ftot * ap_f_data / tn, and
    the blended denom / tcor_H follow the convex-blend identity exactly."""
    tn, ftot = 25.0, 42.0
    cat = _lowsnr_setup(snr_seg=1.0, enable=True, template_norm=tn, ftot=ftot,
                        catalog_cols=False)
    w = cat["tcor_w_1"][0]
    assert 0.0 < w < 1.0  # blend active at low snr
    apf_data = cat["apf_data_1"][0]
    aper_pred = cat["aper_pred_1"][0]
    assert aper_pred == pytest.approx(ftot * apf_data / tn)
    assert cat["aper_rphi_1"][0] == pytest.approx(w * apf_data + (1 - w) * aper_pred)
    assert cat["tcor_1"][0] == pytest.approx(ftot / cat["aper_rphi_1"][0])


def test_tcor_lowsnr_rung1_catalog_anchored():
    """With tot_cor/use_aper: Fap_pred = (ftot/tot_cor) * apF(Rphi)/apF(r_color),
    a NOISE-FREE positive prediction. Verify it is finite, positive, differs from the
    Rung-2 value, and drives the blend/tcor_H identity."""
    tn, ftot, tc = 25.0, 42.0, 3.0
    cat = _lowsnr_setup(snr_seg=1.0, enable=True, template_norm=tn, ftot=ftot,
                        catalog_cols=True, tot_cor=tc, use_aper=0.2)
    w = cat["tcor_w_1"][0]
    assert 0.0 < w < 1.0
    aper_pred = cat["aper_pred_1"][0]
    assert np.isfinite(aper_pred) and aper_pred > 0
    # growth apF(Rphi)/apF(r_color) > 1, so aper_pred > color_flux = ftot/tot_cor
    assert aper_pred > ftot / tc
    assert cat["aper_rphi_1"][0] == pytest.approx(w * cat["apf_data_1"][0] + (1 - w) * aper_pred)
    assert cat["tcor_1"][0] == pytest.approx(ftot / cat["aper_rphi_1"][0])
    # Rung 1 (catalog color aperture) differs from Rung 2 (assumes template total).
    cat2 = _lowsnr_setup(snr_seg=1.0, enable=True, template_norm=tn, ftot=ftot,
                         catalog_cols=False)
    assert cat["aper_pred_1"][0] != pytest.approx(cat2["aper_pred_1"][0])


def test_tcor_lowsnr_clamps_negative_measurement():
    """Over-subtracted (negative measured) F444W aperture flux is clamped to 0 in the
    blend, so the denominator stays positive (no tcor_H spike / residual band tail);
    the raw negative value is preserved in the apf_data diagnostic."""
    from mophongo.fit import FitConfig
    n, tn, ftot = 25, 25.0, 42.0
    c = n // 2
    prof = _gauss(n, 2.5)
    conv = Template(prof.copy(), (c, c), (n, n), label=1); conv.template_norm = tn
    orig = Template(prof.copy(), (c, c), (n, n), label=1); orig.template_norm = tn
    orig.snr_seg = 4.0                              # intermediate SNR -> 0 < w < 1
    img = np.full((n, n), -0.5)                     # negative F444W -> ap_f_data < 0
    cfg = FitConfig(tcor_lowsnr_psf=True, tcor_blend_center=1.5, fit_snrlo_psf=10.0)
    pl = Pipeline([img], np.zeros((n, n)), config=cfg)
    pl.psfs = [np.ones((5, 5))]
    cat = Table({"id": [1]})
    pl._add_aperture_photometry(cat, [conv], np.array([2.0]), np.zeros((n, n)), 1,
                                r_orig_pix=5.0, orig_templates=[orig], f444w_totals={1: ftot})
    w = cat["tcor_w_1"][0]
    assert 0.0 < w < 1.0
    assert cat["apf_data_1"][0] < 0                 # raw measured value negative (diagnostic kept)
    aper_pred = cat["aper_pred_1"][0]
    # clamp: aper_rphi = w*max(apf_data,0) + (1-w)*aper_pred = (1-w)*aper_pred > 0
    assert cat["aper_rphi_1"][0] == pytest.approx((1 - w) * aper_pred)
    assert cat["aper_rphi_1"][0] > 0
    assert cat["tcor_1"][0] > 0                      # no sign flip / spike


def test_apcor_from_psf_uses_band_native_pixel_scale():
    """The band-PSF EE (apB) must be measured at the aperture radius in the BAND
    PSF's NATIVE pixel scale, not the (possibly upsampled) fit-grid r_img_pix.
    Regression for the upsample-mode grid mismatch that collapsed totcor1 to ~1."""
    from mophongo.fit import FitConfig
    import mophongo.utils as utils
    n, tn = 25, 25.0; c = n // 2
    prof = _gauss(n, 2.5)
    conv = Template(prof.copy(), (c, c), (n, n), label=1); conv.template_norm = tn
    orig = Template(prof.copy(), (c, c), (n, n), label=1); orig.template_norm = tn
    orig.snr_seg = 1.0; orig.apcor_from_psf = True
    psf444 = _gauss(21, 2.0); psf_band = _gauss(21, 3.5)
    pl = Pipeline([np.zeros((n, n))], np.zeros((n, n)), config=FitConfig())
    pl.psfs = [psf444, psf_band]
    pl._native_pscale = [0.04, 0.08]           # ref 40 mas, band native 80 mas (upsample x2)
    cat = Table({"id": [1]})
    pl._add_aperture_photometry(cat, [conv], np.array([1.0]), np.zeros((n, n)), 1,
                                r_orig_pix=15.0, orig_templates=[orig])
    # apB measured at r_band = r_orig * 0.04/0.08 = 7.5 native px (NOT r_img_pix)
    assert cat["totcor1_1"][0] == pytest.approx(1.0 / utils.psf_ee_at_radius(psf_band, 7.5))


def test_tcor_lowsnr_rung1_psf_branch():
    """apcor_from_psf source WITH rung-1 columns: apF_frac_color must come from the
    PSF curve of growth (not the template), so Fap_pred uses EE_psf(Rphi)/EE_psf(r_color)."""
    from mophongo.fit import FitConfig
    import mophongo.utils as utils
    n, tn, ftot, tc, ua = 25, 25.0, 42.0, 3.0, 0.2
    c = n // 2
    prof = _gauss(n, 2.5)
    conv = Template(prof.copy(), (c, c), (n, n), label=1); conv.template_norm = tn
    orig = Template(prof.copy(), (c, c), (n, n), label=1); orig.template_norm = tn
    orig.snr_seg = 1.0
    orig.apcor_from_psf = True                       # force the PSF branch
    psf444 = _gauss(21, 2.0); psf_band = _gauss(21, 3.5)
    img = np.zeros((n, n)); img[orig.slices_original] += orig.data[orig.slices_cutout] * tn
    cfg = FitConfig(tcor_lowsnr_psf=True, tcor_blend_center=1.5, fit_snrlo_psf=10.0,
                    f444w_totcor_col="tot_cor", f444w_aper_col="use_aper")
    catalog = Table({"id": [1], "x": [c], "y": [c], "tot_cor": [tc], "use_aper": [ua]})
    pl = Pipeline([img], np.zeros((n, n)), catalog=catalog,
                  wcs=[_simple_wcs(0.04, n)], config=cfg)
    pl.psfs = [psf444, psf_band]                     # ndarray PSFs -> _psf_ee ignores ra/dec
    cat = Table({"id": [1]})
    pl._add_aperture_photometry(cat, [conv], np.array([2.0]), np.zeros((n, n)), 1,
                                r_orig_pix=5.0, orig_templates=[orig], f444w_totals={1: ftot})
    r_color = 0.5 * ua / 0.04  # arcsec radius -> F444W px at 0.04"/px
    expected = (ftot / tc) * (utils.psf_ee_at_radius(psf444, 5.0)
                              / utils.psf_ee_at_radius(psf444, r_color))
    assert cat["aper_pred_1"][0] == pytest.approx(expected)


def test_apcor_from_psf_uses_psf_curve_of_growth():
    """apcor_from_psf source: apF/apB come from the PSF curve of growth (utils.
    psf_ee_at_radius), so totcor1 = 1/EE(PSF_band, r_img). Regression for the
    module-level utils import used by the PSF path."""
    from mophongo.fit import FitConfig
    import mophongo.utils as utils
    n, tn = 25, 25.0
    c = n // 2
    prof = _gauss(n, 2.5)
    conv = Template(prof.copy(), (c, c), (n, n), label=1); conv.template_norm = tn
    orig = Template(prof.copy(), (c, c), (n, n), label=1); orig.template_norm = tn
    orig.snr_seg = 1.0
    orig.apcor_from_psf = True                      # force the PSF branch
    psf444 = _gauss(21, 2.0)
    psf_band = _gauss(21, 3.5)                       # broader band PSF
    pl = Pipeline([np.zeros((n, n))], np.zeros((n, n)), config=FitConfig())
    pl.psfs = [psf444, psf_band]                     # ndarray PSFs -> _psf_ee ignores ra/dec
    cat = Table({"id": [1]})
    pl._add_aperture_photometry(
        cat, [conv], np.array([1.0]), np.zeros((n, n)), 1,
        r_orig_pix=5.0, orig_templates=[orig],
    )
    r_img = pl._resolve_image_ap_radius_pix(1, pl.config)
    assert cat["totcor1_1"][0] == pytest.approx(1.0 / utils.psf_ee_at_radius(psf_band, r_img))
    assert cat["apcor1_1"][0] == pytest.approx(
        utils.psf_ee_at_radius(psf444, 5.0) / utils.psf_ee_at_radius(psf_band, r_img)
    )


def test_apcor_from_psf_bookkeeping_uses_template_not_psf():
    """Real-flux bookkeeping (ap_model, ap_flux) must use the fitted template's own
    aperture fraction, NOT the PSF curve of growth, even for apcor_from_psf sources --
    regression for the Phase-A bug (docs/aperture_corrections.md Sec 4.2/5.3) that
    multiplied ap_model by the PSF/template EE ratio. totcor1/apcor1 keep the switched
    (PSF) behaviour, since those are correction factors, not bookkeeping."""
    from mophongo.fit import FitConfig
    import mophongo.utils as utils
    n, tn, fl = 25, 25.0, 3.0
    c = n // 2
    prof = _gauss(n, 2.5)                       # template profile
    conv = Template(prof.copy(), (c, c), (n, n), label=1); conv.template_norm = tn
    orig = Template(prof.copy(), (c, c), (n, n), label=1); orig.template_norm = tn
    orig.snr_seg = 1.0
    orig.apcor_from_psf = True                  # force the PSF branch for corrections
    psf444 = _gauss(21, 2.0)
    psf_band = _gauss(21, 3.5)                  # broader than the template -> EE differs measurably
    pl = Pipeline([np.zeros((n, n))], np.zeros((n, n)), config=FitConfig())
    pl.psfs = [psf444, psf_band]                # ndarray PSFs -> _psf_ee ignores ra/dec
    cat = Table({"id": [1]})
    residual = np.zeros((n, n))

    pl._add_aperture_photometry(
        cat, [conv], np.array([fl]), residual, 1,
        r_orig_pix=5.0, orig_templates=[orig],
    )

    r_img = pl._resolve_image_ap_radius_pix(1, pl.config)
    apB_template = pl._aperture_sum_on_template(conv, r_img)
    ee_band = utils.psf_ee_at_radius(psf_band, r_img)
    assert apB_template != pytest.approx(ee_band)  # PSF EE differs measurably from the template fraction

    # Bookkeeping: ap_model/ap_flux use the template's own fraction, NOT the PSF value.
    assert cat["ap_model_1"][0] == pytest.approx(fl * apB_template)
    assert cat["ap_model_1"][0] != pytest.approx(fl * ee_band)
    assert cat["ap_flux_1"][0] == pytest.approx(cat["ap_model_1"][0] + cat["res_sum_1"][0])

    # Correction factors: totcor1 keeps the PSF curve-of-growth (switched) behaviour.
    assert cat["totcor1_1"][0] == pytest.approx(1.0 / ee_band)


def test_apcor_from_psf_containment_true_normalizes_totcor1():
    """PSFRegionMap band PSF with containment=0.9: the stamp-normalized EE must be
    true-total normalized by multiplying by containment (docs/aperture_corrections.md
    Sec 4.1/5.2), so totcor1 = 1/(EE_stamp * containment). Real-flux bookkeeping
    (ap_model) is untouched -- containment enters ONLY the correction side."""
    import geopandas as gpd
    import shapely.geometry as sgeom
    from mophongo.fit import FitConfig
    from mophongo.psf_map import PSFRegionMap
    import mophongo.utils as utils

    n, tn, fl = 25, 25.0, 1.0
    c = n // 2
    prof = _gauss(n, 2.5)
    conv = Template(prof.copy(), (c, c), (n, n), label=1); conv.template_norm = tn
    orig = Template(prof.copy(), (c, c), (n, n), label=1); orig.template_norm = tn
    orig.snr_seg = 1.0
    orig.apcor_from_psf = True                      # force the PSF branch
    psf444 = _gauss(21, 2.0)
    psf_band = _gauss(21, 3.5)

    # Single region covering any (ra, dec) the test template resolves to
    # (orig.wcs is None, so ra_dec == input_position_cutout, a pixel position).
    regions = gpd.GeoDataFrame(
        {"psf_key": [0]}, geometry=[sgeom.box(-1e4, -1e4, 1e4, 1e4)], crs=None
    )

    def _run(containment):
        prm_band = PSFRegionMap(regions=regions.copy(), psfs=np.array([psf_band]),
                                 containment=containment)
        pl = Pipeline([np.zeros((n, n))], np.zeros((n, n)), config=FitConfig())
        pl.psfs = [psf444, prm_band]                 # band PSF via PSFRegionMap
        cat = Table({"id": [1]})
        pl._add_aperture_photometry(
            cat, [conv], np.array([fl]), np.zeros((n, n)), 1,
            r_orig_pix=5.0, orig_templates=[orig],
        )
        return pl, cat

    pl90, cat90 = _run(0.9)
    _pl100, cat100 = _run(1.0)

    r_img = pl90._resolve_image_ap_radius_pix(1, pl90.config)
    ee_band = utils.psf_ee_at_radius(psf_band, r_img)
    assert cat90["totcor1_1"][0] == pytest.approx(1.0 / (ee_band * 0.9))
    # Bookkeeping is bit-identical regardless of containment.
    assert cat90["ap_model_1"][0] == cat100["ap_model_1"][0]


def test_psf_ee_cache_keys_on_region_not_psf_id():
    """Regression: the _psf_ee cache must key on (psfmap, region), not id(psf).
    PSFRegionMap.get_psf returns a fresh ndarray view per call and CPython
    reuses freed ids, so an id(psf)-keyed cache collides across regions and
    some sources silently get another region's EE (pre-existing since Phase A).
    20 sources in 20 regions with distinct band-PSF widths: every totcor1 must
    match the direct curve-of-growth computation for its OWN region."""
    import geopandas as gpd
    import shapely.geometry as sgeom
    from astropy.wcs import WCS
    from mophongo.fit import FitConfig
    from mophongo.psf_map import PSFRegionMap
    import mophongo.utils as utils

    n_src, n, tn = 20, 25, 25.0
    W = n_src * 30
    img = np.zeros((n, W))
    blob = _gauss(n, 2.5)
    xs = [i * 30 + 12 for i in range(n_src)]
    for x in xs:
        img[:, x - 12:x + 13] += blob

    # Full-image WCS: Cutout2D adjusts it per template, so each source's
    # ra_dec (used for the region lookup) reflects its true image position.
    w = WCS(naxis=2)
    w.wcs.crpix = [1.0, 1.0]
    w.wcs.crval = [150.0, 2.0]
    w.wcs.cdelt = [-0.04 / 3600.0, 0.04 / 3600.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    convs, origs = [], []
    for i, x in enumerate(xs):
        conv = Template(img, (x, 12), (n, n), label=i + 1, wcs=w); conv.template_norm = tn
        orig = Template(img, (x, 12), (n, n), label=i + 1, wcs=w); orig.template_norm = tn
        orig.snr_seg = 1.0
        orig.apcor_from_psf = True                  # force the PSF branch
        convs.append(conv); origs.append(orig)

    # One region per source (small sky box around it), each with a band PSF of
    # a distinct width so a cross-region cache hit is detectable.
    half = 15 * 0.04 / 3600.0  # half the 30 px source spacing, in deg
    boxes, band_psfs = [], []
    for i, x in enumerate(xs):
        ra, dec = w.wcs_pix2world(x, 12, 0)
        boxes.append(sgeom.box(float(ra) - half, float(dec) - half,
                               float(ra) + half, float(dec) + half))
        band_psfs.append(_gauss(21, 1.5 + 0.15 * i))
    regions = gpd.GeoDataFrame({"psf_key": list(range(n_src))}, geometry=boxes, crs=None)
    prm_band = PSFRegionMap(regions=regions, psfs=np.stack(band_psfs))

    psf444 = _gauss(21, 2.0)
    pl = Pipeline([np.zeros((n, W))], np.zeros((n, W)), config=FitConfig())
    pl.psfs = [psf444, prm_band]
    cat = Table({"id": list(range(1, n_src + 1))})
    pl._add_aperture_photometry(cat, convs, np.ones(n_src), np.zeros((n, W)), 1,
                                r_orig_pix=5.0, orig_templates=origs)

    r_img = pl._resolve_image_ap_radius_pix(1, pl.config)
    for i in range(n_src):
        assert cat["totcor1_1"][i] == pytest.approx(
            1.0 / utils.psf_ee_at_radius(band_psfs[i], r_img)
        ), f"source {i}: cached EE came from another region's PSF"


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
