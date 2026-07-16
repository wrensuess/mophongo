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
    """Without f444w_totals or a catalog aperture column: tcor_int falls back to
    1/apF_corr (the true-normalized point-source form -- docs/aperture_corrections.md
    Sec 5.4); s_cat and the catalog-tied apcor_/ap_flux_est3cat_ columns are
    bad_value (no ftot supplied to tie to)."""
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

    for col in ("ap_model_1", "apcor_1", "apcor1_1", "tcor_int_1", "s_cat_1",
                "f444w_ktot_1", "res_sum_1", "ap_flux_est3int_1", "ap_flux_est3cat_1"):
        assert col in cat.colnames
    assert cat["res_sum_1"][0] == pytest.approx(0.0)

    # No catalog aperture column -> tcor_int falls back to 1/apF_corr.
    apF_corr = pl._aperture_sum_on_template(orig_tmpl, 1.5)
    assert cat["tcor_int_1"][0] == pytest.approx(1.0 / apF_corr)
    assert np.isfinite(cat["ap_flux_est3int_1"][0])

    # No f444w_totals -> s_cat (and the catalog-tied columns) are bad_value.
    assert not np.isfinite(cat["s_cat_1"][0])
    assert not np.isfinite(cat["apcor_1"][0])
    assert not np.isfinite(cat["ap_flux_est3cat_1"][0])

    # ap_flux_est3int = ap_model * apcor1 * tcor_int + res_sum(=0)
    ap_model = cat["ap_model_1"][0]
    assert cat["ap_flux_est3int_1"][0] == pytest.approx(
        ap_model * cat["apcor1_1"][0] * cat["tcor_int_1"][0]
    )


def test_aperture_photometry_with_tcor():
    """With f444w_totals but no catalog aperture column: tcor_int falls back to
    1/apF_corr, s_cat = ftot/f444w_ktot, and apcor = apcor1*tcor_int*s_cat (kept
    factored, never pre-collapsed).

    ap_model * apcor1 * tcor_int * s_cat collapses (only because residual=0,
    ftot given, and the template-path total == template_norm) to
    fl * ftot / template_norm — verified as a value check.
    """
    template_norm_i = 25.0
    fl = 2.0
    f444w_total = 42.0

    conv_tmpl = _make_tmpl(template_norm_i)
    orig_tmpl = _make_tmpl(template_norm_i)
    # F444W image holds this source's own model, so the neighbour-subtracted data
    # aperture flux (the apf_data diagnostic) equals the template aperture flux
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

    assert cat["apf_data_1"][0] > 0   # neighbour-subtracted F444W aperture flux (diagnostic)
    assert cat["s_cat_1"][0] == pytest.approx(f444w_total / cat["f444w_ktot_1"][0])
    apcor_expected = cat["apcor1_1"][0] * cat["tcor_int_1"][0] * cat["s_cat_1"][0]
    assert cat["apcor_1"][0] == pytest.approx(apcor_expected)
    ap_model = cat["ap_model_1"][0]
    assert cat["ap_flux_est3cat_1"][0] == pytest.approx(
        ap_model * apcor_expected + cat["res_sum_1"][0]
    )
    # value identity (res=0, ftot given, no aperture floor -> template-path total
    # == template_norm): ap_model*apcor1*tcor_int*s_cat = fl*ftot/template_norm
    assert cat["ap_flux_est3cat_1"][0] == pytest.approx(fl * f444w_total / template_norm_i)


def test_s_cat_requires_positive_catalog_total():
    """A negative catalog f_f444w (an F444W non-detection) cannot define a
    total-flux system: s_cat/apcor/ap_flux_est3cat must be bad_value, while the
    catalog-independent tcor_int/f444w_ktot/ap_flux_est3int stay valid."""
    template_norm_i = 25.0
    fl = 2.0

    conv_tmpl = _make_tmpl(template_norm_i)
    orig_tmpl = _make_tmpl(template_norm_i)
    img = np.zeros((10, 10))
    img[orig_tmpl.slices_original] += orig_tmpl.data[orig_tmpl.slices_cutout] * template_norm_i
    pl = Pipeline([img], np.zeros((10, 10)))
    pl.psfs = [np.ones((5, 5))]
    cat = Table({"id": [1]})

    pl._add_aperture_photometry(
        cat, [conv_tmpl], np.array([fl]), np.zeros((10, 10)), 1,
        r_orig_pix=1.5,
        orig_templates=[orig_tmpl],
        f444w_totals={1: -3.0},   # finite but negative catalog total
    )

    # Catalog-tied outputs: bad_value.
    assert not np.isfinite(cat["s_cat_1"][0])
    assert not np.isfinite(cat["apcor_1"][0])
    assert not np.isfinite(cat["ap_flux_est3cat_1"][0])
    # Catalog-independent outputs: unaffected.
    assert np.isfinite(cat["tcor_int_1"][0]) and cat["tcor_int_1"][0] > 0
    assert np.isfinite(cat["f444w_ktot_1"][0]) and cat["f444w_ktot_1"][0] > 0
    assert np.isfinite(cat["ap_flux_est3int_1"][0])


# --- estimator suite / transitional tcor_H (Stage 3a) -----------------------

def _gauss(n, sigma):
    c = (n - 1) / 2.0
    y, x = np.mgrid[0:n, 0:n]
    p = np.exp(-((x - c) ** 2 + (y - c) ** 2) / (2 * sigma ** 2))
    return p / p.sum()


def test_ap_flux_est1():
    """ap_flux_est1 = (ap_model + res_sum) * totcor1 -- the IDL-exact estimator
    (aperture flux on the neighbour-subtracted image, scaled to total)."""
    pl = Pipeline([np.zeros((10, 10))], np.zeros((10, 10)))
    pl.psfs = [np.ones((5, 5))]
    cat = Table({"id": [1]})
    conv_tmpl = _make_tmpl()
    orig_tmpl = _make_tmpl()
    residual = np.zeros((10, 10))
    residual[2, 2] = 0.7  # nonzero residual inside the measurement aperture

    pl._add_aperture_photometry(
        cat, [conv_tmpl], np.array([1.0]), residual, 1,
        r_orig_pix=1.5,
        orig_templates=[orig_tmpl],
    )

    assert cat["res_sum_1"][0] != pytest.approx(0.0)
    ap_model = cat["ap_model_1"][0]
    res_sum = cat["res_sum_1"][0]
    totcor1 = cat["totcor1_1"][0]
    assert cat["ap_flux_est1_1"][0] == pytest.approx((ap_model + res_sum) * totcor1)


def test_apf_data_diagnostic_column():
    """apf_data is kept as a measured diagnostic (Stage 3b drops its use as a
    correction denominator -- docs/aperture_corrections.md Sec 5.4): it is the
    real neighbour-subtracted F444W aperture flux, independent of tcor_int/s_cat.
    (Replaces the Stage-3a test_tcor_transitional_measured; the transitional
    tcor_H/aper_rphi columns are gone.)"""
    template_norm_i = 25.0
    fl = 2.0

    conv_tmpl = _make_tmpl(template_norm_i)
    orig_tmpl = _make_tmpl(template_norm_i)
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
    )

    apf_data = cat["apf_data_1"][0]
    assert apf_data > 0
    apF_book = pl._aperture_sum_on_template(orig_tmpl, 1.5)
    assert apf_data == pytest.approx(template_norm_i * apF_book)


# --- Stage 3b: internal Kron total (tcor_int) + catalog tie (s_cat) ---------

def _simple_wcs():
    from astropy.wcs import WCS
    w = WCS(naxis=2)
    w.wcs.crpix = [1.0, 1.0]
    w.wcs.crval = [150.0, 2.0]
    w.wcs.cdelt = [-0.04 / 3600.0, 0.04 / 3600.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


def test_tcor_int_fallback_no_catalog_aperture_column():
    """No f444w_aper_col configured: tcor_int falls back to 1/apF_corr -- the
    true-normalized point-source form -- for every source (docs/
    aperture_corrections.md Sec 5.4 fallback)."""
    n, tn = 41, 60.0
    c = n // 2
    prof = _gauss(n, 3.0)
    convs, origs = [], []
    for i, cc in enumerate((c, c + 5)):
        conv = Template(prof.copy(), (cc, c), (n, n), label=i + 1); conv.template_norm = tn
        orig = Template(prof.copy(), (cc, c), (n, n), label=i + 1); orig.template_norm = tn
        convs.append(conv); origs.append(orig)

    pl = Pipeline([np.zeros((n, n))], np.zeros((n, n), dtype=int))
    pl.psfs = [np.ones((5, 5))]
    cat = Table({"id": [1, 2]})
    pl._add_aperture_photometry(
        cat, convs, np.array([1.0, 1.0]), np.zeros((n, n)), 1,
        r_orig_pix=5.0, orig_templates=origs,
    )
    for i, orig in enumerate(origs):
        apF_corr = pl._aperture_sum_on_template(orig, 5.0)
        assert cat["tcor_int_1"][i] == pytest.approx(1.0 / apF_corr)


def test_tcor_int_kron_construction_with_catalog_aperture():
    """With a catalog aperture column: tcor_int is built from a photutils Kron
    measurement on the model stamp (design doc Sec 5.4), verified against an
    independent photutils computation on the same stamp; s_cat = ftot/f444w_ktot;
    ap_flux_est3cat matches the explicit formula."""
    from photutils.segmentation import SourceCatalog, SegmentationImage
    import mophongo.utils as utils
    from mophongo.fit import FitConfig

    n, tn, fl = 41, 100.0, 4.0
    c = n // 2
    prof = _gauss(n, 3.0)
    conv = Template(prof.copy(), (c, c), (n, n), label=1); conv.template_norm = tn
    orig = Template(prof.copy(), (c, c), (n, n), label=1); orig.template_norm = tn
    orig.snr_seg = 20.0
    orig.apcor_from_psf = False   # bright/template-path -> real Kron measurement

    psf444 = _gauss(21, 2.0)

    segmap = np.zeros((n, n), dtype=int)
    segmap[c - 15:c + 16, c - 15:c + 16] = 1   # the source's own segment

    w = _simple_wcs()
    cat_src = Table({"id": [1], "use_aper": [0.5]})   # arcsec diameter
    cfg = FitConfig(f444w_aper_col="use_aper")
    ftot = 55.0
    pl = Pipeline([np.zeros((n, n))], segmap, catalog=cat_src, wcs=[w], config=cfg)
    pl.psfs = [psf444]
    cat = Table({"id": [1]})

    pl._add_aperture_photometry(
        cat, [conv], np.array([fl]), np.zeros((n, n)), 1,
        r_orig_pix=5.0, orig_templates=[orig],
        f444w_totals={1: ftot},
    )

    pscale_ref = pl._pixel_scale_arcsec(w)
    r_floor_pix = 0.5 * 0.5 / pscale_ref

    stamp = orig.data[orig.slices_cutout] * tn
    seg = segmap[orig.slices_original] == 1
    scat = SourceCatalog(stamp, SegmentationImage(seg.astype(int)),
                          kron_params=(2.5, 1.4, r_floor_pix))
    kron_flux = float(scat.kron_flux[0])
    kron_radius = float(scat.kron_radius[0].value)
    a = float(scat.semimajor_sigma[0].value)
    b = float(scat.semiminor_sigma[0].value)
    r_kron = max(2.5 * kron_radius * np.sqrt(a * b), r_floor_pix)
    r_kron = min(r_kron, 0.5 * min(stamp.shape))
    r_kron = np.round(r_kron * 4.0) / 4.0   # 0.25-px quantization (EE-cache)

    apF_book = pl._aperture_sum_on_template(orig, 5.0)
    ee_kron = utils.psf_ee_at_radius(psf444, r_kron)
    f444w_ktot_expected = kron_flux / ee_kron
    tcor_int_expected = f444w_ktot_expected / (tn * apF_book)

    assert cat["tcor_int_1"][0] == pytest.approx(tcor_int_expected, rel=1e-6)
    assert cat["f444w_ktot_1"][0] == pytest.approx(f444w_ktot_expected, rel=1e-6)
    assert cat["s_cat_1"][0] == pytest.approx(ftot / cat["f444w_ktot_1"][0])

    ap_model = cat["ap_model_1"][0]
    assert cat["ap_flux_est3cat_1"][0] == pytest.approx(
        ap_model * cat["apcor1_1"][0] * cat["tcor_int_1"][0] * cat["s_cat_1"][0]
        + cat["res_sum_1"][0]
    )


def test_tcor_int_kron_cap_shares_radius():
    """When the Kron radius exceeds the stamp half-width, the cap engages and
    kron_flux must be RE-MEASURED at the shared capped radius (regression for
    the numerator/denominator radius mismatch): f444w_ktot must equal
    template_norm * apF(r_cap) / EE(r_cap) exactly, never photutils' flux from
    the larger uncapped aperture divided by EE at the capped radius."""
    import mophongo.utils as utils
    from mophongo.fit import FitConfig

    n, tn, fl = 25, 60.0, 3.0
    c = n // 2
    prof = _gauss(n, 6.0)   # very broad profile in a tight stamp -> Kron radius >> cap
    conv = Template(prof.copy(), (c, c), (n, n), label=1); conv.template_norm = tn
    orig = Template(prof.copy(), (c, c), (n, n), label=1); orig.template_norm = tn
    orig.snr_seg = 30.0
    orig.apcor_from_psf = False   # real Kron measurement path

    psf444 = _gauss(21, 2.0)

    segmap = np.zeros((n, n), dtype=int)
    segmap[c - 10:c + 11, c - 10:c + 11] = 1

    w = _simple_wcs()
    cat_src = Table({"id": [1], "use_aper": [0.4]})
    cfg = FitConfig(f444w_aper_col="use_aper")
    pl = Pipeline([np.zeros((n, n))], segmap, catalog=cat_src, wcs=[w], config=cfg)
    pl.psfs = [psf444]
    cat = Table({"id": [1]})

    pl._add_aperture_photometry(
        cat, [conv], np.array([fl]), np.zeros((n, n)), 1,
        r_orig_pix=5.0, orig_templates=[orig],
    )

    stamp_shape = orig.data[orig.slices_cutout].shape
    r_cap = 0.5 * min(stamp_shape)
    r_cap_q = float(np.round(r_cap * 4.0) / 4.0)   # production's 0.25-px quantization
    # Sanity: the cap must actually have engaged for this profile.
    assert 2.5 * 1.0 * 6.0 > r_cap   # ~2.5*kron_radius*sqrt(ab) >> half-width for sigma=6

    kron_flux_expected = tn * pl._aperture_sum_on_template(orig, r_cap_q)
    ee_expected = utils.psf_ee_at_radius(psf444, r_cap_q)
    f444w_ktot_expected = kron_flux_expected / ee_expected

    assert cat["f444w_ktot_1"][0] == pytest.approx(f444w_ktot_expected, rel=1e-6)
    apF_book = pl._aperture_sum_on_template(orig, 5.0)
    assert cat["tcor_int_1"][0] == pytest.approx(f444w_ktot_expected / (tn * apF_book), rel=1e-6)


def test_tcor_int_apcor_from_psf_shortcut_uses_floor_circle(monkeypatch):
    """apcor_from_psf template + a catalog aperture column: the performance
    shortcut engages (docs Sec 5.4 "Performance note") -- photutils SourceCatalog
    is skipped entirely and tcor_int is built directly from the r_floor circle,
    verified against the same analytic construction used by the internal
    fallback."""
    import mophongo.pipeline as pipeline_mod
    import mophongo.utils as utils
    from mophongo.fit import FitConfig

    n, tn, fl = 25, 40.0, 2.0
    c = n // 2
    prof = _gauss(n, 2.5)
    conv = Template(prof.copy(), (c, c), (n, n), label=1); conv.template_norm = tn
    orig = Template(prof.copy(), (c, c), (n, n), label=1); orig.template_norm = tn
    orig.snr_seg = 1.0
    orig.apcor_from_psf = True     # faint/compact -> shortcut engages

    psf444 = _gauss(21, 2.0)

    segmap = np.zeros((n, n), dtype=int)
    segmap[c - 8:c + 9, c - 8:c + 9] = 1   # present but must NOT be consulted

    w = _simple_wcs()
    cat_src = Table({"id": [1], "use_aper": [0.4]})
    cfg = FitConfig(f444w_aper_col="use_aper")
    pl = Pipeline([np.zeros((n, n))], segmap, catalog=cat_src, wcs=[w], config=cfg)
    pl.psfs = [psf444]
    cat = Table({"id": [1]})

    def _boom(*a, **k):
        raise AssertionError("SourceCatalog must not be constructed for the apcor_from_psf shortcut")
    monkeypatch.setattr(pipeline_mod, "SourceCatalog", _boom)

    pl._add_aperture_photometry(
        cat, [conv], np.array([fl]), np.zeros((n, n)), 1,
        r_orig_pix=5.0, orig_templates=[orig],
    )

    pscale_ref = pl._pixel_scale_arcsec(w)
    r_floor_pix = 0.5 * 0.4 / pscale_ref
    apF_book = pl._aperture_sum_on_template(orig, 5.0)
    kron_flux_expected = tn * pl._aperture_sum_on_template(orig, r_floor_pix)
    ee_kron_expected = utils.psf_ee_at_radius(psf444, r_floor_pix)
    f444w_ktot_expected = kron_flux_expected / ee_kron_expected
    tcor_int_expected = f444w_ktot_expected / (tn * apF_book)

    assert cat["tcor_int_1"][0] == pytest.approx(tcor_int_expected, rel=1e-6)
    assert cat["f444w_ktot_1"][0] == pytest.approx(f444w_ktot_expected, rel=1e-6)


def test_tcor_int_noisy_faint_stamp_finite_positive():
    """Faint/noisy source (SourceCatalog degenerate or the floor engaged):
    tcor_int stays finite and positive, never NaN/negative (docs Sec 5.4's
    noise-robustness bar for the internal Kron total)."""
    from mophongo.fit import FitConfig

    n, tn = 31, 5.0   # faint: small template_norm
    c = n // 2
    rng = np.random.default_rng(0)
    prof = _gauss(n, 2.0)
    noisy = prof + rng.normal(scale=prof.max() * 0.5, size=prof.shape)
    noisy = np.clip(noisy, 0, None)
    noisy /= noisy.sum()   # keep it a valid (unit-sum) template profile

    conv = Template(noisy.copy(), (c, c), (n, n), label=1); conv.template_norm = tn
    orig = Template(noisy.copy(), (c, c), (n, n), label=1); orig.template_norm = tn
    orig.snr_seg = 1.0
    orig.apcor_from_psf = False

    segmap = np.zeros((n, n), dtype=int)
    segmap[c - 10:c + 11, c - 10:c + 11] = 1

    w = _simple_wcs()
    cat_src = Table({"id": [1], "use_aper": [0.6]})
    cfg = FitConfig(f444w_aper_col="use_aper")
    pl = Pipeline([np.zeros((n, n))], segmap, catalog=cat_src, wcs=[w], config=cfg)
    pl.psfs = [_gauss(21, 2.0)]
    cat = Table({"id": [1]})

    pl._add_aperture_photometry(
        cat, [conv], np.array([1.0]), np.zeros((n, n)), 1,
        r_orig_pix=5.0, orig_templates=[orig],
    )
    assert np.isfinite(cat["tcor_int_1"][0])
    assert cat["tcor_int_1"][0] > 0


def test_tcor_lowsnr_psf_kwarg_removed():
    """The low-SNR tcor_H blend machinery (tcor_lowsnr_psf, tcor_blend_center/width)
    is removed; the old kwarg must fail loudly (TypeError) rather than silently
    no-op, guarding against silent resurrection."""
    from mophongo.fit import FitConfig
    with pytest.raises(TypeError):
        FitConfig(tcor_lowsnr_psf=True)


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
    sees both; res_sum sees only the central one, and ap_flux_est3cat must use
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
    # (the apf_data diagnostic) equals the template ap_f (F444W residual = 0).
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
    # Estimator 3cat uses res_sum (in-aperture), NOT res_seg
    assert cat["ap_flux_est3cat_1"][0] == pytest.approx(
        ap_model * cat["apcor1_1"][0] * cat["tcor_int_1"][0] * cat["s_cat_1"][0]
        + cat["res_sum_1"][0]
    )
    assert cat["ap_flux_est3cat_1"][0] != pytest.approx(
        ap_model * cat["apcor1_1"][0] * cat["tcor_int_1"][0] * cat["s_cat_1"][0]
        + cat["res_seg_1"][0]
    )
