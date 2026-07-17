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


def test_truncation_cancels_in_apcor1_survives_in_totcor1():
    """docs/aperture_corrections.md Sec 5.1/6: the per-source truncation term
    (from ``flux_beyond_stamp``, the unified template's PSF-extrapolated
    core-anchored estimate of flux beyond the cutout) cancels exactly in
    apcor1 (a shape ratio) but survives in totcor1 (aperture-to-total).
    Replaces the deleted apcor_from_psf PSF-curve-of-growth branch for
    ``test_apcor_from_psf_uses_band_native_pixel_scale`` /
    ``..._uses_psf_curve_of_growth`` (that branch, and the band-native-pixel-
    scale conversion it needed, no longer exist: apB_corr/apF_corr are now a
    single footprint-truncated fraction times this uniform truncation term
    for every source).

    Stage-4b (ruling "A+cb"): reworked rather than deleted per the brief --
    the SAME truncation-cancellation invariant above still holds exactly, but
    now a SECOND, deliberately non-cancelling factor exists too: the band-side
    containment ratio c_b/c_det (docs Sec 5.1/6). Asserted here explicitly so
    the two invariants (trunc cancels; c_b/c_det does not) are never conflated."""
    from mophongo.fit import FitConfig
    n, tn, fl = 25, 25.0, 3.0
    c = n // 2
    prof = _gauss(n, 2.5)

    conv = Template(prof.copy(), (c, c), (n, n), label=1); conv.template_norm = tn
    orig = Template(prof.copy(), (c, c), (n, n), label=1); orig.template_norm = tn
    pl = Pipeline([np.zeros((n, n))], np.zeros((n, n)), config=FitConfig())
    pl.psfs = [np.ones((5, 5))]
    # Fixed r_img_pix=5.0 (matches r_orig_pix below and the pl3 block further
    # down) so all three sub-comparisons in this test share one aperture radius.
    pl.config.aperture_diam = 10.0
    pl.config.aperture_units = "pix"
    cat = Table({"id": [1]})
    residual = np.zeros((n, n))
    pl._add_aperture_photometry(cat, [conv], np.array([fl]), residual, 1,
                                r_orig_pix=5.0, orig_templates=[orig])
    apcor1_notrunc = cat["apcor1_1"][0]
    totcor1_notrunc = cat["totcor1_1"][0]

    conv2 = Template(prof.copy(), (c, c), (n, n), label=1); conv2.template_norm = tn
    orig2 = Template(prof.copy(), (c, c), (n, n), label=1); orig2.template_norm = tn
    orig2.flux_beyond_stamp = 0.5 * tn   # 1/3 of the source's flux lands beyond the cutout
    cat2 = Table({"id": [1]})
    pl._add_aperture_photometry(cat2, [conv2], np.array([fl]), residual, 1,
                                r_orig_pix=5.0, orig_templates=[orig2])

    assert cat2["apcor1_1"][0] == pytest.approx(apcor1_notrunc, rel=1e-10)
    trunc = tn / (tn + 0.5 * tn)
    assert cat2["totcor1_1"][0] == pytest.approx(totcor1_notrunc / trunc, rel=1e-10)

    # Stage-4b: introduce a genuine c_det != c_b (both psfs[0]/psfs[1] real
    # PSFRegionMaps with distinct containments; same conv/orig/trunc as the
    # apcor1_notrunc case above) -- apcor1 must gain EXACTLY c_det/c_b, not
    # cancel like trunc does.
    import geopandas as gpd
    import shapely.geometry as sgeom
    from mophongo.psf_map import PSFRegionMap

    regions = gpd.GeoDataFrame(
        {"psf_key": [0]}, geometry=[sgeom.box(-1e4, -1e4, 1e4, 1e4)], crs=None
    )
    c_det, c_b = 0.9622, 0.9192
    prm_det = PSFRegionMap(regions=regions, psfs=np.array([_gauss(21, 2.0)]), containment=c_det)
    prm_band = PSFRegionMap(regions=regions, psfs=np.array([_gauss(21, 2.0)]), containment=c_b)

    conv3 = Template(prof.copy(), (c, c), (n, n), label=1); conv3.template_norm = tn
    orig3 = Template(prof.copy(), (c, c), (n, n), label=1); orig3.template_norm = tn
    pl3 = Pipeline([np.zeros((n, n)), np.zeros((n, n))], np.zeros((n, n)), config=FitConfig())
    pl3.psfs = [prm_det, prm_band]
    pl3.config.aperture_diam = 10.0
    pl3.config.aperture_units = "pix"
    cat3 = Table({"id": [1]})
    pl3._add_aperture_photometry(cat3, [conv3], np.array([fl]), residual, 1,
                                 r_orig_pix=5.0, orig_templates=[orig3])
    assert cat3["apcor1_1"][0] == pytest.approx(apcor1_notrunc * (c_det / c_b), rel=1e-10)


def test_bookkeeping_invariant_to_truncation_term():
    """Real-flux bookkeeping (ap_model, ap_flux) must be invariant to
    flux_beyond_stamp -- the per-source truncation only enters apcor1/totcor1,
    never ap_model/ap_flux (docs/aperture_corrections.md Sec 4.2/5.1/5.3
    invariant). Re-targeted from the deleted apcor_from_psf PSF-EE branch: the
    fitted convolved template's own aperture fraction (apB_book) is now used
    for ap_model unconditionally, with no PSF-based override to guard against."""
    from mophongo.fit import FitConfig
    n, tn, fl = 25, 25.0, 3.0
    c = n // 2
    prof = _gauss(n, 2.5)
    conv = Template(prof.copy(), (c, c), (n, n), label=1); conv.template_norm = tn
    orig = Template(prof.copy(), (c, c), (n, n), label=1); orig.template_norm = tn
    orig.flux_beyond_stamp = 2.0 * tn   # large stamp-edge truncation

    pl = Pipeline([np.zeros((n, n))], np.zeros((n, n)), config=FitConfig())
    pl.psfs = [np.ones((5, 5))]
    cat = Table({"id": [1]})
    residual = np.zeros((n, n))
    pl._add_aperture_photometry(cat, [conv], np.array([fl]), residual, 1,
                                r_orig_pix=5.0, orig_templates=[orig])

    r_img = pl._resolve_image_ap_radius_pix(1, pl.config)
    apB_template = pl._aperture_sum_on_template(conv, r_img)
    assert cat["ap_model_1"][0] == pytest.approx(fl * apB_template)
    assert cat["ap_flux_1"][0] == pytest.approx(cat["ap_model_1"][0] + cat["res_sum_1"][0])

    # Correction factor DOES move with the truncation term (contrast with the
    # untouched bookkeeping above).
    assert cat["totcor1_1"][0] != pytest.approx(1.0 / apB_template)


def test_totcor1_faint_limit_matches_true_total_psf_ee():
    """Faint/noise-dominated source (in-segment sum clamped to 0 -> w_core == 0;
    an all-zero halo -> every annulus weight == 0 too), so H == M exactly over
    the model support. Per docs/aperture_corrections.md Sec 5.1/6's "faint
    limit check": Sigma(H) == A_src*f_cut and the true total == A_src/c_det,
    so apB_corr reproduces the Stage-2 true-total PSF EE exactly:
    totcor1 == 1/(EE(psf, r)*containment). Regression against the (deleted)
    apcor_from_psf PSF-EE branch's number, now produced by the unified
    template + truncation term instead.

    The PSF reach (r_reach = 5 px) is deliberately much SMALLER than the
    41-px stamp, leaving ~13% of the PSF's flux inside the cutout but OUTSIDE
    the model support: the identity only holds when f_cut is measured over
    the support H is actually built on (regression for the whole-cutout
    f_cut bug, which passes the identity only when reach covers the stamp).
    The measurement aperture (r=3 px) is set inside the reach."""
    import geopandas as gpd
    import shapely.geometry as sgeom
    from mophongo.templates import Templates
    from mophongo.psf_map import PSFRegionMap
    import mophongo.utils as utils

    n = 41
    c = n // 2
    psf = _gauss(21, 2.5)   # native detection PSF
    regions = gpd.GeoDataFrame(
        {"psf_key": [0]}, geometry=[sgeom.box(-1e4, -1e4, 1e4, 1e4)], crs=None
    )
    prm = PSFRegionMap(regions=regions, psfs=np.array([psf]), containment=0.9)

    image = np.zeros((n, n))
    segmap = np.zeros((n, n), dtype=int)
    segmap[c - 1:c + 2, c - 1:c + 2] = 1
    # Net-negative in-segment sum (IDL positive-pixel clamp -> snr_seg == 0
    # exactly), with one positive pixel so A_src stays well-defined.
    image[c - 1:c + 2, c - 1:c + 2] = -1.0
    image[c, c] = 3.0
    ivar = np.ones((n, n))

    r_reach = 5.0   # sigma=2.5 PSF: EE(5) ~ 0.86 -> support excludes real flux
    tmpls = Templates(min_size=n)
    tmpls.extract_templates(
        image, segmap, [(c, c)], extend_mode="auto",
        detection_psf=prm, detection_weight=ivar,
        max_radius_pix=r_reach, psf_ee_radius_pix=r_reach,
        fit_snrlo_psf=10.0, wings_snr_psf=3.0,
    )
    orig = tmpls._templates[0]
    assert orig.snr_seg == 0.0
    assert orig.flux_beyond_stamp > 0

    pl = Pipeline([image], segmap)
    pl.psfs = [np.ones((5, 5))]  # unused: no f444w_aper_col -> tcor_int fallback only
    pl.config.aperture_diam = 6.0    # r = 3 px, inside the 5-px reach
    pl.config.aperture_units = "pix"
    cat = Table({"id": [1]})
    pl._add_aperture_photometry(
        cat, [orig], np.array([1.0]), np.zeros((n, n)), 1,
        r_orig_pix=5.0, orig_templates=[orig],
    )

    r_img = pl._resolve_image_ap_radius_pix(1, pl.config)
    assert r_img == pytest.approx(3.0)
    expected = 1.0 / (utils.psf_ee_at_radius(psf, r_img) * 0.9)
    assert cat["totcor1_1"][0] == pytest.approx(expected, rel=1e-6)


# NOTE: test_apcor_from_psf_containment_true_normalizes_totcor1 (containment
# entering totcor1 via a PSFRegionMap band PSF) is gone -- the apB_corr PSF-EE
# branch it exercised no longer exists (apB_corr is now always apB_book*trunc,
# and a bare-constructed Template's flux_beyond_stamp defaults to 0, so
# containment on a hand-built "band PSF" has no path into totcor1 any more).
# Its regression intent (containment normalizes totcor1 to the true-total PSF
# EE) is superseded by test_totcor1_faint_limit_matches_true_total_psf_ee
# above, which exercises containment through the surviving pathway: the
# DETECTION PSF's containment feeding flux_beyond_stamp in
# Templates._extended_composite.


def test_psf_ee_cache_keys_on_region_not_psf_id():
    """Regression: the _psf_ee cache must key on (psfmap, region), not id(psf).
    PSFRegionMap.get_psf returns a fresh ndarray view per call and CPython
    reuses freed ids, so an id(psf)-keyed cache collides across regions and
    some sources silently get another region's EE (pre-existing since Phase A).
    Re-targeted from the deleted apB_corr PSF-EE branch to the surviving
    _psf_ee consumer: the apcor_from_psf Kron-shortcut's ee_kron lookup in
    _model_kron (docs Sec 5.4). 20 sources in 20 regions with distinct
    F444W-detection-PSF widths: every tcor_int/f444w_ktot must match the
    direct curve-of-growth computation for its OWN region."""
    import geopandas as gpd
    import shapely.geometry as sgeom
    from astropy.wcs import WCS
    from astropy.table import Table as ATable
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
        orig.apcor_from_psf = True   # Kron-shortcut path -> deterministic floor circle
        convs.append(conv); origs.append(orig)

    # One region per source (small sky box around it), each with a
    # F444W-detection PSF of a distinct width so a cross-region cache hit is
    # detectable.
    half = 15 * 0.04 / 3600.0  # half the 30 px source spacing, in deg
    boxes, hires_psfs = [], []
    for i, x in enumerate(xs):
        ra, dec = w.wcs_pix2world(x, 12, 0)
        boxes.append(sgeom.box(float(ra) - half, float(dec) - half,
                               float(ra) + half, float(dec) + half))
        hires_psfs.append(_gauss(21, 1.5 + 0.15 * i))
    regions = gpd.GeoDataFrame({"psf_key": list(range(n_src))}, geometry=boxes, crs=None)
    prm_hires = PSFRegionMap(regions=regions, psfs=np.stack(hires_psfs))

    # use_aper (arcsec) chosen so r_floor_pix == 5.0 exactly at pscale=0.04
    # arcsec/px, sidestepping _model_kron's 0.25-px quantization.
    cat_src = ATable({"id": list(range(1, n_src + 1)), "use_aper": [0.4] * n_src})
    cfg = FitConfig(f444w_aper_col="use_aper")
    pl = Pipeline([np.zeros((n, W))], np.zeros((n, W), dtype=int),
                  catalog=cat_src, wcs=[w], config=cfg)
    pl.psfs = [prm_hires]
    cat = Table({"id": list(range(1, n_src + 1))})
    pl._add_aperture_photometry(cat, convs, np.ones(n_src), np.zeros((n, W)), 1,
                                r_orig_pix=5.0, orig_templates=origs)

    pscale_ref = pl._pixel_scale_arcsec(w)
    r_floor_pix = 0.5 * 0.4 / pscale_ref
    assert r_floor_pix == pytest.approx(5.0)
    for i, orig in enumerate(origs):
        apF_book = pl._aperture_sum_on_template(orig, 5.0)
        kron_flux_expected = tn * pl._aperture_sum_on_template(orig, r_floor_pix)
        ee_expected = utils.psf_ee_at_radius(hires_psfs[i], r_floor_pix)
        f444w_ktot_expected = kron_flux_expected / ee_expected
        tcor_int_expected = f444w_ktot_expected / (tn * apF_book)
        assert cat["tcor_int_1"][i] == pytest.approx(tcor_int_expected, rel=1e-6), (
            f"source {i}: cached EE came from another region's PSF"
        )


# --- Stage-4b: corrections on the partially-unmasked model + band containment
# factor (docs/aperture_corrections.md Sec 5.1/6, ruling "A+cb") -----------

def _two_band_faint_scene(n=81, r_ap=6.0, ee_reach=25.0, sigma_d=2.0, sigma_b=3.2,
                          c_det=0.9622, c_b=0.9192):
    """Isolated, net-negative (pure-noise) faint source with a REAL matching
    kernel between a narrower detection PSF and a wider band PSF, and two
    PSFRegionMaps carrying distinct containments -- the scene acceptance
    tests 1 and 5 share (docs Sec 5.1/6, ruling "A+cb"). Returns
    ``(cat, pl, r_img, r_ap, psf_det, psf_band)``.
    """
    import geopandas as gpd
    import shapely.geometry as sgeom
    from mophongo.templates import Templates
    from mophongo.psf_map import PSFRegionMap
    from mophongo.psf import PSF

    c = n // 2
    psf_det = _gauss(41, sigma_d)
    psf_band = _gauss(41, sigma_b)
    # Real Fourier-domain matching kernel (mophongo.psf.matching_kernel --
    # Tukey-windowed, the actual production machinery), so detection_PSF (x) K
    # genuinely reproduces the band PSF rather than an idealized assumption.
    kernel = PSF.from_array(psf_det).matching_kernel(psf_band)

    regions = gpd.GeoDataFrame(
        {"psf_key": [0]}, geometry=[sgeom.box(-1e4, -1e4, 1e4, 1e4)], crs=None
    )
    prm_det = PSFRegionMap(regions=regions, psfs=np.array([psf_det]), containment=c_det)
    prm_band = PSFRegionMap(regions=regions, psfs=np.array([psf_band]), containment=c_b)

    image = np.zeros((n, n))
    segmap = np.zeros((n, n), dtype=int)
    segmap[c - 1:c + 2, c - 1:c + 2] = 1
    # Net-negative in-segment sum (IDL positive-pixel clamp -> snr_seg == 0
    # exactly -> w_core == 0, H == M everywhere), one positive pixel so A_src
    # stays well-defined -- same faint-limit construction as
    # test_totcor1_faint_limit_matches_true_total_psf_ee, extended to two bands.
    image[c - 1:c + 2, c - 1:c + 2] = -1.0
    image[c, c] = 3.0
    ivar = np.ones((n, n))

    tmpls = Templates(min_size=n)
    tmpls.extract_templates(
        image, segmap, [(c, c)], extend_mode="auto",
        detection_psf=prm_det, detection_weight=ivar,
        max_radius_pix=ee_reach, psf_ee_radius_pix=ee_reach,
        aperture_radius_pix=r_ap,
        fit_snrlo_psf=10.0, wings_snr_psf=3.0,
    )
    orig = tmpls._templates[0]
    assert orig.snr_seg == 0.0
    assert orig.flux_beyond_aper == pytest.approx(0.0)  # isolated: r_ap fully inside ee_reach

    conv = tmpls.convolve_templates(kernel, inplace=False)[0]

    pl = Pipeline([image, image], segmap)
    pl.psfs = [prm_det, prm_band]
    pl.config.aperture_diam = 2 * r_ap
    pl.config.aperture_units = "pix"
    cat = Table({"id": [1]})
    pl._add_aperture_photometry(
        cat, [conv], np.array([1.0]), np.zeros((n, n)), 1,
        r_orig_pix=r_ap, orig_templates=[orig],
    )
    r_img = pl._resolve_image_ap_radius_pix(1, pl.config)
    return cat, pl, r_img, r_ap, psf_det, psf_band


def test_totcor1_two_band_faint_limit_matches_true_total_psf_ee():
    """Acceptance test 1 (Stage-4b ruling): the two-band generalization of
    ``test_totcor1_faint_limit_matches_true_total_psf_ee`` (which stays,
    single-band/K=identity). With a genuinely wider band PSF, a real
    (non-identity) matching kernel, and DISTINCT detection/band containments,
    the faint (pure-noise) limit must reproduce
    totcor1 == 1/(EE_band_stamp(r_ap) * c_b) -- the band-side containment
    factor, NOT c_det (root cause 1 fixed for the band-frame stamp-vs-band
    containment mismatch, docs Sec 5.1/6). apF_corr (detection side) gets NO
    such factor: it stays the Stage-4 true-total form EE_det_stamp(r_color)*c_det."""
    import mophongo.utils as utils

    c_det, c_b = 0.9622, 0.9192
    cat, pl, r_img, r_ap, psf_det, psf_band = _two_band_faint_scene(c_det=c_det, c_b=c_b)

    totcor1 = cat["totcor1_1"][0]
    expected_totcor1 = 1.0 / (utils.psf_ee_at_radius(psf_band, r_img) * c_b)
    assert totcor1 == pytest.approx(expected_totcor1, rel=2e-3)

    # apF_corr isn't a written column; recover it algebraically from the two
    # that are (apcor1 = apF_corr/apB_corr, totcor1 = 1/apB_corr, so
    # apF_corr = apcor1/totcor1).
    apF_corr = cat["apcor1_1"][0] / cat["totcor1_1"][0]
    expected_apF_corr = utils.psf_ee_at_radius(psf_det, r_ap) * c_det
    assert apF_corr == pytest.approx(expected_apF_corr, rel=1e-6)


def test_apcor1_gains_band_containment_ratio():
    """Acceptance test 5 (Stage-4b ruling): the band containment factor must
    NOT cancel in apcor1 -- it multiplies apcor1 by c_det/c_b, a genuine shape
    effect of the stamp-containment mismatch between the detection and band
    PSFs (contrast with trunc, which DOES cancel --
    test_truncation_cancels_in_apcor1_survives_in_totcor1)."""
    import mophongo.utils as utils

    c_det, c_b = 0.9622, 0.9192
    cat, pl, r_img, r_ap, psf_det, psf_band = _two_band_faint_scene(c_det=c_det, c_b=c_b)

    apcor1 = cat["apcor1_1"][0]
    expected = (
        (utils.psf_ee_at_radius(psf_det, r_ap) * c_det)
        / (utils.psf_ee_at_radius(psf_band, r_img) * c_b)
    )
    assert apcor1 == pytest.approx(expected, rel=2e-3)


def test_asrc_invariance_of_totcor1_with_data_core_and_crowded_wings():
    """Acceptance test 2 (Stage-4b ruling): totcor1 evaluated on the corrected
    model is invariant to the positive-clip amplitude A_src of the PSF-wing
    component -- inflating A_src (e.g. from neighbour-wing light or clipped
    noise, docs Sec "risks"/ruling diagnosis) slides the wing toward the
    pure-PSF limit without moving totcor1, because Sigma(H_corr)+fb_corr and
    the aperture-sum numerator both scale the SAME way with A_src. This is
    the property that kills the crowded tail (a masked-Sigma(H) vs
    unmasked-fb mismatch inflates totcor1 with A_src on unfixed Stage-4 code).

    Construction: a FIXED, real bright compact data core (not PSF-scaled) plus
    PSF wings of amplitude A_src, crowded so the wings are cut off on one side
    well inside the aperture (own territory only spans half the aperture
    disk) -- this is exactly the "data core (w_core~1) + noise-floor wings
    (w_k~0)" regime the ruling's diagnosis describes for the real MIRI-faint
    population (snr_seg measured on deep F444W keeps w_core~1 while the owned
    halo is noise-dominated, w_k~0). template_norm/flux_beyond_stamp differ by
    ~5x between the two A_src values (materially), yet totcor1 must agree to
    <=1% (vs >2% on unfixed Stage-4 code for the same construction)."""
    from photutils.aperture import CircularAperture, aperture_photometry

    n = 61
    c = n // 2
    r_ap = 6.0
    ee_reach = 15.0
    c_det = 0.95

    yy, xx = np.mgrid[0:n, 0:n]
    r2 = (xx - c) ** 2 + (yy - c) ** 2
    psf_cut = _gauss(n, 2.0)  # unit-sum "own stamp" detection PSF, resampled

    own = r2 <= 2.0 ** 2  # tiny bright compact core (r<=2)
    halo = (~own) & (r2 <= ee_reach ** 2)
    # Crowding: a neighbour on the x>c side claims that half of the halo, so
    # the fit support (ext_psf) excludes it even though it is within ee_reach.
    ext_psf = own | (halo & (xx <= c))

    # Real, FIXED bright compact core (not PSF-scaled): a raw (PEAK, not
    # unit-sum) amplitude=50 Gaussian, so the core's total flux genuinely
    # dominates over the modest A_src wing amplitudes tried below.
    data_core = 50.0 * np.exp(-r2 / (2 * 1.3 ** 2)) * own

    f_cut_support = float(psf_cut[ext_psf].sum())
    aper = CircularAperture((c, c), r=r_ap)
    aper_full = float(aperture_photometry(psf_cut, aper, method="exact")["aperture_sum"][0])
    ext_arr = psf_cut * ext_psf
    aper_ext = float(aperture_photometry(ext_arr, aper, method="exact")["aperture_sum"][0])

    def build(A_src):
        # W==1 in own (data core), W==0 in the halo (pure PSF wings) -- the
        # "bright compact" limit (docs Sec 2.1), zero outside ext_psf by
        # construction (own is a subset of ext_psf).
        H = data_core + A_src * psf_cut * (ext_psf & ~own)
        H = np.where(ext_psf, H, 0.0)
        template_norm = float(H.sum())
        flux_beyond_stamp = max(A_src * (1.0 / c_det - f_cut_support), 0.0)
        flux_beyond_aper = max(A_src * (aper_full - aper_ext), 0.0)
        unit = H / template_norm
        orig = Template(unit.copy(), (c, c), (n, n), label=1)
        orig.template_norm = template_norm
        orig.flux_beyond_stamp = flux_beyond_stamp
        orig.flux_beyond_aper = flux_beyond_aper
        conv = Template(unit.copy(), (c, c), (n, n), label=1)
        conv.template_norm = template_norm
        return orig, conv, template_norm, flux_beyond_stamp

    results = {}
    for A_src in (5.0, 30.0):
        orig, conv, tn, fb = build(A_src)
        pl = Pipeline([np.zeros((n, n))], np.zeros((n, n), dtype=int))
        pl.psfs = [np.ones((5, 5))]  # containment fallback 1.0 both sides -- isolates the delta
        pl.config.aperture_diam = 2 * r_ap
        pl.config.aperture_units = "pix"
        cat = Table({"id": [1]})
        pl._add_aperture_photometry(cat, [conv], np.array([1.0]), np.zeros((n, n)), 1,
                                    r_orig_pix=r_ap, orig_templates=[orig])
        results[A_src] = dict(tn=tn, fb=fb, totcor1=cat["totcor1_1"][0])

    # template_norm/flux_beyond_stamp differ materially (>3x) between the two
    # A_src values...
    assert results[30.0]["tn"] / results[5.0]["tn"] > 1.02
    assert results[30.0]["fb"] / results[5.0]["fb"] > 3.0
    # ...yet totcor1 (the corrected aperture-to-total) agrees to <=1%.
    t5, t30 = results[5.0]["totcor1"], results[30.0]["totcor1"]
    assert abs(t30 - t5) / t5 <= 0.01


def test_crowding_regression_totcor1_matches_isolated():
    """Acceptance test 3 (Stage-4b ruling): a faint source plus a 40x brighter
    neighbour ~15-25 px away must give totcor1 within 3% of the SAME source
    measured in isolation (real two-segment ownership machinery, i.e. the
    genuine ``Templates._build_ownership`` area contest, not a hand-built
    mask). On unfixed Stage-4 code the SAME construction inflates totcor1 by
    ~1.6x (docs Sec 4.3's diagnosed tail; ruling table)."""
    import geopandas as gpd
    import shapely.geometry as sgeom
    from mophongo.templates import Templates
    from mophongo.psf_map import PSFRegionMap

    n = 121
    sigma = 2.0
    psf = _gauss(41, sigma)
    regions = gpd.GeoDataFrame(
        {"psf_key": [0]}, geometry=[sgeom.box(-1e4, -1e4, 1e4, 1e4)], crs=None
    )
    prm = PSFRegionMap(regions=regions, psfs=np.array([psf]), containment=0.95)
    r_ap = 5.0
    ee_reach = 20.0
    ty, tx = 60, 40

    def run_scene(sep):
        image = np.zeros((n, n))
        segmap = np.zeros((n, n), dtype=int)
        segmap[ty - 1:ty + 2, tx - 1:tx + 2] = 1  # faint target: net-negative 3x3 segment
        image[ty - 1:ty + 2, tx - 1:tx + 2] = -1.0
        image[ty, tx] = 3.0
        ivar = np.ones((n, n))
        positions = [(tx, ty)]
        if sep is not None:
            nb = _gauss(n, 3.0) * (40 * 3.0 * (2 * np.pi * 3.0 ** 2))  # 40x brighter peak
            image += nb
            segmap[nb > nb.max() * np.exp(-0.5)] = 2  # ~1-sigma isophote segment
            positions.append((tx + sep, ty))

        tmpls = Templates(min_size=n)
        tmpls.extract_templates(
            image, segmap, positions, extend_mode="auto",
            detection_psf=prm, detection_weight=ivar,
            max_radius_pix=ee_reach, psf_ee_radius_pix=ee_reach,
            aperture_radius_pix=r_ap,
            fit_snrlo_psf=10.0, wings_snr_psf=3.0,
        )
        orig = tmpls._templates[0]
        assert orig.snr_seg == 0.0

        pl = Pipeline([image], segmap)
        pl.psfs = [prm]
        pl.config.aperture_diam = 2 * r_ap
        pl.config.aperture_units = "pix"
        cat = Table({"id": [1]})
        pl._add_aperture_photometry(cat, [orig], np.array([1.0]), np.zeros((n, n)), 1,
                                    r_orig_pix=r_ap, orig_templates=[orig])
        return cat["totcor1_1"][0]

    totcor1_iso = run_scene(None)
    for sep in (15, 20, 25):
        totcor1_crowded = run_scene(sep)
        assert totcor1_crowded == pytest.approx(totcor1_iso, rel=0.03), (
            f"sep={sep}: crowded totcor1 {totcor1_crowded} vs isolated {totcor1_iso}"
        )


def test_fit_invariant_to_containment_perturbation():
    """Acceptance test 4 (Stage-4b ruling): perturbing the containments that
    feed the band-side c_b/c_det factor must leave the FIT-side bookkeeping
    (ap_model, ap_flux, res_sum -- and, by construction, the fitted flux_1
    that feeds them, unchanged here since the same ``fluxes`` array is reused)
    bit-identical; only totcor1/apcor1 (and, downstream, apcor/tcor_int/est3*,
    which are algebraic functions of apcor1/totcor1) move."""
    import geopandas as gpd
    import shapely.geometry as sgeom
    from mophongo.templates import Templates
    from mophongo.psf_map import PSFRegionMap

    n = 61
    c = n // 2
    psf = _gauss(31, 2.0)
    r_ap = 5.0
    ee_reach = 18.0

    image = np.zeros((n, n))
    segmap = np.zeros((n, n), dtype=int)
    segmap[c - 1:c + 2, c - 1:c + 2] = 1
    image[c - 1:c + 2, c - 1:c + 2] = -1.0
    image[c, c] = 3.0
    ivar = np.ones((n, n))

    regions = gpd.GeoDataFrame(
        {"psf_key": [0]}, geometry=[sgeom.box(-1e4, -1e4, 1e4, 1e4)], crs=None
    )
    prm_extract = PSFRegionMap(regions=regions, psfs=np.array([psf]), containment=0.9622)
    tmpls = Templates(min_size=n)
    tmpls.extract_templates(
        image, segmap, [(c, c)], extend_mode="auto",
        detection_psf=prm_extract, detection_weight=ivar,
        max_radius_pix=ee_reach, psf_ee_radius_pix=ee_reach,
        aperture_radius_pix=r_ap,
        fit_snrlo_psf=10.0, wings_snr_psf=3.0,
    )
    orig = tmpls._templates[0]

    rows = {}
    for c_det_ratio in (0.9622, 1.0):
        prm_ratio = PSFRegionMap(regions=regions, psfs=np.array([psf]), containment=c_det_ratio)
        pl = Pipeline([image], segmap)
        pl.psfs = [prm_ratio]  # only self.psfs[0] (c_det) perturbed; SAME fluxes/templates
        pl.config.aperture_diam = 2 * r_ap
        pl.config.aperture_units = "pix"
        cat = Table({"id": [1]})
        pl._add_aperture_photometry(cat, [orig], np.array([1.0]), np.zeros((n, n)), 1,
                                    r_orig_pix=r_ap, orig_templates=[orig])
        rows[c_det_ratio] = dict(cat[0])

    a, b = rows[0.9622], rows[1.0]
    for col in ("ap_model_1", "ap_flux_1", "res_sum_1"):
        assert a[col] == b[col], f"{col} must be bit-identical under a containment perturbation"
    assert a["totcor1_1"] != pytest.approx(b["totcor1_1"])
    assert a["apcor1_1"] != pytest.approx(b["apcor1_1"])


def test_band_ee_uses_native_pixel_scale_not_fit_grid():
    """Stage-4b BLOCKING fix: the band-side EE (band_det_ratio) must be
    evaluated on the band PSF's NATIVE pixel grid, not the (possibly upsampled)
    fit grid. In the default multi_resolution_method='upsample' path, the fit
    loop rewrites wcs[idx]=wcs[0], so r_img_pix ends up in fine detection
    pixels (~0.04\") while self.psfs[idx] is stored on its coarse native grid
    (~0.11\"/px); measuring the band EE at the fine radius samples the wrong
    physical radius (EE -> ~1), inflating band_det_ratio, overestimating
    flux_beyond_aper_band and depressing totcor1 for exactly the crowded faint
    sources this stage fixes. The pipeline converts via the captured
    ``self._native_pscale`` (commit 716811b Stage-2 pattern):
    r_band = r_orig_pix * _native_pscale[0] / _native_pscale[idx].

    This is the ONLY test in the suite that exercises the band-EE radius:
    band_det_ratio enters totcor1 only through flux_beyond_aper_band, which is
    ZERO for isolated sources -- so a CROWDED source (flux_beyond_aper > 0) is
    required. flux_beyond_stamp/flux_beyond_aper are set directly here (rather
    than via extraction) so the band-EE frame is isolated as the single degree
    of freedom. Revert-verify: this FAILS against the unconverted r_img_pix
    band-EE call (which would use EE_band(r_img) ~ 1 instead of
    EE_band(r_native))."""
    import geopandas as gpd
    import shapely.geometry as sgeom
    from mophongo.psf_map import PSFRegionMap
    import mophongo.utils as utils

    n = 61
    c = n // 2
    r_orig = 6.0                       # reference (detection 0.04") grid aperture radius
    # Coarse band native grid: 0.11"/px vs the 0.04" fit grid (factor 2.75),
    # exactly the F1500W-vs-F444W upsample setup the reviewer measured.
    psn = [0.04, 0.11]
    r_native = r_orig * psn[0] / psn[1]  # ~2.18 native px -- the physically correct radius

    psf_det = _gauss(21, 2.0)          # detection PSF, reference grid
    psf_band_native = _gauss(21, 2.5)  # band PSF, coarse native grid (self.psfs[1])
    # EE differs hugely between the two radii -> strong discrimination:
    assert utils.psf_ee_at_radius(psf_band_native, r_native) < 0.5
    assert utils.psf_ee_at_radius(psf_band_native, r_orig) > 0.9  # the wrong (fit-grid) value

    regions = gpd.GeoDataFrame(
        {"psf_key": [0]}, geometry=[sgeom.box(-1e4, -1e4, 1e4, 1e4)], crs=None
    )
    c_det, c_b = 0.9622, 0.9192
    prm_det = PSFRegionMap(regions=regions, psfs=np.array([psf_det]), containment=c_det)
    prm_band = PSFRegionMap(regions=regions, psfs=np.array([psf_band_native]), containment=c_b)

    # Crowded faint source: substantial flux_beyond_aper (own PSF tail the fit
    # support excluded) so band_det_ratio genuinely enters totcor1.
    TN, FBS, FBA = 5.0, 2.0, 1.5
    prof = _gauss(n, 2.5)
    conv = Template(prof.copy(), (c, c), (n, n), label=1); conv.template_norm = TN
    orig = Template(prof.copy(), (c, c), (n, n), label=1)
    orig.template_norm = TN
    orig.flux_beyond_stamp = FBS
    orig.flux_beyond_aper = FBA

    pl = Pipeline([np.zeros((n, n)), np.zeros((n, n))], np.zeros((n, n), dtype=int))
    pl.psfs = [prm_det, prm_band]
    pl._native_pscale = psn        # as captured by run() BEFORE the upsample step
    pl.config.aperture_diam = 2 * r_orig  # r_img_pix == 6.0 (fine fit px, == r_orig here)
    pl.config.aperture_units = "pix"
    cat = Table({"id": [1]})
    pl._add_aperture_photometry(cat, [conv], np.array([1.0]), np.zeros((n, n)), 1,
                                r_orig_pix=r_orig, orig_templates=[orig])
    totcor1 = cat["totcor1_1"][0]

    # Exact reconstruction from the documented Stage-4b formula, with the band
    # EE evaluated on the NATIVE grid (r_native). Any use of r_img (=6.0) for
    # the band EE inflates band_det_ratio ~3x and breaks this to ~15%.
    apB_book = pl._aperture_sum_on_template(conv, r_orig)  # r_img == r_orig here
    trunc_denom = TN + FBS
    trunc = TN / trunc_denom
    ee_band = utils.psf_ee_at_radius(psf_band_native, r_native)
    ee_det = utils.psf_ee_at_radius(psf_det, r_orig)
    flux_beyond_aper_band = FBA * (ee_band / ee_det)
    apB_corr = (apB_book * trunc + flux_beyond_aper_band / trunc_denom) * (c_b / c_det)
    expected_totcor1 = 1.0 / apB_corr
    assert totcor1 == pytest.approx(expected_totcor1, rel=1e-6)

    # Guard: the wrong (fit-grid) radius would give a materially different value.
    ee_band_wrong = utils.psf_ee_at_radius(psf_band_native, r_orig)
    apB_corr_wrong = (apB_book * trunc + FBA * (ee_band_wrong / ee_det) / trunc_denom) * (c_b / c_det)
    assert abs(1.0 / apB_corr_wrong - expected_totcor1) / expected_totcor1 > 0.10


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
