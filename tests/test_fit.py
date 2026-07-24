import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from mophongo.fit import FitConfig, SparseFitter, build_normal_matrix
from mophongo.psf import PSF
from mophongo.templates import Templates, Template
from utils import make_simple_data, save_fit_diagnostic


def test_flux_recovery(tmp_path):
    images, segmap, catalog, psfs, truth_img, rms = make_simple_data()

    psf_hi = PSF.from_array(psfs[0])
    psf_lo = PSF.from_array(psfs[1])
    kernel = psf_hi.matching_kernel(psf_lo)

    tmpls = Templates.from_image(
        images[0], segmap, list(zip(catalog["x"], catalog["y"])), kernel
    )

    fitter = SparseFitter(tmpls.templates, images[1], 1.0 / rms[1] ** 2, FitConfig())
    # build_normal_matrix() is no longer a bound method (fit.py:1643 is now a
    # module-level free function); the method-based builder is build_normal_tree().
    fitter.build_normal_tree()
    x, err, info = fitter.solve()

    # Default FitConfig has fit_astrometry_niter=2 and fit_astrometry_joint=True,
    # so solve() routes through the per-scene joint flux+shift solver, whose
    # per-scene CG flags live at info["cg_info"]["cg_info"] (a list, one per
    # scene), not a single top-level int.
    cg_flags = info["cg_info"]["cg_info"]
    assert all(flag == 0 for flag in cg_flags)
#    assert np.allclose(x, np.array(catalog['flux_true']), rtol=1e-1)
    model = fitter.model_image()
    fname = tmp_path / "fit.png"
    save_fit_diagnostic(fname, images[1], model, fitter.residual())

    assert fname.exists()


@pytest.mark.xfail(
    reason="SparseFitter has no solve_lo() method -- the linear-operator solve "
    "path ('lo' in FitConfig.solve_method's docstring) was never implemented "
    "on the class, this is not a rename. See docs/test_suite_cleanup_plan.md B2/A6.",
    strict=False,
)
def test_lsqr_lo_matches_cg():
    images, segmap, catalog, psfs, _, rms = make_simple_data()

    psf_hi = PSF.from_array(psfs[0])
    psf_lo = PSF.from_array(psfs[1])
    kernel = psf_hi.matching_kernel(psf_lo)

    tmpls1 = Templates.from_image(
        images[0], segmap, list(zip(catalog["x"], catalog["y"])), kernel
    )
    fitter_lo = SparseFitter(tmpls1.templates, images[1], 1.0 / rms[1] ** 2, FitConfig())
    flux_lo, err_lo, _ = fitter_lo.solve_lo()

    tmpls2 = Templates.from_image(
        images[0], segmap, list(zip(catalog["x"], catalog["y"])), kernel
    )
    fitter_cg = SparseFitter(tmpls2.templates, images[1], 1.0 / rms[1] ** 2, FitConfig())
    flux_cg, err_cg, _ = fitter_cg.solve()

    np.testing.assert_allclose(flux_lo, flux_cg, rtol=2e-3, atol=2e-3)
    assert err_lo.shape == err_cg.shape


def test_ata_symmetry():
    images, segmap, catalog, psfs, _, rms = make_simple_data()

    psf_hi = PSF.from_array(psfs[0])
    psf_lo = PSF.from_array(psfs[1])
    kernel = psf_hi.matching_kernel(psf_lo)

    tmpls = Templates.from_image(
        images[0], segmap, list(zip(catalog["x"], catalog["y"])), kernel
    )
    fitter = SparseFitter(tmpls.templates, images[1], 1.0 / rms[1] ** 2, FitConfig())
    fitter.build_normal_tree()
    ata = fitter.ata.toarray()
    assert np.allclose(ata, ata.T)


def test_zero_weight_template_dropped():
    img = np.zeros((4, 4))
    weights = np.ones_like(img)
    weights[2:4, 2:4] = 0

    t1 = Template(img, (1, 1), (2, 2))
    t1.data[:] = 1.0
    t2 = Template(img, (3, 3), (2, 2))
    t2.data[:] = 1.0

    fitter = SparseFitter([t1, t2], img, weights, FitConfig())
    # build_normal_tree() keeps low-norm templates (just logs a warning);
    # only the loop-based free function build_normal_matrix() actually drops
    # them, so use that here to preserve the test's intent.
    build_normal_matrix(fitter)

    assert len(fitter.templates) == 1


def test_flux_errors_regularized():
    img = np.zeros((3, 3))
    weights = np.ones_like(img)
    tmpl_data = np.zeros((3, 3))
    tmpl_data[1, 1] = 1.0

    t1 = Template(img, (1, 1), (3, 3))
    t1.data[:] = tmpl_data
    t2 = Template(img, (1, 1), (3, 3))
    tmpl_data2 = tmpl_data.copy()
    tmpl_data2[0, 0] = 0.1  # break perfect degeneracy
    t2.data[:] = tmpl_data2

    fitter = SparseFitter([t1, t2], img, weights, FitConfig())
    fitter.build_normal_tree()
    _, _, _ = fitter.solve()
    err = fitter.flux_errors()

    assert err.size == 2
    assert np.all(np.isfinite(err))


def test_flux_and_rms_estimation():
    """SparseFitter.flux_and_rms matches quick flux and error estimates."""
    images, segmap, catalog, psfs, _, rms = make_simple_data()

    tmpls = Templates.from_image(
        images[0], segmap, list(zip(catalog["x"], catalog["y"])), kernel=None
    )

    fitter = SparseFitter(tmpls.templates, images[1], 1.0 / rms[1] ** 2, FitConfig())

    flux, err = fitter.flux_and_rms()
    np.testing.assert_allclose(flux, fitter.quick_flux())
    np.testing.assert_allclose(err, fitter.predicted_errors())

    for t in tmpls.templates:
        t.flux = 42.0
    flux2, _ = fitter.flux_and_rms()
    assert np.all(flux2 == 42.0)


def test_build_normal_tree_matches_loop():
    images, segmap, catalog, psfs, _, rms = make_simple_data()

    psf_hi = PSF.from_array(psfs[0])
    psf_lo = PSF.from_array(psfs[1])
    kernel = psf_hi.matching_kernel(psf_lo)

    tmpls = Templates.from_image(
        images[0], segmap, list(zip(catalog["x"], catalog["y"])), kernel
    )
    fitter_loop = SparseFitter(
        tmpls.templates, images[1], 1.0 / rms[1] ** 2, FitConfig()
    )
    fitter_tree = SparseFitter(
        tmpls.templates, images[1], 1.0 / rms[1] ** 2, FitConfig(normal="tree")
    )

    build_normal_matrix(fitter_loop)
    fitter_tree.build_normal_tree()

    np.testing.assert_allclose(
        fitter_loop.ata.toarray(), fitter_tree._ata.toarray()
    )
    np.testing.assert_allclose(fitter_loop.atb, fitter_tree._atb)


def test_solve_scene_matches_global():
    img = np.zeros((6, 6))
    weights = np.ones_like(img)

    t1 = Template(img, (2, 2), (3, 3))
    t1.data[:] = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    t2 = Template(img, (2, 3), (3, 3))
    t2.data[:] = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    t3 = Template(img, (5, 5), (1, 1))
    t3.data[:] = np.array([[1]])

    tmpls = [t1, t2, t3]
    fluxes = [1.0, 2.0, 3.0]
    image = np.zeros_like(img)
    for f, t in zip(fluxes, tmpls):
        image[t.slices_original] += f * t.data[t.slices_cutout]

    # Global solve
    fitter_all = SparseFitter(
        [Template(img, (2, 2), (3, 3)), Template(img, (2, 3), (3, 3)), Template(img, (5, 5), (1, 1))],
        image,
        weights,
        FitConfig(),
    )
    fitter_all.templates[0].data[:] = t1.data
    fitter_all.templates[1].data[:] = t2.data
    fitter_all.templates[2].data[:] = t3.data
    flux_all, _, _ = fitter_all.solve()

    # Component solve
    fitter_comp = SparseFitter(
        [Template(img, (2, 2), (3, 3)), Template(img, (2, 3), (3, 3)), Template(img, (5, 5), (1, 1))],
        image,
        weights,
        FitConfig(),
    )
    fitter_comp.templates[0].data[:] = t1.data
    fitter_comp.templates[1].data[:] = t2.data
    fitter_comp.templates[2].data[:] = t3.data
    flux_comp, _, _ = fitter_comp.solve_scene()

    np.testing.assert_allclose(flux_comp, flux_all, rtol=1e-6, atol=1e-6)


# NOTE: the old test_build_normal_matrix_new_equivalence() was already a
# no-op (an unconditional `return` on the first line) and referenced APIs
# that don't exist anywhere in the current source: `extract_templates` isn't
# a standalone importable function (templates.py:1405 has a method of the
# same name on a different class), and SparseFitter has no
# `build_normal_matrix_new`/`model_image_new`. There is nothing left to
# adapt it to, so the dead test has been removed (see plan B2).


@pytest.mark.xfail(
    reason="SparseFitter.bright_mask is never set: __init__ computes snr but "
    "the assignment is commented out (fit.py:751, `self.orig_bright = ...`), "
    "and no other code path defines a `bright_mask` attribute (the only other "
    "reference, fit.py:1582, is dead code inside an unreachable nested def). "
    "This is a real gap, not a rename. See docs/test_suite_cleanup_plan.md B2/A6.",
    strict=False,
)
def test_bright_source_detection():
    images, segmap, catalog, psfs, _, rms = make_simple_data()
    tmpls = Templates.from_image(
        images[0], segmap, list(zip(catalog["x"], catalog["y"])), kernel=None
    )
    cfg = FitConfig(snr_thresh_astrom=5.0)
    fitter = SparseFitter(
        tmpls.templates, images[1], 1.0 / rms[1] ** 2, cfg
    )
    flux = Templates.quick_flux(tmpls.templates, images[1])
    err = Templates.predicted_errors(tmpls.templates, 1.0 / rms[1] ** 2)
    snr = flux / err
    expected = snr > cfg.snr_thresh_astrom
    assert np.array_equal(fitter.bright_mask, expected)


def test_solve_scene_shifts_matches_global():
    img = np.zeros((6, 6))
    weights = np.ones_like(img)

    t1 = Template(img, (2, 2), (3, 3))
    t1.data[:] = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    t2 = Template(img, (2, 3), (3, 3))
    t2.data[:] = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    t3 = Template(img, (5, 5), (1, 1))
    t3.data[:] = np.array([[1]])

    tmpls = [t1, t2, t3]
    fluxes = [1.0, 2.0, 3.0]
    image = np.zeros_like(img)
    for f, t in zip(fluxes, tmpls):
        image[t.slices_original] += f * t.data[t.slices_cutout]

    fitter_all = SparseFitter(
        [Template(img, (2, 2), (3, 3)), Template(img, (2, 3), (3, 3)), Template(img, (5, 5), (1, 1))],
        image,
        weights,
        FitConfig(),
    )
    fitter_all.templates[0].data[:] = t1.data
    fitter_all.templates[1].data[:] = t2.data
    fitter_all.templates[2].data[:] = t3.data
    flux_all, _, _ = fitter_all.solve()

    fitter_shift = SparseFitter(
        [Template(img, (2, 2), (3, 3)), Template(img, (2, 3), (3, 3)), Template(img, (5, 5), (1, 1))],
        image,
        weights,
        FitConfig(snr_thresh_astrom=0.0),
    )
    fitter_shift.templates[0].data[:] = t1.data
    fitter_shift.templates[1].data[:] = t2.data
    fitter_shift.templates[2].data[:] = t3.data
    # SparseFitter.solve_scene_shifts() is dead code: it is defined *after* an
    # unreachable `return` inside the free function merge_small_scenes_old
    # (fit.py:1557-1559), so it is nested there, not a method of SparseFitter,
    # and was never reachable as `fitter.solve_scene_shifts(...)`. The live
    # equivalent (joint flux+shift solve per scene) is
    # SparseFitter.solve()/solve_scene() with fit_astrometry_niter>0 and
    # fit_astrometry_joint=True (both default), which is exactly the path
    # FitConfig(snr_thresh_astrom=0.0) below exercises.
    flux_shift, _, info = fitter_shift.solve()
    betas = info["cg_info"]["betas"]

    np.testing.assert_allclose(flux_shift, flux_all, rtol=1e-4, atol=1e-4)
    for beta_entry in betas:
        beta = beta_entry[1]
        assert np.all(np.abs(beta) < 1e-6)


# ---------------------------------------------------------------------------
# B6 coverage: regularization-no-bias, solve_method guard, flux-error orientation
# ---------------------------------------------------------------------------


def test_regularization_does_not_bias_flux():
    """An isolated source's recovered flux should be ~independent of `reg`.

    Guards commit 9d2ed2d, which fixed a flux bias introduced by
    regularization. Uses a single isolated Gaussian-like template so the
    analytic answer is just b/d (weighted least squares with one unknown).
    """
    img = np.zeros((11, 11))
    weights = np.ones_like(img)

    yy, xx = np.mgrid[0:5, 0:5]
    sigma = 1.0
    psf = np.exp(-((xx - 2) ** 2 + (yy - 2) ** 2) / (2 * sigma**2))
    psf /= psf.sum()

    true_flux = 7.3
    t = Template(img, (5, 5), (5, 5))
    t.data[:] = psf
    image = np.zeros_like(img)
    image[t.slices_original] += true_flux * t.data[t.slices_cutout]

    b = np.sum(t.data[t.slices_cutout] * weights[t.slices_original] * image[t.slices_original])
    d = np.sum(t.data[t.slices_cutout] * weights[t.slices_original] * t.data[t.slices_cutout])
    analytic = b / d
    np.testing.assert_allclose(analytic, true_flux, rtol=1e-6)

    # Typical/small regularization strengths (including the config default,
    # reg=0.0 -> auto 1e-6*median(diag)) should leave the flux essentially
    # unbiased.
    for reg in (0.0, 1e-8, 1e-6, 1e-4):
        tmpl = Template(img, (5, 5), (5, 5))
        tmpl.data[:] = psf
        fitter = SparseFitter([tmpl], image, weights, FitConfig(reg=reg))
        x, _, info = fitter.solve()
        rel_err = abs(x[0] - analytic) / analytic
        assert rel_err < 1e-2, f"reg={reg}: relative flux bias {rel_err} too large"


def test_solve_method_scene_is_supported():
    """SparseFitter.solve() with the default/supported 'scene' method works."""
    images, segmap, catalog, psfs, _, rms = make_simple_data(nsrc=5, size=51, seed=3)
    tmpls = Templates.from_image(
        images[0], segmap, list(zip(catalog["x"], catalog["y"])), kernel=None
    )
    n = len(tmpls.templates)
    fitter = SparseFitter(
        tmpls.templates, images[1], 1.0 / rms[1] ** 2, FitConfig(solve_method="scene")
    )
    x, err, info = fitter.solve()
    assert x.shape == (n,)
    assert err.shape == (n,)
    assert np.all(np.isfinite(x))


@pytest.mark.xfail(
    reason="SparseFitter.solve() dispatches any solve_method other than 'scene' "
    "to self.solve_all(), which is not defined anywhere on the class -> "
    "AttributeError. See docs/test_suite_cleanup_plan.md A6 ('solve_method guard').",
    strict=False,
)
def test_solve_method_all_not_supported():
    img = np.zeros((4, 4))
    weights = np.ones_like(img)
    t = Template(img, (1, 1), (2, 2))
    t.data[:] = 1.0
    fitter = SparseFitter([t], img, weights, FitConfig(solve_method="all"))
    x, err, info = fitter.solve()
    assert x.shape == (1,)


@pytest.mark.xfail(
    reason="SparseFitter._flux_errors returns sqrt(diag)*covar_power, while "
    "SceneFitter._flux_errors returns 1/sqrt(diag) -- inverse conventions on "
    "the same whitened system. For a single isolated (perfectly diagonal) "
    "template, covar_power collapses to 0 and SparseFitter reports a flux "
    "error of exactly 0.0, whereas the analytic 1-sigma error for weighted "
    "least squares is 1/sqrt(d). Only one of these can be right; pinning the "
    "physically-correct analytic value here documents the discrepancy rather "
    "than silently accepting either. See docs/test_suite_cleanup_plan.md A6.",
    strict=False,
)
def test_flux_error_orientation_matches_analytic():
    img = np.zeros((11, 11))
    weights = np.ones_like(img)

    yy, xx = np.mgrid[0:5, 0:5]
    sigma = 1.0
    psf = np.exp(-((xx - 2) ** 2 + (yy - 2) ** 2) / (2 * sigma**2))
    psf /= psf.sum()

    t = Template(img, (5, 5), (5, 5))
    t.data[:] = psf
    image = np.zeros_like(img)
    image[t.slices_original] += 7.3 * t.data[t.slices_cutout]

    d = np.sum(t.data[t.slices_cutout] * weights[t.slices_original] * t.data[t.slices_cutout])
    analytic_sigma = 1.0 / np.sqrt(d)

    fitter = SparseFitter([t], image, weights, FitConfig())
    fitter.solve()
    err = fitter.flux_errors()
    np.testing.assert_allclose(err[0], analytic_sigma, rtol=0.05)
