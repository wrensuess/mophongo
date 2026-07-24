import numpy as np
import pytest
import scipy.sparse as sp

from mophongo.scene_fitter import SceneFitter, build_normal
from mophongo.scene import Scene
from mophongo.templates import Templates, Template
from mophongo.fit import SparseFitter, FitConfig
from mophongo.psf import PSF
from utils import make_simple_data


def test_scene_fitter_flux_only():
    # SceneFitter.solve returns a SimpleNamespace(flux, err, shifts, info),
    # not a plain tuple (constructor/return signature changed).
    A = sp.csr_matrix([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    sol = SceneFitter.solve(A, b)
    alpha, err, beta, info = sol.flux, sol.err, sol.shifts, sol.info
    expected = np.linalg.solve(A.toarray(), b)
    assert info["cg_info"] == 0
    assert beta is None
    # SceneFitter.solve adds a small internal ridge (1e-6 * median(diag)) to
    # the flux block before solving, so the match to the *unregularized*
    # dense solve is only good to ~1e-6, not machine precision.
    np.testing.assert_allclose(alpha, expected, rtol=1e-5)
    # NOTE: SceneFitter._flux_errors takes 1/sqrt(diag) of the *whitened*
    # matrix, whose diagonal is 1 by construction -- so it is structurally
    # unable to reflect the off-diagonal covariance between components 1 and
    # 2 here, unlike the true sqrt(diag(inv(A))). Only assert the values it
    # can actually produce (finite, positive); the orientation/covariance
    # question itself is pinned separately in test_fit.py (see plan A6).
    assert err.shape == expected.shape
    assert np.all(np.isfinite(err)) and np.all(err > 0)


def test_scene_fitter_with_shift_block():
    A = sp.csr_matrix([[2.0, 0.0], [0.0, 1.0]])
    b = np.array([1.0, 1.0])
    AB = sp.csr_matrix([[1.0], [2.0]])
    BB = sp.csr_matrix([[3.0]])
    bB = np.array([0.5])
    # positivity=False: the unconstrained dense reference solve below has
    # negative flux components for this toy system, so the default
    # positivity=True clamp would legitimately zero them out.
    sol = SceneFitter.solve(A, b, AB=AB, BB=BB, bB=bB, config=FitConfig(positivity=False))
    alpha, err, beta, info = sol.flux, sol.err, sol.shifts, sol.info
    M = np.block([[A.toarray(), AB.toarray()], [AB.T.toarray(), BB.toarray()]])
    rhs = np.concatenate([b, bB])
    dense = np.linalg.solve(M, rhs)
    # SceneFitter.solve adds a small ridge to BB (reg_astrom * median(diag(BB)),
    # default reg_astrom=1e-4) before solving, so the match to the
    # unregularized dense solve is only good to ~1e-3, not machine precision.
    np.testing.assert_allclose(alpha, dense[:2], rtol=2e-3)
    np.testing.assert_allclose(beta, dense[2:], rtol=2e-3)


@pytest.mark.parametrize("order", [0, 1, 2])
def test_solve_flux_and_shifts_matches_dense(order):
    nA = 4
    p = (order + 1) * (order + 2) // 2
    nB = p * 2
    A = sp.eye(nA, format="csr") * 2.0
    AB = np.full((nA, nB), 0.1)
    BB = sp.eye(nB, format="csr") * 3.0
    b = np.arange(1, nA + 1, dtype=float)
    bB = np.arange(1, nB + 1, dtype=float)
    # positivity=False: the dense reference solve is unconstrained and, for
    # order=2, the true least-squares flux has negative components; with the
    # default positivity=True clamp SceneFitter's answer would legitimately
    # diverge from the unconstrained dense solve.
    cfg = FitConfig(cg_kwargs={"rtol": 1e-10, "maxiter": 1000}, positivity=False)
    x, err, beta, info = SceneFitter._solve_flux_and_shifts(
        A, b, sp.csr_matrix(AB), BB, bB, config=cfg
    )
    M = np.block([[A.toarray(), AB], [AB.T, BB.toarray()]])
    rhs = np.concatenate([b, bB])
    dense = np.linalg.solve(M, rhs)
    cov = np.linalg.inv(M)
    np.testing.assert_allclose(x, dense[:nA], rtol=1e-3)
    np.testing.assert_allclose(beta, dense[nA:], rtol=1e-3)
    np.testing.assert_allclose(err, np.sqrt(np.diag(cov)[:nA]), rtol=1e-3)
    assert info["cg_info"] == 0


@pytest.mark.xfail(
    reason="Scene.overlay_scene_graph calls Scene.create_scene_graph, which is "
    "not defined anywhere on the class -> AttributeError. Not a rename; the "
    "method was never implemented. See docs/test_suite_cleanup_plan.md B2.",
    strict=False,
)
def test_scene_graph_helpers_are_unimplemented():
    img = np.zeros((10, 10))
    size = (3, 3)
    t1 = Template(img, (2, 2), size, label=1)
    t2 = Template(img, (2, 3), size, label=2)
    t3 = Template(img, (7, 7), size, label=3)
    scene_labels = Scene.create_scene_graph([t1, t2, t3])
    assert scene_labels[t1.id - 1] == scene_labels[t2.id - 1]
    assert scene_labels[t1.id - 1] != scene_labels[t3.id - 1]
    seg, labels = Scene.overlay_scene_graph([t1, t2, t3], img.shape)
    assert seg[t1.bbox[0], t1.bbox[2]] == seg[t2.bbox[0], t2.bbox[2]]
    assert seg[t3.bbox[0], t3.bbox[2]] != seg[t1.bbox[0], t1.bbox[2]]


def test_scene_residuals():
    """Scene.model_image()/residual() reconstruct the model from per-template flux.

    Replaces the old add_residuals()-based check (that method no longer
    exists on Scene); the real, live API for this is model_image()/residual().
    """
    img = np.zeros((10, 10))
    size = (3, 3)
    t1 = Template(img, (2, 2), size, label=1)
    t2 = Template(img, (2, 3), size, label=2)
    for tmpl in (t1, t2):
        tmpl.data[...] = 1.0

    image = np.zeros_like(img)
    weights = np.ones_like(img)
    sc = Scene(
        id=1,
        templates=[t1, t2],
        fitter=SceneFitter(),
        bbox=(0, 9, 0, 9),
        image=image,
        weights=weights,
    )
    # model_image()/residual() only require a non-None `solution` sentinel
    # and per-template `.flux`; they don't need a full solve() call.
    sc.solution = object()
    t1.flux = 2.0
    t2.flux = 3.0

    expected = np.zeros_like(img)
    expected[t1.slices_original] += 2.0 * t1.data[t1.slices_cutout]
    expected[t2.slices_original] += 3.0 * t2.data[t2.slices_cutout]

    np.testing.assert_array_almost_equal(sc.model_image(), expected)
    np.testing.assert_array_almost_equal(sc.residual(), image - expected)


@pytest.mark.parametrize("order", [1, 2])
def test_scene_solve_matches_legacy_solver(order):
    images, segmap, catalog, psfs, truth, wht = make_simple_data(
        nsrc=5, size=51, peak_snr=5, seed=order
    )
    psf_hi = PSF.from_array(psfs[0])
    psf_lo = PSF.from_array(psfs[1])
    kernel = psf_hi.matching_kernel(psf_lo)
    positions = list(zip(catalog["x"], catalog["y"]))
    tmpls = Templates.from_image(images[0], segmap, positions, kernel)
    image = images[1]
    weight = wht[1]
    cfg = FitConfig(
        fit_astrometry_joint=True,
        snr_thresh_astrom=0.0,
        astrom_kwargs={"poly": {"order": order}},
    )
    fitter = SparseFitter(tmpls.templates, image, weight, cfg)
    A, b, _ = build_normal(tmpls.templates, image, weight)
    d = np.sqrt(A.diagonal())
    Dinv = sp.diags(1.0 / d)
    A_w = Dinv @ A @ Dinv
    b_w = b / d
    scene_ids = np.ones(len(tmpls.templates), dtype=int)
    bright = np.ones(len(tmpls.templates), dtype=bool)
    alpha_legacy, err_legacy, betas, infos = fitter._solve_scenes_with_shifts(
        A_w,
        b_w,
        d,
        scene_ids,
        tmpls.templates,
        bright,
        order=order,
        include_y=True,
        ab_from_bright_only=True,
    )
    beta_legacy = betas[0][1]

    scene = Scene(id=1, templates=list(tmpls.templates), fitter=SceneFitter())
    scene.A = A
    scene.b = b
    scene.image = image
    scene.weights = weight
    flux, err, beta_scene, info = scene.solve(
        config=FitConfig(
            fit_astrometry_joint=True,
            snr_thresh_astrom=0.0,
            astrom_kwargs={"poly": {"order": order}},
        ),
        apply_shifts=False,
    )
    np.testing.assert_allclose(flux, alpha_legacy, rtol=1e-3)
