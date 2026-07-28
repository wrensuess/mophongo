import numpy as np
import pytest
import scipy.sparse as sp

from mophongo.scene_fitter import SceneFitter, build_normal
from mophongo.scene import Scene, make_scene_basis, assemble_scene_system_AB
from mophongo.templates import Templates, Template
from mophongo.fit import FitConfig
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
    # can actually produce (finite, positive). NOTE: nothing currently pins the
    # error orientation/covariance question -- the test that did lived in
    # test_fit.py, which was deleted with the legacy SparseFitter. See the
    # "coverage to restore" note in docs/dead_code.md.
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
    cfg = FitConfig(positivity=False)
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
def test_scene_solve_matches_dense_on_real_templates(order):
    """Scene.solve()'s joint flux+shift result matches a dense solve of the
    same augmented system, on templates extracted from real simulated data.

    This is the end-to-end counterpart to
    test_solve_flux_and_shifts_matches_dense(), which pins the same linear
    algebra on a small synthetic system. Here A and the coupling blocks come
    from actual templates, so this additionally exercises Scene.solve()'s
    scaling of the blocks (alpha0), its bright-mask handling, and the
    whitening/unwhitening round trip on a realistic, badly-scaled system.

    Scope limit, deliberately: the reference is built with the *same*
    make_scene_basis()/assemble_scene_system_AB() that Scene.solve() calls, so
    a bug inside those two functions would cancel on both sides and is NOT
    caught here. What is caught is Scene.solve() feeding them the wrong inputs
    or mishandling their output. Pinning the assembly itself against an
    independent implementation is separate, missing coverage.

    The `beta` assertion is the load-bearing one. The fluxes are nearly
    insensitive to the shift block for isolated symmetric templates (the
    gradient integrates to ~0, so AB ~ 0); `beta` scales as 1/alpha0 and does
    respond, so it is what actually pins the coupling.
    """
    # nsrc=20 (not 5) so the shift block is full rank at BOTH orders: with 5
    # sources, order=2 gives 12 shift coefficients against 5 sources ->
    # rank(BB)=10/12, cond ~1e17, and the dense `beta` is meaningless. At
    # nsrc=20: order=1 cond(BB)=7, order=2 cond(BB)=43, both full rank.
    images, segmap, catalog, psfs, truth, wht = make_simple_data(
        nsrc=20, size=101, peak_snr=5, seed=order
    )
    psf_hi = PSF.from_array(psfs[0])
    psf_lo = PSF.from_array(psfs[1])
    kernel = psf_hi.matching_kernel(psf_lo)
    positions = list(zip(catalog["x"], catalog["y"]))
    tmpls = Templates.from_image(images[0], segmap, positions, kernel)
    image = images[1]
    weight = wht[1]
    # positivity=False so the unconstrained dense reference below is the right
    # ground truth (the default clamp would legitimately zero negative fluxes).
    cfg = FitConfig(
        fit_astrometry_niter=1,
        snr_thresh_astrom=0.0,
        astrom_isolation_thresh=0.0,
        positivity=False,
        astrom_kwargs={"poly": {"order": order}},
    )

    A, b, _ = build_normal(tmpls.templates, image, weight)

    # Reference: assemble the same augmented system Scene.solve() builds and
    # solve it densely.
    d = np.asarray(A.diagonal(), dtype=float)
    alpha0 = np.divide(b, d, out=np.zeros_like(b, dtype=float), where=d > 0)
    bright = np.ones(len(tmpls.templates), dtype=bool)
    basis, _, _ = make_scene_basis(tmpls.templates, bright, order=order)
    AB, BB, bB = assemble_scene_system_AB(
        tmpls.templates,
        image,
        weight,
        basis,
        alpha0=alpha0,
        order=order,
        include_y=True,
        ab_from_bright_only=True,
    )
    M = np.block([[A.toarray(), AB.toarray()], [AB.T.toarray(), BB.toarray()]])
    rhs = np.concatenate([b, bB])
    dense = np.linalg.solve(M, rhs)
    n = A.shape[0]

    scene = Scene(id=1, templates=list(tmpls.templates), fitter=SceneFitter())
    scene.A = A
    scene.b = b
    scene.image = image
    scene.weights = weight
    flux, err, beta_scene, info = scene.solve(config=cfg, apply_shifts=False)

    # Guard the premise: if BB ever loses rank the dense `beta` below stops
    # being a valid reference and the assertion would silently go vacuous.
    assert np.linalg.matrix_rank(BB.toarray()) == BB.shape[0]

    # SceneFitter.solve adds a small ridge to the flux block and to BB
    # (reg_astrom * median(diag(BB))) before solving, so the agreement with the
    # unregularized dense solve is ~1e-3, not machine precision.
    np.testing.assert_allclose(flux, dense[:n], rtol=1e-3)
    np.testing.assert_allclose(beta_scene, dense[n:], rtol=1e-2, atol=1e-3)
