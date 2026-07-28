import numpy as np
import mophongo.pipeline as pipeline
import mophongo.utils as mutils
from mophongo.fit import FitConfig
from utils import make_simple_data
from dataclasses import dataclass


def test_pipeline_multitemplate_pass():
    images, segmap, catalog, psfs, truth, wht = make_simple_data(nsrc=3, size=51)
    from scipy.ndimage import shift as nd_shift
    images[1] = nd_shift(images[1], (0.5, -0.3))
    kernel = [mutils.matching_kernel(psfs[0], p) for p in psfs]
    kernel[0] = np.array([[1.0]])
    config = FitConfig(fit_astrometry_niter=0)
    pl = pipeline.Pipeline(
        images,
        segmap,
        catalog=catalog,
        psfs=psfs,
        weights=wht,
        kernels=kernel,
        config=config,
    )
    table, resid = pl.run()
    # NOTE: the multi-template pass (adding extra templates for poorly-fit
    # sources when multi_tmpl_chi2_thresh is crossed) is not implemented on the
    # current solver. Its helper, _add_templates_for_bad_fits, was deleted with
    # the legacy SparseFitter -- its only call site was already commented out
    # inside the legacy branch (see docs/dead_code.md). So this no longer
    # exercises the multi-template feature at all: the assertion below is a
    # trivial sanity check that every catalog source produced a template. The
    # `fitter` object is also not exposed on Pipeline, so the original
    # `len(fitter.templates)` check was replaced with the equivalent
    # `pl.all_templates[0]` (templates used for the first/only fitted band).
    assert len(pl.all_templates[0]) >= len(catalog)
    assert np.all(np.isfinite(table['flux_1']))


@dataclass
class DummyTemplate:
    data: np.ndarray
    input_position_cutout: tuple[float, float]


def test_extract_psf_fractional_center_alignment():
    psf = np.zeros((21, 21))
    psf[10, 10] = 1.0
    tmpl = DummyTemplate(data=np.zeros((11, 11)), input_position_cutout=(5.3, 5.7))
    stamp = pipeline._extract_psf_at(tmpl, psf)
    y, x = np.indices(stamp.shape)
    x_c = float((stamp * x).sum())
    y_c = float((stamp * y).sum())
    assert np.isclose(stamp.sum(), 1.0)
    assert np.allclose([x_c, y_c], tmpl.input_position_cutout, atol=1e-3)
