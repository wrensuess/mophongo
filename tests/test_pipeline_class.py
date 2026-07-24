import numpy as np
import mophongo.pipeline as pipeline
from utils import make_simple_data
from mophongo.fit import FitConfig
import mophongo.utils as mutils


def test_pipeline_class_attributes():
    images, segmap, catalog, psfs, _, wht = make_simple_data(nsrc=3, size=51)
    kernel = [mutils.matching_kernel(psfs[0], p) for p in psfs]
    kernel[0] = np.array([[1.0]])
    pl = pipeline.Pipeline(
        images,
        segmap,
        catalog=catalog,
        psfs=psfs,
        weights=wht,
        kernels=kernel,
        config=FitConfig(fit_astrometry_niter=0),
    )
    table, residuals = pl.run()

    # run() returns (self.table, self.residuals): self.table is the output
    # flux catalog, distinct from self.catalog (the untouched input catalog
    # passed to __init__). The original assertion `pl.catalog is cat` compared
    # against the wrong attribute; fixed to match what run() actually returns.
    assert pl.table is table
    assert pl.catalog is catalog
    assert pl.residuals is residuals
    assert pl.astro is not None
