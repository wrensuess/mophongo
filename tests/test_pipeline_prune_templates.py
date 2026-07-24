import numpy as np
from astropy.table import Table
import mophongo.pipeline as pipeline


def test_pipeline_prunes_templates_with_zero_weight():
    hires = np.zeros((4, 4))
    lowres = np.zeros((4, 4))
    segmap = np.zeros((4, 4), dtype=int)
    segmap[0:2, 0:2] = 1
    segmap[2:4, 2:4] = 2
    hires[segmap > 0] = 1.0

    images = [hires, lowres]
    catalog = Table({"id": [1, 2], "x": [0, 3], "y": [0, 3]})
    w0 = np.ones_like(hires)
    w1 = np.ones_like(lowres)
    w1[2:4, 2:4] = 0
    weights = [w0, w1]
    kernels = [None, None]

    pl = pipeline.Pipeline(images, segmap, catalog=catalog, weights=weights, kernels=kernels)
    table, residuals = pl.run()

    # `fitter` is not exposed on Pipeline; pl.all_templates[0] holds the
    # templates actually used to fit the (only) band, which is the equivalent
    # post-pruning count.
    assert len(pl.all_templates[0]) == 1
    assert np.isfinite(table["flux_1"][0])
    assert np.isnan(table["flux_1"][1])
