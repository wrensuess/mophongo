import numpy as np

from mophongo.catalog import Catalog
from photutils.segmentation import SegmentationImage
from photutils.aperture import CircularAperture, aperture_photometry

from utils import make_simple_data


def test_catalog_synthetic_data():
    """``Catalog.run`` on synthetic data produces finite measurement columns.

    Previously this test read an external FITS file that isn't shipped with
    the repo (``OSError: No SIMPLE card found``), so it always failed.
    Rebuilt here on ``make_simple_data`` (tests/utils.py) so it exercises the
    real API without any external data dependency.
    """
    images, segmap, catalog, psfs, truth, wht = make_simple_data(
        seed=3, nsrc=20, size=101, ndilate=2, peak_snr=5
    )
    sci, ivar = images[0], wht[0]

    cat = Catalog(sci, ivar, segmap=SegmentationImage(segmap))
    cat.run()

    assert cat.segmap.shape == cat.sci.shape
    assert cat.ivar.shape == cat.sci.shape
    assert len(cat.table) > 0
    assert np.all(np.isfinite(cat.ivar))

    table = cat.table
    for col in ("segment_flux", "kron_flux", "kron_radius", "r50", "sharpness", "snr"):
        assert col in table.colnames
        assert np.all(np.isfinite(table[col])), f"{col} has non-finite values"

    assert np.allclose(table["snr"], table["segment_flux"] / table["segment_fluxerr"])

    # Small unweighted aperture errors should reflect the weight map.
    positions = np.column_stack([table["x"], table["y"]])
    apertures = CircularAperture(positions, r=4.0)
    phot = aperture_photometry(cat.sci, apertures, error=np.sqrt(1.0 / cat.ivar))
    measured_err = phot["aperture_sum_err"].data
    expected_err = []
    for mask in apertures.to_mask(method="exact"):
        cutout = mask.multiply(1.0 / cat.ivar)
        expected_err.append(np.sqrt(np.sum(cutout[mask.data > 0])))
    expected_err = np.asarray(expected_err)
    assert np.allclose(measured_err, expected_err)
