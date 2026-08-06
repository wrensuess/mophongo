"""Simple pipeline orchestrator.

This module exposes the :func:`run_photometry` function which ties together the
high level steps of the photometry pipeline. The actual implementation of the
template extraction and sparse fitting are delegated to the ``templates`` and
``fit`` modules which will be implemented separately.
"""

from __future__ import annotations

import os
import psutil
from typing import Sequence
from copy import deepcopy
import logging
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D, block_replicate, block_reduce
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.segmentation import SegmentationImage, SourceCatalog
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.ndimage import find_objects, maximum_filter

from .psf_map import PSFRegionMap
from . import utils
from .utils import bin_factor_from_wcs, downsample_psf, bin_remap
from .templates import Templates, Template, _slices_from_bbox
from .fit import FitConfig as _FitConfig
from .scene import generate_scenes

import logging

logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)  # show info for *this* logger only
if not logger.handlers:  # avoid duplicate handlers on reloads
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(module)s.%(funcName)s: %(message)s"))
    logger.addHandler(handler)

memory = lambda: psutil.Process(os.getpid()).memory_info().rss / 1e9



# should support PSFRegionMap as well, like in template.convolve_templates
#   ra, dec = tmpl.wcs.wcs_pix2world(x, y, 0)
# else:
#     ra, dec = x, y
# kern = kernel.get_psf(ra, dec)


def _sources_with_coverage(
    segmap: np.ndarray,
    cat: Table,
    wcs: Sequence[WCS] | None,
    weights: Sequence[np.ndarray] | None,
    min_size: int,
) -> np.ndarray:
    """Boolean mask of catalog rows worth building a template for.

    MIRI mosaics often cover a small fraction of the detection (F444W) area, so
    building every detection template and pruning afterwards wastes most of the
    extraction. A source is kept when *any* fitted band has positive weight
    inside the footprint its template will occupy -- the segment bbox, floored
    at ``min_size // 2``, which the pipeline pre-sizes to hold the PSF extension
    radius ``r_fill`` (see the min_size floor in Pipeline.run).

    The radius is per-source on purpose: one global radius would be set by the
    largest segment in the field, and a single bright-star halo (~4000 px in
    UDS) dilates the coverage mask until nothing is cut at all.

    This mirrors ``Templates.prune_outside_weight``, which still does the exact
    per-band cut -- on unconvolved templates, hence no kernel margin here. The
    rough cut only skips templates that cannot survive it in any band, and is
    itself skipped whenever coverage cannot be established (no weights, no WCS,
    or a weightless band, which counts as full coverage).
    """
    keep = np.ones(len(cat), dtype=bool)
    if weights is None or wcs is None or len(weights) < 2:
        return keep
    bands = range(1, len(weights))
    if any(weights[i] is None or wcs[i] is None for i in bands):
        return keep

    # Non-finite or off-image positions are parked at pixel 0 and dropped via
    # has_seg -- the same sources extract_templates skips.
    x = np.asarray(cat["x"], dtype=float)
    y = np.asarray(cat["y"], dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    xi = np.round(np.where(finite, x, 0.0)).astype(int)
    yi = np.round(np.where(finite, y, 0.0)).astype(int)
    ny, nx = segmap.shape
    inside = finite & (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
    labels = np.where(inside, segmap[np.clip(yi, 0, ny - 1), np.clip(xi, 0, nx - 1)], 0)

    # Per-source template half-size on the detection grid, exactly as
    # Templates.extract_templates sizes its cutouts. photutils' bbox.iymax is
    # exclusive, i.e. equal to the find_objects slice stop.
    slices = find_objects(segmap)
    bbox = np.zeros((len(slices) + 1, 4), dtype=float)
    for lab, sl in enumerate(slices, start=1):
        if sl is not None:
            bbox[lab] = (sl[0].start, sl[0].stop, sl[1].start, sl[1].stop)

    has_seg = labels > 0
    half = np.full(len(cat), float(min_size // 2))
    b = bbox[labels]
    half[has_seg] = np.maximum.reduce(
        [yi - b[:, 0], b[:, 1] - yi, xi - b[:, 2], b[:, 3] - xi, half]
    )[has_seg]

    sky = wcs[0].pixel_to_world(np.where(finite, x, 0.0), np.where(finite, y, 0.0))
    keep = np.zeros(len(cat), dtype=bool)
    for i in bands:
        w = weights[i]
        wh, ww = w.shape
        k = bin_factor_from_wcs(wcs[0], wcs[i])
        # +1 absorbs the rounding of the centre to an integer pixel and the WCS
        # round-trip, so the tested box always contains the template footprint.
        radius = np.ceil(half / k).astype(int) + 1
        xb, yb = wcs[i].world_to_pixel(sky)
        xb, yb = np.round(xb).astype(int), np.round(yb).astype(int)

        # Nearly every source sits at the floor radius, so dilating the coverage
        # by it once turns their box test into a single lookup. Segments larger
        # than the floor are ~1% of a real segmap, but a single star halo can be
        # thousands of pixels across -- using one global radius would dilate the
        # whole field by it -- so those get their own box tested instead.
        r0 = int(np.ceil((min_size // 2) / k)) + 1
        cov = maximum_filter(w > 0, size=2 * r0 + 1)
        centred = (xb >= 0) & (xb < ww) & (yb >= 0) & (yb < wh)
        fast = has_seg & ~keep & centred & (radius == r0)
        keep[fast] = cov[yb[fast], xb[fast]]

        near = (
            (xb >= -radius) & (xb < ww + radius) & (yb >= -radius) & (yb < wh + radius)
        )
        for j in np.nonzero(has_seg & ~keep & ~fast & near)[0]:
            r = int(radius[j])
            y0, y1 = max(int(yb[j]) - r, 0), min(int(yb[j]) + r + 1, wh)
            x0, x1 = max(int(xb[j]) - r, 0), min(int(xb[j]) + r + 1, ww)
            keep[j] = y1 > y0 and x1 > x0 and bool(np.any(w[y0:y1, x0:x1] > 0))

    return keep


def _extract_psf_at(tmpl: Template, psf: np.ndarray | PSFRegionMap) -> np.ndarray:
    """Return a PSF stamp matching the template size.

    Parameters
    ----------
    tmpl : Template
        Template object providing position and size information
    psf : np.ndarray or PSFRegionMap
        Either a static PSF array or a PSFRegionMap for spatially varying PSFs

    Returns
    -------
    np.ndarray
        PSF stamp normalized to sum=1, matching template size
    """
    from scipy.ndimage import shift

    # Get the PSF array - either directly or via lookup
    if isinstance(psf, PSFRegionMap):
        # Look up PSF at template position
        x, y = tmpl.input_position_original
        if hasattr(tmpl, "wcs") and tmpl.wcs is not None:
            ra, dec = tmpl.wcs.wcs_pix2world(x, y, 0)
        else:
            ra, dec = x, y
        psf_array = psf.get_psf(ra, dec)
        if psf_array is None:
            raise ValueError(f"No PSF found at position ({ra}, {dec})")
    else:
        # Use static PSF array
        psf_array = psf

    ny, nx = tmpl.data.shape
    cx_psf, cy_psf = psf_array.shape[1] // 2, psf_array.shape[0] // 2

    xc, yc = tmpl.input_position_cutout
    dx = xc - (nx // 2)
    dy = yc - (ny // 2)

    shifted = shift(psf_array, shift=(dy, dx), order=3, mode="constant", cval=0.0, prefilter=False)
    cut = Cutout2D(
        shifted,
        (cx_psf, cy_psf),
        tmpl.data.shape,
        mode="partial",
        fill_value=0.0,
    )
    stamp = cut.data.copy()
    s = stamp.sum()
    if s > 0:
        stamp /= s
    return stamp


class Pipeline:
    """Photometry pipeline orchestrator.

    Parameters mirror :func:`run` for backwards compatibility. After
    calling :meth:`run` the resulting catalog, residual images and fitter
    instance are stored on the object and returned.
    """

    def __init__(
        self,
        images: Sequence[np.ndarray],
        segmap: np.ndarray,
        *,
        catalog: Table | None = None,
        psfs: Sequence[np.ndarray] | None = None,
        weights: Sequence[np.ndarray] | None = None,
        kernels: Sequence[np.ndarray | PSFRegionMap] | None = None,
        wcs: Sequence[WCS] | None = None,
        window: Window | None = None,
        extend_templates: str | None = None,
        config: FitConfig | None = None,
    ) -> None:
        if psfs is not None and len(images) != len(psfs):
            raise ValueError("Number of images and PSFs must match")
        if weights is not None and len(weights) != len(images):
            raise ValueError("Number of weight images must match number of images")

        if config is None:
            config = _FitConfig()

        self.images = images
        self.segmap = segmap
        self.catalog = catalog
        self.psfs = psfs
        self.weights = weights
        self.kernels = kernels
        self.wcs = wcs
        self.window = window
        self.extend_templates = extend_templates
        self.config = config

        if kernels is None:
            kernels = [None] * len(images)
        if psfs is None:
            psfs = [None] * len(images)

        self.residuals: list[np.ndarray] = []
        self.fit: list[np.ndarray] = []
        self.astro: list[np.ndarray] = []
        #        self.templates: list[np.ndarray] = []
        self.infos: list[dict] = []
        self.tmpls: Templates()

        print(f"Pipeline (init) memory: {memory():.1f} GB")

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _update_catalog_with_fluxes(
        self,
        cat: Table,
        templates: list[Template],
        fluxes: np.ndarray,
        errs: np.ndarray,
        err_pred: np.ndarray,
        idx: int,
    ) -> None:
        """Insert measured fluxes into the output catalog.

        Parameters
        ----------
        cat
            Catalog to update.
        templates
            Templates associated with the fitted sources.
        fluxes, errs, err_pred
            Flux measurements and their uncertainties.
        idx
            Index of the current image (used for column naming).
        """

        parent_ids = [
            tmpl.id_parent if getattr(tmpl, "parent_id", None) is not None else tmpl.id
            for tmpl in templates
        ]
        id_to_index = {id_: i for i, id_ in enumerate(cat["id"])}
        cat[f"flux_{idx}"] = self.config.bad_value
        cat[f"err_{idx}"] = self.config.bad_value
        cat[f"err_pred_{idx}"] = self.config.bad_value

        flux_sum: defaultdict[int, float] = defaultdict(float)
        err_sum: defaultdict[int, float] = defaultdict(float)
        err_pred_sum: defaultdict[int, float] = defaultdict(float)
        for pid, fl, er, ep in zip(parent_ids, fluxes, errs, err_pred):
            if pid is None:
                continue
            flux_sum[pid] += fl
            err_sum[pid] = float(np.sqrt(err_sum[pid] ** 2 + er**2))
            err_pred_sum[pid] = float(np.sqrt(err_pred_sum[pid] ** 2 + ep**2))

        for pid, fl in flux_sum.items():
            ci = id_to_index.get(pid)
            if ci is None:
                continue
            cat[f"flux_{idx}"][ci] = fl
            cat[f"err_{idx}"][ci] = err_sum[pid]
            cat[f"err_pred_{idx}"][ci] = err_pred_sum[pid]

    def _pixel_scale_arcsec(self, w: WCS | None) -> float | None:
        try:
            if w is None:
                return None
            # (dy, dx) scale; pick x
            return float(proj_plane_pixel_scales(w)[0] * 3600.0)
        except Exception:
            return None

    def _gaussian_fwhm_pix(self, psf: np.ndarray | None) -> float | None:
        if psf is None:
            return None
        try:
            from .utils import measure_shape

            mask = psf > (0.0 if np.min(psf) >= 0 else np.median(psf))
            _, _, sx, sy, _ = measure_shape(psf.astype(np.float32), mask.astype(bool))
            return 2.354820045 * float(np.sqrt(sx * sy))
        except Exception:
            return None

    def _resolve_image_ap_radius_pix(self, idx: int, cfg: _FitConfig) -> float:
        """
        Diameter source: cfg.aperture_diam
        - float/int => same for all images
        - np.ndarray(len(images)-1) => per image (idx>=1), pick [idx-1]
        - None => 1.5 × FWHM of PSF[idx] (in *pixels* of image idx),
                    fallback 3.0 pixels if PSF is missing.
        Units: cfg.aperture_units ("arcsec" or "pix")
        """
        diam = None
        if isinstance(cfg.aperture_diam, (int, float)):
            diam = float(cfg.aperture_diam)
        elif isinstance(cfg.aperture_diam, np.ndarray):
            # array corresponds to images[1:], so use [idx-1]
            if cfg.aperture_diam.size != (len(self.images) - 1):
                raise ValueError("aperture_diam array must have len(images)-1 elements")
            diam = float(cfg.aperture_diam[idx - 1])  # idx>=1 by construction here

        if diam is None:
            # default: 1.5×FWHM of this image PSF (pixels)
            psf_i = None
            if self.psfs is not None and len(self.psfs) > idx:
                psf_i = self.psfs[idx]
                if isinstance(psf_i, np.ndarray):
                    fwhm_pix = self._gaussian_fwhm_pix(psf_i)
                else:
                    # PSFRegionMap: use the first PSF as a representative
                    try:
                        fwhm_pix = self._gaussian_fwhm_pix(psf_i.psfs[0])
                    except Exception:
                        fwhm_pix = None
            else:
                fwhm_pix = None
            rad_pix = 1.5 * fwhm_pix if fwhm_pix and fwhm_pix > 0 else 3.0
            logger.info(f"Using aperture diam 1.5x fwhm {2*rad_pix:.2f} pix for image {idx}")
            return float(rad_pix)

        # convert diameter to pixels if needed
        if cfg.aperture_units.lower().startswith("arc"):
            pscale = self._pixel_scale_arcsec(self.wcs[idx] if self.wcs is not None else None)
            if not pscale or pscale <= 0:
                raise ValueError("aperture_diam in arcsec requires valid WCS for each image")
            return float(diam / (2.0 * pscale))
        else:
            return float(diam / 2.0)  # already in pixels

    def _resolve_catalog_ap_radius_pix(
        self, cat: Table, cfg: _FitConfig, r_default: float | None = None
    ) -> dict[int, float]:
        """
        Return per-source catalog aperture *radius in pixels of the reference image (idx=0)*.

        Source:
        - str => table column name with per-source *diameters*
        - float/int => fixed *diameter* for all sources
        - None => default 1.5 × FWHM of PSF[0] in pixels (fallback 3.0)

        Units: cfg.aperture_units ("arcsec" or "pix")
        """
        # get reference pixel scale
        pscale_ref = self._pixel_scale_arcsec(self.wcs[0] if self.wcs is not None else None)

        out: dict[int, float] = {}

        # if no catalog, default to r_default for all (if given)
        if cfg.aperture_catalog is None:
            for i, _ in enumerate(cat["id"]):
                out[int(cat["id"][i])] = r_default
            return out

        # get from catalog
        if isinstance(cfg.aperture_catalog, (int, float)):
            diam = float(cfg.aperture_catalog)
            if cfg.aperture_units.lower().startswith("arc"):
                if not pscale_ref or pscale_ref <= 0:
                    raise ValueError("aperture_catalog in arcsec requires valid ref WCS")
                rad = diam / (2.0 * pscale_ref)
            else:
                rad = diam / 2.0
            for i, _ in enumerate(cat["id"]):
                out[int(cat["id"][i])] = float(rad)
            return out

        # string column name
        col = str(cfg.aperture_catalog)
        if col not in cat.colnames:
            raise ValueError(f"aperture_catalog column '{col}' not found in table")
        if cfg.aperture_units.lower().startswith("arc"):
            if not pscale_ref or pscale_ref <= 0:
                raise ValueError("aperture_catalog in arcsec requires valid ref WCS")
            for i, _ in enumerate(cat["id"]):
                diam = float(cat[col][i])
                out[int(cat["id"][i])] = float(diam / (2.0 * pscale_ref))
        else:
            for i, _ in enumerate(cat["id"]):
                diam = float(cat[col][i])
                out[int(cat["id"][i])] = float(diam / 2.0)

        return out

    @staticmethod
    def _get_representative_kernel(kernel) -> np.ndarray | None:
        """Return a normalized representative kernel from a PSFRegionMap or ndarray, or None."""
        if isinstance(kernel, np.ndarray) and kernel.ndim == 2:
            arr = kernel
        elif hasattr(kernel, "psfs") and len(kernel.psfs) > 0:
            # prefer the middle element; fall back to any non-zero kernel
            mid = len(kernel.psfs) // 2
            n = len(kernel.psfs)
            candidates = sorted(range(n), key=lambda i: abs(i - mid))
            arr = next((kernel.psfs[i] for i in candidates if float(kernel.psfs[i].sum()) > 0), None)
            if arr is None:
                return None
        else:
            return None
        total = float(arr.sum())
        return (arr / total) if total > 0 else None

    def _aperture_sum_on_template(self, tmpl: Template, radius_pix: float) -> float:
        """Exact circular aperture sum on a template, centred on the source.

        Uses tmpl.data[tmpl.slices_cutout] with the source position shifted
        into that slice's frame. This matches the geometry used for the
        matching residual patch (residual[tmpl.slices_original]), so
        aperture-sum linearity gives ap_flux = ap_model + res_sum exactly.
        """
        xc = tmpl.input_position_cutout[0] - tmpl.slices_cutout[1].start
        yc = tmpl.input_position_cutout[1] - tmpl.slices_cutout[0].start
        aper = CircularAperture((float(xc), float(yc)), r=float(radius_pix))
        phot = aperture_photometry(tmpl.data[tmpl.slices_cutout], aper, method="exact")
        return float(phot["aperture_sum"][0])

    def _aperture_sum_on_map(self, fullmap: np.ndarray, tmpl: Template, radius_pix: float) -> float:
        """Exact circular aperture sum of a FULL-image map at a source's position.

        Same geometry as :meth:`_aperture_sum_on_template`, but the values come
        from ``fullmap[tmpl.slices_original]`` (e.g. the F444W residual) instead
        of the template's own data. Aperture-sum linearity then gives, for the
        neighbour-subtracted F444W flux of this source,
        ``aper(residual + model_i) = aper(residual) + template_norm * apF_book``.
        """
        xc = tmpl.input_position_cutout[0] - tmpl.slices_cutout[1].start
        yc = tmpl.input_position_cutout[1] - tmpl.slices_cutout[0].start
        aper = CircularAperture((float(xc), float(yc)), r=float(radius_pix))
        phot = aperture_photometry(fullmap[tmpl.slices_original], aper, method="exact")
        return float(phot["aperture_sum"][0])

    def _model_kron(
        self,
        orig_t: Template,
        label: int,
        r_floor_pix: float,
        use_source_catalog: bool = True,
    ) -> tuple[float, float]:
        """Model-side Kron aperture-to-total measurement (Stage 3b, design doc
        Sec 5.4 -- the Skelton-style floor).

        Runs photutils ``SourceCatalog`` on the fitted F444W template's own
        model stamp (``orig_t.data[orig_t.slices_cutout] * orig_t.
        template_norm``, real flux units -- NOT image data), using the
        source's own segment (``self.segmap[orig_t.slices_original] ==
        label``), with the Kron circular-radius floored at ``r_floor_pix``
        (the catalog color-aperture radius) via photutils' own
        ``kron_params`` minimum-circular-radius mechanism -- the same
        machinery and conventions as :mod:`mophongo.catalog`.

        Returns ``(kron_flux_model, r_kron_circ_pix)``: the model Kron flux
        (real units) and the circularized Kron radius
        (``max(kron_params[0] * kron_radius * sqrt(a*b), r_floor_pix)``,
        capped at the stamp half-width). The two always share one radius:
        when the cap engages, photutils' Kron flux (measured on a larger,
        edge-truncated aperture) is REPLACED by the circular flux at the
        capped radius, so ``kron_flux / EE(r_kron_circ)`` never mixes radii.
        The returned radius is quantized to a 0.25-px grid so the caller's
        per-region PSF-EE cache can hit (EE varies slowly with radius: the
        <0.125 px rounding is a <0.2% effect).

        ``use_source_catalog=False`` -- the ``apcor_from_psf`` performance
        shortcut (a PSF-converged faint/compact template's Kron radius floors
        anyway, so photutils is skipped entirely) -- and any failure
        (degenerate moments, an empty segment in the stamp, a photutils
        exception) both take the same fallback: a circular aperture at
        ``r_floor_pix`` on the unit template, scaled by ``template_norm``.
        """
        stamp = orig_t.data[orig_t.slices_cutout] * orig_t.template_norm

        def _fallback() -> tuple[float, float]:
            # Same 0.25-px quantization as the main path (EE-cache hit rate),
            # applied BEFORE the flux measurement so flux and radius share.
            r_q = float(np.round(r_floor_pix * 4.0) / 4.0)
            frac = self._aperture_sum_on_template(orig_t, r_q)
            return orig_t.template_norm * frac, r_q

        if not use_source_catalog:
            return _fallback()

        try:
            seg = self.segmap[orig_t.slices_original] == label
            if seg.shape != stamp.shape or not seg.any():
                return _fallback()
            scat = SourceCatalog(
                stamp, SegmentationImage(seg.astype(int)),
                kron_params=(2.5, 1.4, r_floor_pix),
            )
            kron_flux = float(scat.kron_flux[0])
            kron_radius = float(scat.kron_radius[0].value)
            a = float(scat.semimajor_sigma[0].value)
            b = float(scat.semiminor_sigma[0].value)
            if not (np.isfinite(kron_flux) and kron_flux > 0
                    and np.isfinite(kron_radius) and np.isfinite(a) and np.isfinite(b)):
                return _fallback()
            r_circ = max(2.5 * kron_radius * np.sqrt(a * b), r_floor_pix)
            r_cap = 0.5 * min(stamp.shape)
            capped = r_circ > r_cap
            if capped:
                r_circ = r_cap
            # Quantize to a 0.25-px grid (EE-cache hit rate; <0.2% EE effect),
            # BEFORE the cap re-measurement below so flux and EE stay shared.
            # r_cap is a multiple of 0.25 (0.5 * integer), so rounding cannot
            # push the radius back above the cap.
            r_circ = float(np.round(r_circ * 4.0) / 4.0)
            if capped:
                # Shared-radius invariant: photutils' Kron flux was measured
                # on the larger, edge-truncated elliptical aperture; re-measure
                # the numerator as the circular flux at the SAME capped radius
                # the EE denominator will use.
                kron_flux = orig_t.template_norm * self._aperture_sum_on_template(orig_t, r_circ)
            return kron_flux, r_circ
        except Exception:  # pragma: no cover - degenerate stamp/segment
            return _fallback()

    def _build_f444w_residual(self, orig_templates: list[Template]) -> np.ndarray:
        """F444W neighbour-subtracted residual map: images[0] - Σ_j model_j.

        Each high-res template (unit-sum) is scaled by ``template_norm`` (its
        detection-band flux) and subtracted from the F444W image. Built once and
        reused across bands; for each source we add its own model back before the
        aperture sum, so the ``apf_data`` diagnostic is the real neighbour-
        subtracted F444W aperture flux rather than the (noise-level) template
        sum (diagnostic only -- it feeds no correction). The map is written to
        disk for over-subtraction diagnosis.
        """
        # Native dtype copy (typically float32) — avoids doubling memory on the
        # full mosaic. Each pixel is covered by only a few templates, so float32
        # accumulation is fine for this diagnostic/aperture-sum use.
        res = np.array(self.images[0], copy=True)
        for t in orig_templates:
            tn = float(getattr(t, "template_norm", 0.0) or 0.0)
            if tn and np.isfinite(tn):
                res[t.slices_original] -= t.data[t.slices_cutout] * tn
        try:
            hdr = (
                self.wcs[0].to_header()
                if self.wcs is not None and self.wcs[0] is not None
                else None
            )
            fits.writeto("f444w_template_residual.fits", res.astype(np.float32), hdr, overwrite=True)
            logger.info("Wrote F444W template residual map -> f444w_template_residual.fits")
        except Exception as exc:  # pragma: no cover - diagnostics only
            logger.warning("Could not save F444W template residual map: %s", exc)
        return res

    def _residual_segmap_sum(
        self,
        residual: np.ndarray,
        source_id: int,
        orig_t: Template,
        k: int,
    ) -> float:
        """Sum residual within (segmap == source_id), at the residual's resolution.

        If the residual is at lower resolution than the segmap (k > 1), the
        high-res segmap mask is binned down: a low-res pixel is included when
        any high-res pixel in the corresponding k×k block belongs to the source.
        """
        sl_hi = orig_t.slices_original
        mask_hi = self.segmap[sl_hi] == source_id
        if not mask_hi.any():
            return 0.0

        if k == 1:
            return float(residual[sl_hi][mask_hi].sum())

        # Multi-resolution: bin mask_hi down to low-res. Pad so the high-res
        # region starts and ends on k-block boundaries, then block_reduce.
        y0_hi, x0_hi = sl_hi[0].start, sl_hi[1].start
        y_pre = y0_hi % k
        x_pre = x0_hi % k
        m = np.pad(mask_hi.astype(np.uint8), ((y_pre, 0), (x_pre, 0)), constant_values=0)
        y_post = (-m.shape[0]) % k
        x_post = (-m.shape[1]) % k
        if y_post or x_post:
            m = np.pad(m, ((0, y_post), (0, x_post)), constant_values=0)
        mask_lo = block_reduce(m, k, func=np.sum) > 0

        y0_lo = (y0_hi - y_pre) // k
        x0_lo = (x0_hi - x_pre) // k
        sl_lo = (
            slice(y0_lo, y0_lo + mask_lo.shape[0]),
            slice(x0_lo, x0_lo + mask_lo.shape[1]),
        )
        # Clip to residual bounds
        h_r, w_r = residual.shape
        if sl_lo[0].stop > h_r:
            excess = sl_lo[0].stop - h_r
            sl_lo = (slice(sl_lo[0].start, h_r), sl_lo[1])
            mask_lo = mask_lo[: mask_lo.shape[0] - excess, :]
        if sl_lo[1].stop > w_r:
            excess = sl_lo[1].stop - w_r
            sl_lo = (sl_lo[0], slice(sl_lo[1].start, w_r))
            mask_lo = mask_lo[:, : mask_lo.shape[1] - excess]

        return float((residual[sl_lo] * mask_lo).sum())

    def _other_source_mask(self, tmpl: Template, source_id: int, k: int) -> np.ndarray | None:
        """Boolean mask (shape of residual[tmpl.slices_original]) of pixels that
        belong to OTHER sources' segments.

        Used to exclude neighbour-segment pixels from a source's residual aperture
        (a partial mitigation of the blend residual double-count -- see the TODO
        in ``_add_aperture_photometry``). Returns None if shapes don't line up.
        """
        sl = tmpl.slices_original
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        seg = self.segmap
        if k == 1:
            sub = seg[sl]
            if sub.shape != (h, w):
                return None
            return (sub != 0) & (sub != source_id)
        # Multi-resolution: map the low-res patch to the high-res segmap block,
        # mark "other source" at high-res, then OR-reduce to low-res.
        y0, x0 = sl[0].start * k, sl[1].start * k
        hi = np.zeros((h * k, w * k), dtype=seg.dtype)
        ys1 = min(y0 + h * k, seg.shape[0])
        xs1 = min(x0 + w * k, seg.shape[1])
        hi[: ys1 - y0, : xs1 - x0] = seg[y0:ys1, x0:xs1]
        other_hi = (hi != 0) & (hi != source_id)
        other_lo = block_reduce(other_hi.astype(np.uint8), k, func=np.sum) > 0
        return other_lo if other_lo.shape == (h, w) else None

    @staticmethod
    def _intersect_slices(tmpl: Template, y0: int, y1: int, x0: int, x1: int):
        """Compute the intersection of a template's valid region with a bounding box.

        Returns (patch_ys, patch_xs, data_ys, data_xs) in patch-local and
        template-data coordinates, or None if there is no overlap.
        """
        iy0 = tmpl.slices_original[0].start
        iy1 = tmpl.slices_original[0].stop
        ix0 = tmpl.slices_original[1].start
        ix1 = tmpl.slices_original[1].stop

        cy0 = max(iy0, y0)
        cy1 = min(iy1, y1)
        cx0 = max(ix0, x0)
        cx1 = min(ix1, x1)

        if cy1 <= cy0 or cx1 <= cx0:
            return None

        patch_ys = slice(cy0 - y0, cy1 - y0)
        patch_xs = slice(cx0 - x0, cx1 - x0)

        dc_y0 = tmpl.slices_cutout[0].start
        dc_x0 = tmpl.slices_cutout[1].start
        data_ys = slice(dc_y0 + (cy0 - iy0), dc_y0 + (cy1 - iy0))
        data_xs = slice(dc_x0 + (cx0 - ix0), dc_x0 + (cx1 - ix0))

        return patch_ys, patch_xs, data_ys, data_xs

    def _add_aperture_photometry(
        self,
        cat: Table,
        templates: list[Template],
        fluxes: np.ndarray,
        residual: np.ndarray,
        idx: int,
        r_orig_pix: float | None = None,
        orig_templates: list[Template] | None = None,
        f444w_totals: dict[int, float] | None = None,
    ) -> None:
        """Compute per-source aperture corrections (Estimator 3).

        Invariant: real-flux bookkeeping (``ap_model``, ``ap_flux``, ``ap_f_data``)
        always uses the fitted template's own aperture fraction, since ``fl`` and
        ``template_norm`` are defined against that unit-sum template; the
        per-source truncation term (below) enters ONLY the correction factors
        ``apcor1``/``totcor1``.

        Templates are unit-sum normalised; ``orig_t.template_norm`` holds the
        pre-normalisation detection-band sum, converting aperture fractions to
        real flux units. All quantities below in real (image) flux units:

        trunc    = template_norm / (template_norm + flux_beyond_stamp)  — per-source
                   truncation (docs/aperture_corrections.md Sec 5.1/6); flux_beyond_stamp
                   is the unified template's PSF-extrapolated, core-anchored estimate
                   of this source's flux beyond the cutout (set in
                   ``Templates._extended_composite``).
        apB_corr = apB_book * trunc, apF_corr = apF_book * trunc   — trunc cancels in
                   apcor1 (a shape ratio) and survives in totcor1 (aperture-to-total).
        ap_b_corr = template_norm * apB_corr         — low-res aperture flux, correction side
        ap_f_corr = template_norm * apF_corr         — high-res aperture flux, correction side
        apcor1   = ap_f_corr / ap_b_corr             — shape correction (low-res→high-res)
        ap_f_data = aper(F444W_residual + model_i, r_phi)   — REAL neighbour-subtracted
                                                    F444W aperture flux (= template_norm*apF_book
                                                    + residual-in-aper), kept as the
                                                    ``apf_data_{idx}`` diagnostic column only
                                                    (docs/aperture_corrections.md Sec 5.4).

        ap_model     = fl * aper(H*K, r_phi)         (model flux in aperture, low-res)
        res_sum      = Σ_aperture(residual)          (residual within the aperture disk)
        res_seg      = Σ_segmap(residual)            (diagnostic only)
        ap_flux      = ap_model + res_sum            (observed aperture flux)
        ap_flux_est1    = (ap_model + res_sum) * totcor1      (internal-IDL Estimator 1)
        ap_flux_est2    = ap_model * totcor1 + res_sum        (internal-IDL Estimator 2, residual unscaled)
        ap_flux_est3int = ap_model * apcor1 * tcor_int + res_sum          (internal-Kron total)
        ap_flux_est3cat = ap_model * apcor1 * tcor_int * s_cat + res_sum  (catalog-tied release)

        Per-estimator errors -- the fractional profile-fit error (err/flux from
        the sparse solve) propagated onto each estimator's corrected MODEL flux
        (correction * ap_model = est minus its residual term). Multiplicative
        corrections are treated as noiseless and the res_sum aperture pixel-noise
        term is NOT included, so these track the fit SNR, err/flux:
        err_ap_flux_est1    = |totcor1 * ap_model|                 * (err/flux)
        err_ap_flux_est2    = |totcor1 * ap_model|                 * (err/flux)  (= est1's)
        err_ap_flux_est3int = |apcor1 * tcor_int * ap_model|        * (err/flux)
        err_ap_flux_est3cat = |apcor1 * tcor_int * s_cat * ap_model| * (err/flux)

        Stage-3b two-step catalog tie (docs/aperture_corrections.md Sec 5.4), all
        evaluated on MODELS (never on measured aperture flux). Stage-4c (docs
        stage4c_scope_and_brief.md): the tie denominator and F444W_total_moph
        are UNMASKED, consistent with the Stage-4b apcor1/totcor1 fix --
        ``template_norm`` alone (Sigma(H) over the OWNED support only) is
        replaced by ``template_norm + flux_beyond_stamp`` (the containment-
        corrected TRUE total, crowding-independent by construction), so a
        close neighbour's ownership boundary no longer leaks into the
        catalog-tied estimator the way it still could after 4b:

        tcor_int = F444W_total_moph / [(template_norm + flux_beyond_stamp) * apF_book]
          F444W_total_moph (= f444w_ktot) is, for the floored/PSF-converged
          population (``apcor_from_psf`` True -- exactly where the ownership-
          masked leak lived, docs stage4c brief Sec "KEY GROUNDING FINDING"),
          set DIRECTLY to ``template_norm + flux_beyond_stamp``: the exact
          point-source total in the faint pure-PSF limit
          (``A_src/c_det``), no photutils/PSF-EE lookup needed. For the
          non-floored (bright/extended, real photutils Kron) population it
          stays ``kron_flux_model / EE_true_444(r_kron_circ)`` as before
          (``kron_flux_model``/``r_kron_circ`` from :meth:`_model_kron`,
          Kron radius floored at the catalog color-aperture radius
          ``r_floor_pix`` = 0.5 * catalog[f444w_aper_col] / pscale_ref) --
          these sources have large owned support so the clipped fraction is
          small, and the masked photutils Kron preserves real extended
          structure a PSF-total would throw away. Without a usable
          ``f444w_aper_col`` this degrades to the true-normalized
          point-source form, ``tcor_int = 1/apF_corr`` (already unmasked per
          4b) with ``f444w_ktot = template_norm + flux_beyond_stamp`` for
          consistency, for every source (noted once).
        s_cat = ftot / F444W_total_moph   (bad_value without a POSITIVE catalog total)
        apcor  = apcor1 * tcor_int * s_cat   (the full released correction;
                 bad_value when s_cat is bad -- REPURPOSED from Stage 3a; note
                 f444w_ktot always CANCELS in tcor_int*s_cat = ftot/denom, so
                 est3cat is fixed by the denom change alone -- docs stage4c
                 brief Sec 1)

        Per parent id, ap_model is accumulated over any multi-component
        templates; the corrections and residual are computed once.

        Writes ap_model_{idx}, apcor1_{idx}, totcor1_{idx}, apf_data_{idx},
        tcor_int_{idx}, s_cat_{idx}, f444w_ktot_{idx} (= F444W_total_moph),
        apcor_{idx} (repurposed product), res_sum_{idx}, res_seg_{idx},
        ap_flux_{idx}, ap_flux_est1_{idx}, ap_flux_est2_{idx},
        ap_flux_est3int_{idx}, ap_flux_est3cat_{idx}, and the matching
        err_ap_flux_est1_{idx}, err_ap_flux_est2_{idx},
        err_ap_flux_est3int_{idx}, err_ap_flux_est3cat_{idx}.
        """
        cfg = self.config
        id_to_row = {int(i): k for k, i in enumerate(cat["id"])}
        r_img_pix = self._resolve_image_ap_radius_pix(idx, cfg)
        pscale_ref = self._pixel_scale_arcsec(self.wcs[0] if self.wcs is not None else None)

        if r_orig_pix is None:
            pscale_img = self._pixel_scale_arcsec(self.wcs[idx] if self.wcs is not None else None)
            r_orig_pix = r_img_pix * pscale_img / pscale_ref if (pscale_img and pscale_ref) else r_img_pix

        # Stage-4b band-side EE radius, in the BAND PSF's NATIVE pixel scale
        # (constant across sources; computed once). psf_hires (detection) is on
        # the reference grid so its EE uses r_orig_pix directly, but psf_band is
        # stored on its native grid; in upsample mode the fit grid overwrites
        # wcs[idx]=wcs[0], so r_img_pix would be in fine 0.04" fit pixels while
        # the band PSF is ~0.11"/px native -- measuring band EE at r_img_pix
        # then samples the wrong physical radius (EE -> ~1, band_det_ratio
        # inflated, totcor1 depressed for exactly the crowded faint sources
        # this stage fixes). r_orig_pix is the reference-grid (0.04") aperture
        # radius in EVERY mode, so converting from it via the captured native
        # scales (_native_pscale[0]=ref, [idx]=band; commit 716811b Stage-2
        # pattern) is frame-robust: r_band = r_orig_pix * pscale_ref /
        # pscale_band. Falls back to r_img_pix when the native scales are
        # unavailable (WCS-less unit tests / legacy runs).
        r_band_pix = r_img_pix
        _psn = getattr(self, "_native_pscale", None)
        if _psn and len(_psn) > idx and _psn[0] and _psn[idx]:
            r_band_pix = r_orig_pix * float(_psn[0]) / float(_psn[idx])

        # Stage-3b catalog color-aperture floor (docs Sec 5.4; same per-source
        # ingestion idiom as the deleted rung-1 low-SNR blend, commit 91c96d0):
        # r_floor_pix = 0.5 * catalog[f444w_aper_col] / pscale_ref, on the
        # reference (F444W) grid. Column not configured/available -> tcor_int
        # falls back to 1/apF_corr for every source (noted once).
        r_floor_by_id: dict[int, float] = {}
        cat_src = getattr(self, "catalog", None)
        if (cfg.f444w_aper_col and cat_src is not None and pscale_ref
                and cfg.f444w_aper_col in cat_src.colnames):
            ids_src = np.asarray(cat_src["id"]).astype(int)
            uas = np.asarray(cat_src[cfg.f444w_aper_col], dtype=float)
            for sid, ua in zip(ids_src, uas):
                if np.isfinite(ua) and ua > 0:
                    r_floor_by_id[int(sid)] = 0.5 * float(ua) / pscale_ref
        have_r_floor = len(r_floor_by_id) > 0
        if not have_r_floor and not getattr(self, "_warned_no_f444w_aper_col", False):
            print(
                f"  tcor_int: no usable r_floor ('{cfg.f444w_aper_col}' column "
                "or WCS pixel scale missing) -- falling back to 1/apF_corr "
                "(true-normalized point-source total) for every source"
            )
            self._warned_no_f444w_aper_col = True

        for name in (
            f"ap_model_{idx}",
            f"apcor_{idx}",
            f"apcor1_{idx}",
            f"totcor1_{idx}",
            f"apf_data_{idx}",
            f"tcor_int_{idx}",
            f"s_cat_{idx}",
            f"f444w_ktot_{idx}",
            f"res_sum_{idx}",
            f"res_seg_{idx}",
            f"ap_flux_{idx}",
            f"ap_flux_est1_{idx}",
            f"ap_flux_est2_{idx}",
            f"ap_flux_est3int_{idx}",
            f"ap_flux_est3cat_{idx}",
            f"err_ap_flux_est1_{idx}",
            f"err_ap_flux_est2_{idx}",
            f"err_ap_flux_est3int_{idx}",
            f"err_ap_flux_est3cat_{idx}",
        ):
            if name not in cat.colnames:
                cat[name] = cfg.bad_value

        orig_by_id = {t.id: t for t in orig_templates} if orig_templates else {}
        use_tcor = f444w_totals is not None

        # F444W neighbour-subtracted residual map (built once, reused across
        # bands): the apf_data diagnostic is the REAL neighbour-subtracted
        # F444W aperture flux instead of the noise-level template sum. For each
        # source we add its own model back via aperture-sum linearity:
        #   ap_f_data = aper(residual + model_i) = aper(residual) + ap_f_template.
        f444w_res = None
        if orig_templates is not None:
            if getattr(self, "_f444w_residual", None) is None:
                self._f444w_residual = self._build_f444w_residual(orig_templates)
            f444w_res = self._f444w_residual

        # Bin factor between the high-res segmap (F444W) and this image's
        # residual. k=1 when shapes already match (e.g. upsample mode).
        if self.wcs is not None and self.segmap.shape != residual.shape:
            k = bin_factor_from_wcs(self.wcs[0], self.wcs[idx])
        else:
            k = 1

        print(
            f"  Computing aperture corrections (image {idx}, {len(templates)} sources, "
            f"{'with catalog tie (s_cat)' if use_tcor else 'internal (tcor_int) only'})"
        )

        # psf_hires (F444W detection PSF) is still needed for _model_kron's
        # EE_true_444(r_kron) lookup (docs/aperture_corrections.md Sec 5.4);
        # the PSF-EE apB_corr/apF_corr branch itself is gone (Sec 5.1 -- the
        # unified template makes apB/apF a single footprint-truncated fraction
        # for every source, corrected uniformly by the per-source truncation
        # term below). EE cached per PSF-region id (few distinct).
        psf_hires = self.psfs[0] if (self.psfs is not None and len(self.psfs) > 0) else None
        _ee_cache: dict = {}

        def _psf_ee(psfmap, ra, dec, radius, *, with_containment: bool = True):
            if psfmap is None:
                return None
            # Cache key: (psfmap identity, region, radius) -- NOT id(psf):
            # PSFRegionMap.get_psf returns a fresh ndarray view per call and
            # CPython reuses freed ids, so id(psf) collided across regions
            # (pre-existing since Phase A). The region key is resolved ONCE
            # and reused for the PSF and its containment so they cannot diverge.
            if isinstance(psfmap, PSFRegionMap):
                region = psfmap.resolve_key(ra, dec)
                psf = psfmap.psfs[region]
                c = psfmap.containment
                containment = 1.0 if c is None else float(c if np.isscalar(c) else c[region])
            else:
                region = None
                psf = psfmap
                containment = 1.0
            if psf is None:
                return None
            # with_containment is part of the key: Stage-4b's band-side stamp-
            # EE ratio (docs Sec 5.1/6) needs the RAW stamp EE, not the
            # true-total one tcor_int's ee_kron lookup uses -- same cache,
            # two independent entries so neither call site's value is stale.
            key = (id(psfmap), region, float(radius), with_containment)
            if key not in _ee_cache:
                try:
                    # True-total normalization (docs/aperture_corrections.md
                    # Sec 4.1/5.2): psf_ee_at_radius is stamp-normalized, so
                    # multiply by the region's containment (fraction of the
                    # PSF's true total flux in the stamp). ndarray PSFs -> 1.0.
                    val = utils.psf_ee_at_radius(psf, radius)
                    _ee_cache[key] = val * containment if with_containment else val
                except Exception:  # pragma: no cover - degenerate PSF
                    _ee_cache[key] = None
            return _ee_cache[key]

        def _psf_containment(psfmap, ra, dec):
            """Per-region PSF stamp containment alone (docs Sec 5.2), 1.0
            default for a non-PSFRegionMap/unset containment -- mirrors
            _psf_ee's containment handling (loud warning already emitted at
            PSFRegionMap load time, not here). Feeds the Stage-4b band-side
            c_b/c_det factor (ruling Sec "Ruling"), which is evaluated fresh
            from self.psfs[0]/self.psfs[idx] here -- independent of whatever
            detection_psf was passed to Templates.extract_templates -- so the
            two agree in a real run (same PSFRegionMap object) without coupling
            this method to the extraction call site."""
            if isinstance(psfmap, PSFRegionMap):
                c = psfmap.get_containment(ra, dec)
                return float(c) if (np.isfinite(c) and c > 0) else 1.0
            return 1.0

        # Accumulate the model aperture flux per parent id (multi-component
        # templates share an id); the correction factors and the residual are
        # computed once per parent. apcor1, tcor_int and s_cat are kept as
        # SEPARATE factors (and columns) and never algebraically collapsed.
        model_acc: dict[int, float] = defaultdict(float)
        per: dict[int, dict] = {}

        for tmpl, fl in tqdm(zip(templates, fluxes), desc="Aperture corrections", total=len(templates)):
            row = id_to_row.get(int(tmpl.id))
            if row is None:
                continue

            orig_t = orig_by_id.get(tmpl.id)
            if orig_t is None:
                continue

            template_norm_i = orig_t.template_norm
            if template_norm_i <= 0:
                continue

            # Bookkeeping fraction: ALWAYS the fitted convolved template's own
            # aperture sum, since fl is defined against that unit-sum template
            # (real-flux invariant -- docs/aperture_corrections.md Sec 4.2/5.3).
            apB_book = self._aperture_sum_on_template(tmpl, r_img_pix)
            if apB_book <= 0:
                continue

            # Cutout-frame position with the cutout-adjusted WCS (the CRPIX is
            # shifted to the cutout in Template.__init__; see the downsample
            # convention in templates.py). position_original is the full-image
            # frame and would give a wrong sky position -> wrong PSF region.
            # Needed for the Stage-3b Kron->total EE lookup (docs Sec 5.4),
            # which runs on all sources.
            pos = orig_t.input_position_cutout
            if getattr(orig_t, "wcs", None) is not None:
                ra_dec = orig_t.wcs.wcs_pix2world(pos[0], pos[1], 0)
            else:
                ra_dec = pos

            # Model flux inside the aperture (low-res grid), summed over any
            # multi-component templates that share this parent id.
            # APPROXIMATION (multi-component only): apcor1 below is computed from
            # the FIRST (primary) component's shape and applied to this summed
            # ap_model. This is EXACT for single-component sources (the default;
            # multi_tmpl_psf_core/colour are off). For added PSF-core/colour
            # components the secondary has no well-defined native-F444W shape, so
            # its aperture correction uses the primary's apF/apB ratio -> a small
            # bias on the (usually small) secondary flux. TODO: per-component apF.
            model_acc[row] += fl * apB_book

            if row in per:
                continue  # once-per-parent quantities already computed

            # --- once-per-parent correction factors and residual ---
            # High-res aperture fraction (footprint-truncated fitted template).
            apF_book = self._aperture_sum_on_template(orig_t, r_orig_pix)

            # Per-source truncation (docs/aperture_corrections.md Sec 5.1/6):
            # the unified template's PSF-EE correction path collapses to one
            # truncation term applied to BOTH apB_book and apF_book, replacing
            # the old apcor_from_psf PSF-curve-of-growth branch. flux_beyond_stamp
            # (set in _extended_composite) is the PSF-extrapolated, core-anchored
            # estimate of this source's flux landing outside the cutout, in the
            # same real-flux units as template_norm (Sigma(H)).
            flux_beyond = float(getattr(orig_t, "flux_beyond_stamp", 0.0) or 0.0)
            trunc_denom = template_norm_i + flux_beyond
            trunc = template_norm_i / trunc_denom if trunc_denom > 0 else 1.0

            # Stage-4b (docs Sec 5.1/6, ruling "A+cb"): correction-only
            # crowding delta -- PSF flux the fit's own support excluded
            # (neighbour territory and/or beyond ee_reach) that is still the
            # source's own light. Sigma(H_corr) + fb_corr telescopes to the
            # SAME trunc_denom above (the full-stamp PSF piece cancels
            # algebraically), so only the APERTURE-SUM numerators change here.
            # flux_beyond_aper (set in _extended_composite, detection frame,
            # a per-template scalar) is reshaped to band space via the
            # per-region stamp-EE ratio below rather than a per-source band
            # convolution (ruling Sec "risks" item 5: the matching kernel
            # maps the detection PSF onto the band PSF by construction).
            flux_beyond_aper = float(getattr(orig_t, "flux_beyond_aper", 0.0) or 0.0)
            c_det = _psf_containment(psf_hires, ra_dec[0], ra_dec[1])
            band_psf = self.psfs[idx] if (self.psfs is not None and idx < len(self.psfs)) else None
            c_b = _psf_containment(band_psf, ra_dec[0], ra_dec[1])
            # Band EE at r_band_pix (band NATIVE grid, converted above); the
            # detection EE stays at r_orig_pix (reference grid, already correct).
            ee_band_stamp = _psf_ee(band_psf, ra_dec[0], ra_dec[1], r_band_pix, with_containment=False)
            ee_det_stamp = _psf_ee(psf_hires, ra_dec[0], ra_dec[1], r_orig_pix, with_containment=False)
            band_det_ratio = (
                ee_band_stamp / ee_det_stamp
                if (ee_band_stamp is not None and ee_det_stamp) else 1.0
            )
            flux_beyond_aper_band = flux_beyond_aper * band_det_ratio

            apF_corr = apF_book * trunc + (flux_beyond_aper / trunc_denom if trunc_denom > 0 else 0.0)
            apB_corr_book = apB_book * trunc + (flux_beyond_aper_band / trunc_denom if trunc_denom > 0 else 0.0)
            # Band side ONLY (ruling): the stamp-built kernel maps the
            # c_det-truncated detection PSF onto the c_b-truncated band PSF,
            # so the model's band-aperture flux is high by c_det/c_b -- the
            # detection side needs no such factor (flux_beyond_aper above
            # already true-normalizes it via c_det, embedded in trunc_denom).
            apB_corr = apB_corr_book * (c_b / c_det) if c_det > 0 else apB_corr_book

            # Real-unit template aperture fluxes (template_norm restores image
            # flux units; it cancels in apcor1 but is required so tcor_int and
            # s_cat are dimensionless).
            ap_b_corr = template_norm_i * apB_corr  # low-res convolved template flux in aperture
            ap_f_corr = template_norm_i * apF_corr if apF_corr > 0 else 0.0  # high-res template flux in aperture

            # Shape correction: high-res / low-res aperture flux (real units).
            # trunc cancels exactly (both apF_corr and apB_corr share the same
            # trunc factor); the c_b/c_det band factor deliberately does NOT
            # (ruling: apcor1 gains x c_det/c_b -- a genuine shape effect of
            # the stamp-containment mismatch between the two PSFs).
            apcor1 = ap_f_corr / ap_b_corr if (ap_b_corr > 0 and ap_f_corr > 0) else 1.0
            # Internal aperture-to-total (design-doc Eq. 7; = IDL totcor). trunc
            # survives here (aperture-to-TOTAL, not a shape ratio).
            totcor1 = 1.0 / apB_corr if apB_corr > 0 else 1.0

            # apf_data diagnostic: the REAL neighbour-subtracted F444W aperture
            # flux (template model_i + residual), measured on data. Bookkeeping
            # side -- ALWAYS the fitted original template's own fraction, so
            # aperture-sum linearity holds against the F444W residual map
            # (built from the same templates). No longer feeds any correction
            # (docs/aperture_corrections.md Sec 5.4); kept as a diagnostic only.
            ap_f_book = template_norm_i * apF_book if apF_book > 0 else 0.0
            ap_f_data = ap_f_book
            if f444w_res is not None:
                ap_f_data = ap_f_book + self._aperture_sum_on_map(f444w_res, orig_t, r_orig_pix)

            ftot = f444w_totals.get(int(tmpl.id)) if use_tcor else None
            has_ftot = ftot is not None and np.isfinite(ftot)

            # --- Stage-3b two-step catalog tie (docs Sec 5.4) -- evaluated
            # entirely on MODELS, never on apf_data/measured aperture flux.
            # TODO(multi-band): tcor_int/f444w_ktot/s_cat are F444W-side and
            # band-independent, yet recomputed per band in a multi-band run;
            # single-band runs (the production pattern) are unaffected. ---
            r_floor_pix = r_floor_by_id.get(int(tmpl.id)) if have_r_floor else None
            if r_floor_pix is None or not (r_floor_pix > 0):
                # No usable catalog color-aperture radius for this source
                # (column missing/not configured, or id absent from the
                # lookup): the true-normalized point-source fallback.
                # tcor_int itself is already unmasked (apF_corr carries the
                # Stage-4b trunc/flux_beyond_aper correction); f444w_ktot is
                # unmasked here too (Stage-4c D1) for consistency with the
                # r_floor branch below -- was `template_norm_i * apF_book *
                # tcor_int` (a MASKED numerator, apF_book, times the unmasked
                # tcor_int), which re-introduced the ownership-masked leak
                # into this diagnostic column. This fallback IS the point-
                # source case, so its total is exactly trunc_denom, the same
                # identity the floored branch below uses.
                tcor_int_ok = apF_corr > 0
                if tcor_int_ok:
                    tcor_int = 1.0 / apF_corr
                    f444w_ktot = trunc_denom
                else:
                    tcor_int = float(cfg.bad_value)
                    f444w_ktot = float(cfg.bad_value)
            else:
                # Stage-4c (D1 whole system, D2 option a): the tie
                # denominator uses the UNMASKED total (trunc_denom =
                # template_norm_i + flux_beyond, already computed above for
                # the apF_corr/apB_corr block) in place of the ownership-
                # MASKED template_norm_i alone -- this is the fix for the
                # released est3cat leak (docs stage4c brief Sec 1: f444w_ktot
                # cancels in tcor_int*s_cat, so this denom change alone makes
                # est3cat crowding-independent). apF_book itself is left
                # untouched (bookkeeping fraction, per the brief).
                denom = trunc_denom * apF_book

                # apcor_from_psf performance shortcut: a PSF-converged
                # faint/compact template's Kron radius floors anyway, so skip
                # photutils SourceCatalog and use the floor circle directly
                # (also the scientifically right faint limit per the doc).
                # Stage-4c: this is exactly the population where the
                # ownership-masked Kron leak lived (docs stage4c brief Sec
                # "KEY GROUNDING FINDING"), so f444w_ktot is set DIRECTLY to
                # the unmasked point-source total instead of measuring a
                # masked circular Kron flux and dividing by EE(r_floor) (which
                # reproduced the SAME masked total the fallback branch used
                # to) -- no _model_kron/PSF-EE lookup needed at all here
                # (cheaper, and exact in the faint pure-PSF limit where
                # trunc_denom == A_src/c_det, docs Sec 2).
                if getattr(orig_t, "apcor_from_psf", False):
                    f444w_ktot = trunc_denom
                    tcor_int_ok = bool(denom > 0 and f444w_ktot > 0)
                    if tcor_int_ok:
                        tcor_int = f444w_ktot / denom
                    else:
                        tcor_int = float(cfg.bad_value)
                        f444w_ktot = float(cfg.bad_value)
                else:
                    # Non-floored (bright/extended) population: keep the
                    # masked photutils Kron measurement (docs stage4c brief
                    # Sec 2 "recommended route" -- these sources have large
                    # owned support so the clipped fraction is small, and the
                    # masked Kron preserves real extended structure that the
                    # unmasked PSF-total substitution above would throw away).
                    kron_flux_model, r_kron_circ = self._model_kron(
                        orig_t, int(tmpl.id), r_floor_pix, use_source_catalog=True,
                    )
                    ee_kron = _psf_ee(psf_hires, ra_dec[0], ra_dec[1], r_kron_circ)
                    if ee_kron is None or ee_kron <= 0:
                        ee_kron = self._aperture_sum_on_template(orig_t, r_kron_circ)
                    tcor_int_ok = bool(
                        ee_kron and ee_kron > 0 and kron_flux_model > 0 and denom > 0
                    )
                    if tcor_int_ok:
                        f444w_ktot = kron_flux_model / ee_kron
                        tcor_int = f444w_ktot / denom
                    else:
                        tcor_int = float(cfg.bad_value)
                        f444w_ktot = float(cfg.bad_value)

            # s_cat requires a POSITIVE catalog total: a negative f_f444w (an
            # F444W non-detection) cannot define a total-flux system, so the
            # tie is meaningless there -> s_cat/apcor/est3cat go bad_value
            # (tcor_int/f444w_ktot/est3int are catalog-independent, unaffected).
            # No upper bound on s_cat: the large-positive tail is a ratio
            # artifact that cancels in est3cat (closure: est3cat = f_f444w *
            # [model-shape fraction] + res_sum); left visible as a diagnostic.
            s_cat_ok = has_ftot and float(ftot) > 0 and tcor_int_ok and f444w_ktot > 0
            s_cat = float(ftot) / f444w_ktot if s_cat_ok else float(cfg.bad_value)

            # Residual within the measurement aperture (disk), added UNSCALED.
            # Same aperture geometry as _aperture_sum_on_template applied to the
            # residual patch, so ap_flux = ap_model + res_sum exactly.
            # TODO(blend residual double-count): for two close sources the aperture
            # disks overlap, so background residual in the shared region is added to
            # both sources' corrected flux values. Full fix would partition the
            # shared residual.
            # For now we at least exclude pixels that explicitly belong to OTHER
            # sources' segments from this source's residual aperture.
            res_patch = residual[tmpl.slices_original]
            other = self._other_source_mask(tmpl, int(tmpl.id), k)
            if other is not None and other.shape == res_patch.shape:
                res_patch = res_patch * (~other)
            xc = tmpl.input_position_cutout[0] - tmpl.slices_cutout[1].start
            yc = tmpl.input_position_cutout[1] - tmpl.slices_cutout[0].start
            res_sum = float(
                aperture_photometry(
                    res_patch,
                    CircularAperture((xc, yc), r=r_img_pix),
                    method="exact",
                )["aperture_sum"][0]
            )
            # Residual over the full segmap (diagnostic only; not used in f3).
            res_seg = self._residual_segmap_sum(residual, int(tmpl.id), orig_t, k)

            per[row] = dict(
                apcor1=apcor1, totcor1=totcor1, apf_data=ap_f_data,
                tcor_int=tcor_int, tcor_int_ok=tcor_int_ok,
                s_cat=s_cat, s_cat_ok=s_cat_ok, f444w_ktot=f444w_ktot,
                res_sum=res_sum, res_seg=res_seg,
            )

        # Write per-parent Estimator-3 results. Corrections are applied as the
        # explicit product apcor1 * tcor_int [* s_cat] (never pre-collapsed).
        for row, d in per.items():
            ap_model = model_acc[row]
            apcor1 = d["apcor1"]
            totcor1 = d["totcor1"]
            tcor_int = d["tcor_int"]
            s_cat = d["s_cat"]
            res_sum = d["res_sum"]
            # Fractional profile-fit error (err/flux) from the sparse solve,
            # used to propagate a photometric error onto each estimator below.
            # flux_{idx}/err_{idx} are populated by _update_catalog_with_fluxes
            # before this method in a full run; guard for their absence so the
            # method is still callable standalone (unit tests) -> errors stay bad.
            if f"flux_{idx}" in cat.colnames and f"err_{idx}" in cat.colnames:
                flux_fit = cat[f"flux_{idx}"][row]
                err_fit = cat[f"err_{idx}"][row]
                frac_err = (
                    float(err_fit / flux_fit)
                    if (np.isfinite(flux_fit) and flux_fit > 0 and np.isfinite(err_fit))
                    else float("nan")
                )
            else:
                frac_err = float("nan")
            cat[f"ap_model_{idx}"][row] = ap_model
            cat[f"apcor1_{idx}"][row] = apcor1
            cat[f"totcor1_{idx}"][row] = totcor1
            cat[f"apf_data_{idx}"][row] = d["apf_data"]
            cat[f"tcor_int_{idx}"][row] = tcor_int
            cat[f"s_cat_{idx}"][row] = s_cat
            cat[f"f444w_ktot_{idx}"][row] = d["f444w_ktot"]
            cat[f"res_sum_{idx}"][row] = res_sum
            cat[f"res_seg_{idx}"][row] = d["res_seg"]
            cat[f"ap_flux_{idx}"][row] = ap_model + res_sum
            # Estimator 1 (internal-IDL, exact): aperture flux on the
            # neighbour-subtracted image (ap_model + res_sum) scaled to total
            # by totcor1.
            cat[f"ap_flux_est1_{idx}"][row] = (ap_model + res_sum) * totcor1
            # Estimator 2 (internal-IDL): model aperture flux scaled to total
            # by the internal template curve of growth (totcor1 = 1/apB), +
            # residual.
            cat[f"ap_flux_est2_{idx}"][row] = ap_model * totcor1 + res_sum
            # Per-estimator errors: the fractional profile-fit error carried onto
            # each estimator's corrected MODEL flux (correction * ap_model, i.e.
            # est minus its residual term). Multiplicative corrections are treated
            # as noiseless; the res_sum aperture pixel-noise term is NOT included.
            # est1 and est2 share the model part (totcor1 * ap_model) -> same error.
            if np.isfinite(frac_err):
                err_est12 = abs(totcor1 * ap_model) * frac_err
                cat[f"err_ap_flux_est1_{idx}"][row] = err_est12
                cat[f"err_ap_flux_est2_{idx}"][row] = err_est12
            # Estimator 3int (internal-Kron total): model aperture flux scaled
            # by apcor1 * tcor_int, + unscaled residual.
            if d["tcor_int_ok"]:
                cat[f"ap_flux_est3int_{idx}"][row] = ap_model * apcor1 * tcor_int + res_sum
                if np.isfinite(frac_err):
                    cat[f"err_ap_flux_est3int_{idx}"][row] = (
                        abs(apcor1 * tcor_int * ap_model) * frac_err
                    )
            else:
                cat[f"ap_flux_est3int_{idx}"][row] = cfg.bad_value
            # apcor (repurposed): the full released correction apcor1 * tcor_int
            # * s_cat, and Estimator 3cat (catalog-tied release) built from it --
            # both bad_value when s_cat is bad (no catalog total / s_cat guard failed).
            if d["s_cat_ok"]:
                apcor_released = apcor1 * tcor_int * s_cat
                cat[f"apcor_{idx}"][row] = apcor_released
                cat[f"ap_flux_est3cat_{idx}"][row] = ap_model * apcor_released + res_sum
                if np.isfinite(frac_err):
                    cat[f"err_ap_flux_est3cat_{idx}"][row] = (
                        abs(apcor_released * ap_model) * frac_err
                    )
            else:
                cat[f"apcor_{idx}"][row] = cfg.bad_value
                cat[f"ap_flux_est3cat_{idx}"][row] = cfg.bad_value

    def run(self, config: FitConfig | None = None) -> tuple[Table, list[np.ndarray]]:
        """Run photometry on the configured images.

        Returns
        -------
        Table
            Catalog containing flux measurements for each image.
        list of ndarray
            Residual images corresponding to each fitted image.
        """

        images = self.images
        segmap = self.segmap
        catalog = self.catalog
        psfs = self.psfs
        weights = self.weights
        kernels = self.kernels
        if kernels is None:
            kernels = [None] * len(images)
        wcs = self.wcs
        if config is None:
            config = self.config
        else:
            self.config = config

        print(f"Pipeline (start) memory: {memory():.1f} GB")
        print(f"Pipeline config: {config}")

        # test for NaN values in images and weights
        for i in range(len(images)):
            if images[i] is None:
                assert np.all(np.isfinite(images[i])), "Image contains NaN values"
            if weights[i] is not None:
                assert np.all(np.isfinite(weights[i])), "Weights contain NaN values"

        if catalog is None:
            # use astropy to make catalog from image[0] + segmap
            print("No catalog provided, generating from segmap")
            raise NotImplementedError("Catalog generation not implemented yet")
        else:
            cat = catalog.copy()
            cat = cat["id", "x", "y"]  # minimal set required
            if config.aperture_catalog is not None:
                cat[config.aperture_catalog] = catalog[config.aperture_catalog]
            if config.f444w_col is not None and config.f444w_col in catalog.colnames:
                cat[config.f444w_col] = catalog[config.f444w_col]

        # Native per-band pixel scales, captured BEFORE the fit loop's upsample step
        # overwrites wcs[idx] with wcs[0]. The band PSF (self.psfs[idx]) is stored on
        # its native grid, so the aperture-correction PSF EE must be measured at the
        # aperture radius in that native scale, not the (possibly upsampled) fit grid.
        self._native_pscale = [
            (self._pixel_scale_arcsec(w) if w is not None else None)
            for w in (self.wcs if self.wcs is not None else [])
        ]

        # --- Representative detection-PSF growth curve (Estimator-3 plan v5) ---
        # Cache one curve of growth near the mosaic centre so the 50/95/99% EE
        # radii are available downstream for ownership sizing (rhalf_det = R50)
        # and the template max-size cap (R95/R99). Reuses utils.psf_ee_radius_pix.
        # psfs[0] is the detection-band (F444W = images[0]) PSF; all dereferences
        # are guarded so legacy runs without PSFs are unaffected.
        extend_mode = str(getattr(config, "template_extend_mode", "none"))
        ee_cap = float(config.extend_template_ee)
        self.detection_psf = None
        self.ee_radii_pix: dict[float, float] = {}
        self._f444w_residual = None  # rebuilt in _add_aperture_photometry (F444W neighbour-sub map)
        min_size = 8
        r_fill = 0.0  # extension fill radius (F444W px); 0 -> no extension
        if psfs is not None and len(psfs) > 0 and psfs[0] is not None and wcs is not None:
            psf0 = psfs[0]
            # A spatially varying PSFRegionMap -> pick the widest region so every
            # source has room; a plain ndarray is used directly.
            if isinstance(psf0, PSFRegionMap):
                rep_psf = max(
                    (np.asarray(p, dtype=float) for p in psf0.psfs),
                    key=lambda p: utils.psf_ee_radius_pix(p, ee_cap),
                )
            else:
                rep_psf = np.asarray(psf0, dtype=float)
            self.detection_psf = rep_psf
            for frac in sorted({0.5, 0.95, 0.99, ee_cap}):
                try:
                    self.ee_radii_pix[frac] = float(utils.psf_ee_radius_pix(rep_psf, frac))
                except Exception as exc:  # pragma: no cover - PSF shape guard
                    logger.warning("psf_ee_radius_pix(%.2f) failed: %s", frac, exc)
            # When extending, choose the fill radius and pre-size cutouts to hold
            # it *before* extraction so slice bookkeeping is correct from birth.
            # r_fill = max(R95, aperture_radius_F444W + kernel_half_width): the
            # template must cover the measurement aperture (plus a convolution
            # margin so the convolved apB is valid out to that radius), and never
            # be smaller than the R95 EE cap.
            if extend_mode != "none":
                r95 = self.ee_radii_pix.get(ee_cap)
                r_fill = float(r95) if r95 is not None else 0.0

                # F444W-grid aperture radius (scalar aperture; arcsec or pixels).
                r_orig = None
                scalar_ap = (
                    np.isscalar(config.aperture_diam)
                    and not isinstance(config.aperture_diam, str)
                )
                if scalar_ap and config.aperture_units == "arcsec":
                    pscale_ref = self._pixel_scale_arcsec(wcs[0])
                    if pscale_ref:
                        r_orig = 0.5 * float(config.aperture_diam) / pscale_ref
                elif scalar_ap and config.aperture_units == "pix":
                    # aperture already in detection-grid pixels
                    r_orig = 0.5 * float(config.aperture_diam)

                # Stage-4b crowding correction requires a scalar detection-grid
                # aperture radius (aperture_radius_pix below): flux_beyond_aper
                # is a per-source SCALAR keyed to one aperture, so a per-band
                # aperture_diam ARRAY (scalar_ap False -> r_orig None ->
                # aperture_radius_pix None) leaves flux_beyond_aper == 0 for
                # every source, silently reverting the correction to Stage-4
                # behavior. Warn loudly (default-off features must not fail
                # silently) -- fluxes are still valid, just without the crowding
                # aperture-to-total fix.
                if r_orig is None and not scalar_ap:
                    logger.warning(
                        "aperture_diam is a per-band array: the Stage-4b crowding "
                        "correction (docs/aperture_corrections.md Sec 5.1/6) is "
                        "DISABLED (flux_beyond_aper == 0 for every source, totcor1/"
                        "apcor1 revert to the Stage-4 masked-support values). Use a "
                        "scalar aperture_diam to enable it."
                    )

                # Largest matching-kernel EFFECTIVE half-width across the fitted
                # bands (the 95% encircled radius of |K|, NOT the zero-padded
                # array size -- otherwise large kernels would inflate template
                # sizes/memory). This is the convolution margin so the convolved
                # apB is valid out to the aperture radius.
                kernel_hw = 0.0
                for kern in (kernels or []):
                    arr = None
                    if isinstance(kern, PSFRegionMap):
                        arr = np.asarray(kern.psfs[0], dtype=float) if len(kern.psfs) else None
                    elif kern is not None:
                        arr = np.asarray(kern, dtype=float)
                    if arr is not None and arr.ndim == 2:
                        a = np.abs(arr)
                        if a.sum() > 0:
                            try:
                                kernel_hw = max(kernel_hw, utils.psf_ee_radius_pix(a, 0.95))
                            except Exception:  # pragma: no cover - degenerate kernel
                                pass

                if r_orig is not None:
                    r_fill = max(r_fill, r_orig + kernel_hw)

                if r_fill > 0:
                    floor = 2 * int(np.ceil(r_fill)) + 1
                    floor += floor % 2
                    min_size = max(min_size, floor)

        # Template-extension parameters (Phase 2). The ownership/halo reach is
        # r_fill = max(R95, aperture_radius + kernel_half_width), so the extended
        # template covers the measurement aperture. Falls back to no extension if
        # the PSF growth-curve / detection PSF is unavailable.
        extract_kw: dict = {}
        if extend_mode != "none":
            if r_fill > 0 and psfs is not None and psfs[0] is not None:
                det_weight = (
                    weights[0] if (weights is not None and len(weights) > 0 and weights[0] is not None)
                    else None
                )
                psf_ee_radius = self.ee_radii_pix.get(float(config.extend_template_ee))
                extract_kw = dict(
                    extend_mode=extend_mode,
                    detection_psf=psfs[0],
                    detection_weight=det_weight,
                    max_radius_pix=float(r_fill),
                    psf_ee_radius_pix=float(psf_ee_radius) if psf_ee_radius is not None else None,
                    aperture_radius_pix=float(r_orig) if r_orig is not None else None,
                    fit_snrlo_psf=float(config.fit_snrlo_psf),
                    wings_snr_psf=float(config.wings_snr_psf),
                    template_blend_p=float(config.template_blend_p),
                    template_blend_annulus=float(config.template_blend_annulus),
                )
                logger.info(
                    "Template extension (auto): every template extended to no more "
                    "than %.1f pix (PSF-wing reach %.1f pix @ %.0f%% EE)",
                    r_fill,
                    psf_ee_radius if psf_ee_radius is not None else r_fill,
                    100.0 * float(config.extend_template_ee),
                )
            else:
                logger.warning(
                    "template_extend_mode=%s but PSF growth-curve radii unavailable; "
                    "extracting truncated templates (no extension).", extend_mode,
                )

        # Rough cut: skip sources that cannot touch any fitted band's coverage.
        keep = _sources_with_coverage(segmap, cat, wcs, weights, min_size)
        n_cut = int((~keep).sum())
        if n_cut:
            print(
                f"Rough cut: {n_cut} of {len(cat)} sources have no coverage in any "
                "fitted band; skipping their templates."
            )
        xs = np.asarray(cat["x"], dtype=float)[keep]
        ys = np.asarray(cat["y"], dtype=float)[keep]

        self.tmpls = Templates(min_size=min_size)
        self.tmpls.extract_templates(
            images[0],
            segmap,
            list(zip(xs, ys)),
            wcs=wcs[0] if wcs is not None else None,
            **extract_kw,
        )
        templates = self.tmpls.templates
        for t in templates:
            assert np.all(np.isfinite(t.data)), "Templates contain NaN values"

        if catalog is not None and "flag_star" in catalog.colnames:
            star_ids = set(int(r["id"]) for r in catalog if r["flag_star"] == 1)
            for t in templates:
                if int(t.id) in star_ids:
                    t.is_star = True
            logger.info("Marked %d templates as stars (excluded from astrometry)", sum(t.is_star for t in templates))

        ndropped = len(cat) - len(templates)
        # @@@ this is because of reliance of x,y in catalog -> use segmap + weight?
        print(f"Pipepline: {len(templates)} extracted templates, dropped {ndropped}.")
        print(f"Pipeline (templates) memory: {memory():.1f} GB")

        residuals: list[np.ndarray] = []
        self.all_templates: list[Template] = []
        self.all_scenes: list[Scene] = []

        # Build F444W total-flux lookup for Mode B aperture correction (Yoshi's formula).
        f444w_totals: dict[int, float] | None = None
        if config.f444w_col is not None:
            if config.f444w_col in cat.colnames:
                f444w_totals = {
                    int(cat["id"][k]): float(cat[config.f444w_col][k])
                    for k in range(len(cat))
                    if np.isfinite(float(cat[config.f444w_col][k]))
                }
                print(f"Mode B aperture correction: using '{config.f444w_col}' ({len(f444w_totals)} sources with finite flux)")
            else:
                print(f"WARNING: f444w_col '{config.f444w_col}' not found in catalog; falling back to Mode A")

        for ifilt in range(1, len(images)):
            weights_i = weights[ifilt] if weights is not None else None

            kernel = kernels[ifilt]
            if kernel is None:
                kernel = np.array([[1.0]])  # @@@ this shouldnt be necessary?
            elif isinstance(kernel, PSFRegionMap):
                print(f"Using kernel lookup table {kernel.name}")

            k = bin_factor_from_wcs(wcs[0], wcs[ifilt]) if wcs is not None else 1

            if k > 1:
                if config.multi_resolution_method == "upsample":
                    print(f"upsampling image {ifilt} by factor {k}")
                    images[ifilt] = block_replicate(images[ifilt], k, conserve_sum=True).astype(
                        np.float32
                    )
                    if weights_i is not None:
                        weights_i = block_replicate(weights[ifilt], k).astype(np.float32) * k**2
                    wcs[ifilt] = wcs[0]
                else:
                    print(f"Downsampling templates and kernels by factor {k}")
                    tmpls_lo = Templates()
                    tmpls_lo.original_shape = images[ifilt].shape
                    tmpls_lo.wcs = wcs[ifilt]
                    tmpls_lo._templates = [
                        t.downsample(k, wcs_lo=wcs[ifilt]) for t in self.tmpls._templates
                    ]

                    if isinstance(kernel, PSFRegionMap):
                        kernel.psfs = np.array([downsample_psf(psf, k) for psf in kernel.psfs])
                    else:
                        kernel = downsample_psf(kernel, k)

            if k == 1 or config.multi_resolution_method == "upsample":
                tmpls_lo = deepcopy(self.tmpls)

            if weights_i is not None:
                tmpls_lo.prune_outside_weight(weights_i)

            templates = tmpls_lo.convolve_templates(kernel, inplace=False)
            print(f"Pipeline (convolved) memory: {memory():.1f} GB")

            for t in templates:
                assert np.all(np.isfinite(t.data)), "Templates contain NaN values"

            # @@@ split scenes here
            scenes, labels = generate_scenes(
                templates,
                images[ifilt],
                weights_i,
                coupling_thresh=float(config.scene_coupling_thresh),
                snr_thresh_astrom=float(config.snr_thresh_astrom),
                minimum_bright=int(config.scene_minimum_bright),
                max_merge_radius=float(getattr(config, "scene_max_merge_radius", np.inf)),
            )
            # Assume each scene has .ra and .dec attributes (center coordinates)
            # Compute RA/Dec for each scene center using WCS
            if config.generate_scene_catalog:
                self.all_scenes.append(scenes)
                ras, decs = [], []
                for s in scenes:
                    xy_mean = np.mean([t.position_original for t in s.templates], axis=0)
                    if wcs[0] is not None:
                        ra, dec = wcs[0].wcs_pix2world([xy_mean], 0)[0]
                    else:
                        ra, dec = np.nan, np.nan
                    ras.append(ra)
                    decs.append(dec)

                scene_table = Table(
                    {
                        "id": [s.id for s in scenes],
                        "n_templates": [len(s.templates) for s in scenes],
                        "is_bright": [s.is_bright.sum() for s in scenes],
                        "ra": ras,
                        "dec": decs,
                    }
                )
                scene_table.write(
                    f"scene_catalog_{ifilt}.ecsv", format="ascii.ecsv", overwrite=True
                )
                print(f"Wrote scene catalog scene_catalog_{ifilt}.ecsv")
                import sys

                sys.exit()

            for s in scenes:
                logger.info(f"Scene {s.id}: {len(s.templates)} (bright: {s.is_bright.sum()})")

            niter_scene = max(config.fit_astrometry_niter, 1)
            for j in range(niter_scene):
                logger.info(f"[Scenes] Running iteration {j+1} of {niter_scene}")
                for scn in scenes:
                    scn.set_band(images[ifilt], weights_i, config=config)
                    scn.solve(config=config, apply_shifts=True)

            # build model in res first, then subtract from image
            res = np.zeros_like(images[ifilt])
            for s in scenes:
                sl = _slices_from_bbox(s.bbox)
                res[sl] += s.model_image()  # adds models in place
            # then subtract from image to get residual
            res = images[ifilt] - res

            fluxes = [t.flux for t in templates]
            errs = [t.err for t in templates]
            err_pred = Templates.predicted_errors(templates, weights_i)

            if config.aperture_diam is not None:
                pscale = self._pixel_scale_arcsec(
                    self.wcs[ifilt] if self.wcs is not None else None
                )
                r_img_pix = self._resolve_image_ap_radius_pix(ifilt, config)
                r_img_arcsec = r_img_pix * pscale
                cat["aper_" + str(ifilt)] = 2 * r_img_arcsec
            self._update_catalog_with_fluxes(cat, templates, fluxes, errs, err_pred, ifilt)
            self._add_aperture_photometry(
                cat,
                templates,
                fluxes,
                res,
                ifilt,
                orig_templates=self.tmpls._templates,
                f444w_totals=f444w_totals,
            )

            self.residuals.append(res)
            #            self.fit.append(fitter)
            self.all_templates.append(templates)
            self.all_scenes.append(scenes)
        #            self.infos.append(info)

        print(f"Pipeline (end) memory: {psutil.Process(os.getpid()).memory_info().rss/1e9:.1f} GB")
        self.table = cat

        return self.table, self.residuals  # , self.all_templates, self.all_scenes

    def plot_result(
        self,
        ifilt: int = 1,
        scene_id: int | None = None,
        source_id: int | None = None,
        display_sig: float = 3.0,
    ) -> tuple["matplotlib.figure.Figure", np.ndarray]:
        """Plot the fitted image, model, residual, and color composite.

        The high-resolution template image (``images[0]``) is shown with scene
        overlays alongside the segmentation map, the selected low-resolution
        image, its model, and the residual. A Lupton RGB image combining the
        template and low-resolution images is also displayed.

        Args:
            ifilt: Index of the low-resolution image to display. Defaults to ``1``.
            scene_id: Optional scene identifier to zoom into. Defaults to ``None``.
            source_id: Optional source identifier to zoom into. Defaults to
                ``None``. Ignored if ``scene_id`` is provided.

        Returns:
            Tuple containing the created figure and the array of axes.
        """

        import math

        import matplotlib.pyplot as plt
        import numpy as np
        from copy import deepcopy
        from astropy.visualization import make_lupton_rgb
        from photutils.segmentation import SegmentationImage
        from astropy.table import Table

        if ifilt <= 0 or ifilt >= len(self.images):
            raise ValueError("idx must be between 1 and len(images)-1")

        nscenes = len(np.unique(self.fit[ifilt - 1].scene_ids))

        segmap = self.segmap
        segm = SegmentationImage(segmap)
        segmap_cmap = segm.cmap
        scene_cmap = deepcopy(segmap_cmap)
        scene_cmap.colors[0] = (1.0, 1.0, 1.0, 0.0)

        fitter = self.fit[ifilt - 1]

        if not hasattr(self, "scenes"):
            logger.info("Building scene map for diagnostics")
            scenes = np.zeros_like(segmap, dtype=int)
            # fitter.scene_ids
            for tmpl in fitter.templates:
                iseg = segm.get_index(tmpl.id)
                sl = segm.segments[iseg].slices
                scenes_slice = scenes[sl]
                scenes_slice[segm.data[sl] == tmpl.id] = tmpl.id_scene

        logger.info(f"Plotting image {ifilt} with {nscenes} scenes")

        mask: np.ndarray | None = None
        if scene_id is not None:
            mask = scenes == scene_id
        elif source_id is not None:
            mask = segmap == source_id

        buf = 10
        if mask is not None and np.any(mask):
            ys, xs = np.where(mask)
            y0, y1 = max(ys.min() - buf, 0), min(ys.max() + buf, segmap.shape[0]) + 1
            x0, x1 = max(xs.min() - buf, 0), min(xs.max() + buf, segmap.shape[1]) + 1
        else:
            y0, x0 = 0, 0
            y1, x1 = segmap.shape

        sl_hi = (slice(y0, y1), slice(x0, x1))
        kbin = bin_factor_from_wcs(self.wcs[0], self.wcs[ifilt])
        y0_lo, y1_lo, x0_lo, x1_lo = np.round(bin_remap([y0, y1, x0, x1], kbin)).astype(int)
        sl_lo = (slice(y0_lo, y1_lo), slice(x0_lo, x1_lo))

        img_hi = self.images[0]
        img_lo = self.images[ifilt]

        img_cut = img_lo[sl_lo]
        model_cut = fitter.model_image()[sl_lo]

        tmpl_cut = img_hi[sl_hi]
        seg_cut = segmap[sl_hi]
        scenes_cut = scenes[sl_hi]
        # @@@ for now assume upsampled residual image
        res_cut = self.residuals[ifilt - 1][sl_hi]

        # RGB composite using template as blue and low-res as red
        tmpl_cut_lo = block_reduce(tmpl_cut, kbin, func=np.mean)
        b = tmpl_cut_lo / np.nanstd(tmpl_cut_lo) if np.nanstd(tmpl_cut_lo) != 0 else tmpl_cut_lo
        r = img_cut / np.nanstd(img_cut) if np.nanstd(img_cut) != 0 else img_cut
        g = (r + b) / 2.0
        col_cut = make_lupton_rgb(r, g, b, stretch=display_sig / 1.5)

        # aspect is w/h
        aspect = img_cut.shape[1] / img_cut.shape[0]

        fig, ax = plt.subplots(3, 2, figsize=(10, 13 / aspect))
        ax = ax.flatten()
        images = [
            tmpl_cut,
            seg_cut,
            img_cut,
            model_cut,
            res_cut,
            col_cut,
        ]
        titles = [
            f"template + scenes",
            "segmap",
            f"image{ifilt}",
            f"model image{ifilt}",
            "residual",
            "color",
        ]

        for i, (im, title) in enumerate(zip(images, titles)):
            if title == "segmap":
                ax[i].imshow(im, origin="lower", cmap=segmap_cmap, interpolation="nearest")
                # if plotting a scene, overplot template id as text
                if scene_id is not None or source_id is not None:
                    for tmpl in fitter.templates:
                        if tmpl.id_scene == scene_id:
                            x, y = tmpl.position_original - np.array([x0, y0])
                            ax[i].text(
                                x,
                                y,
                                str(tmpl.id),
                                color="white",
                                fontsize=6,
                                ha="center",
                                va="center",
                            )
            elif title == "color":
                ax[i].imshow(im, origin="lower", interpolation="nearest")
            else:
                ivalid = img_cut != 0
                v = (
                    display_sig * np.nanstd(img_cut[ivalid])
                    if np.any(np.isfinite(img_cut[ivalid]))
                    else 1.0
                )
                ax[i].imshow(im, origin="lower", cmap="gray", vmin=-v, vmax=v)
                if i == 0:
                    # set background of segmap to transparent
                    ax[i].imshow(
                        scenes_cut,
                        origin="lower",
                        cmap=scene_cmap,
                        alpha=0.5,
                        interpolation="nearest",
                    )
            ax[i].set_title(title)

        plt.tight_layout()
        return fig, ax


def run(
    images: Sequence[np.ndarray],
    segmap: np.ndarray,
    *,
    catalog: Table | None = None,
    psfs: Sequence[np.ndarray] | None = None,
    weights: Sequence[np.ndarray] | None = None,
    wht_images: Sequence[np.ndarray] | None = None,
    kernels: Sequence[np.ndarray | PSFRegionMap] | None = None,
    wcs: Sequence[WCS] | None = None,
    window: Window | None = None,
    extend_templates: str | None = None,
    config: FitConfig | None = None,
) -> tuple[Table, list[np.ndarray]]:
    """Backward compatible wrapper for :class:`Pipeline`"""

    pipeline = Pipeline(
        images,
        segmap,
        catalog=catalog,
        psfs=psfs,
        weights=weights,
        wht_images=wht_images,
        kernels=kernels,
        wcs=wcs,
        window=window,
        extend_templates=extend_templates,
        config=config,
    )
    return pipeline.run()
