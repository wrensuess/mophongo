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
from photutils.segmentation import SegmentationImage
from astropy.wcs.utils import proj_plane_pixel_scales

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


def _per_source_chi2(
    residual: np.ndarray, weights: np.ndarray, templates: Sequence[Template]
) -> np.ndarray:
    """Compute template-weighted chi² for each template.

    For each template, computes the sum of squared, template-weighted residuals
    divided by the noise variance, normalized by the sum of template weights.

    Returns
    -------
    ndarray
        Array of template-weighted chi² values, one per template in ``templates``.
    """
    chi2 = np.zeros(len(templates), dtype=float)
    for i, tmpl in enumerate(templates):
        res = residual[tmpl.slices_original]
        tmpl_data = tmpl.data[tmpl.slices_cutout]
        ivar = weights[tmpl.slices_original]  # inverse variance
        mask = ivar > 0
        # Template-weighted chi²: sum((res * tmpl)^2 / var) / sum(tmpl^2)
        num = np.sum(mask * (res * tmpl_data) ** 2 * ivar)
        denom = np.sum(mask * tmpl_data**2)
        chi2[i] = num / denom if denom > 0 else 0.0
    return chi2


# should support PSFRegionMap as well, like in template.convolve_templates
#   ra, dec = tmpl.wcs.wcs_pix2world(x, y, 0)
# else:
#     ra, dec = x, y
# kern = kernel.get_psf(ra, dec)


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
    def _add_templates_for_bad_fits(
        self,
        templates: list[Template],
        tmpls_lo: Templates,
        psf: np.ndarray | PSFRegionMap | None,
        weights: np.ndarray | None,
        fitter: "SparseFitter",
        image: np.ndarray,
        fitter_cls,
        config: _FitConfig,
    ) -> tuple[list[Template], "SparseFitter"]:
        """Add secondary templates for poorly fitted sources.

        Parameters
        ----------
        templates
            Current list of templates used in the fit.
        tmpls_lo
            Base templates prior to convolution. Used when adding new
            components.
        psf
            PSF image for the current low-resolution frame.
        weights
            Weight map corresponding to ``image``.
        fitter
            Fitter instance from the initial solve.
        image
            Image data being modelled.
        fitter_cls
            Fitter class used to instantiate a new fitter if additional
            templates are required.
        config
            Fit configuration options.

        Returns
        -------
        list[Template], SparseFitter
            Possibly extended template list and a corresponding fitter
            instance.
        """

        if not (
            (config.multi_tmpl_psf_core or config.multi_tmpl_colour)
            and psf is not None
            and weights is not None
        ):
            fitter._ata = None
            return templates, fitter

        res = fitter.residual()
        chi_nu = _per_source_chi2(res, weights, templates)
        bad_idx = np.where(chi_nu > config.multi_tmpl_chi2_thresh)[0]
        if bad_idx.size > 0:
            logger.info("Adding %d new templates for poor fits", bad_idx.size)
            for bi in bad_idx:
                parent = templates[bi]
                if config.multi_tmpl_psf_core:
                    stamp = _extract_psf_at(parent, psf)
                    add_tmpl = tmpls_lo.add_component(parent, stamp, "psf")
                    templates.append(add_tmpl)
            fitter = fitter_cls(templates, image, weights, config)
        else:
            fitter._ata = None
        return templates, fitter

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

    @staticmethod
    def _tcor_blend_weight(snr_seg: float, center: float, width: float) -> float:
        """Smooth logistic weight for the low-SNR tcor_H denominator blend.

        w -> 1 at high ``snr_seg`` (trust the direct Rphi measurement ``ap_f_data``),
        w -> 0 at low ``snr_seg`` (trust the small-aperture + template-growth estimate).
        NaN ``snr_seg`` (e.g. non-extended templates) returns 1.0 so those sources keep
        the current direct path. ``center``/``width`` are in ``snr_seg`` units (width is
        the logistic scale = ``width*center``).
        """
        if not np.isfinite(snr_seg):
            return 1.0
        scale = max(float(width) * float(center), 1e-6)
        # clip the logistic argument so np.exp cannot overflow for extreme SNR/params
        z = np.clip((float(snr_seg) - float(center)) / scale, -700.0, 700.0)
        return float(1.0 / (1.0 + np.exp(-z)))

    def _build_f444w_residual(self, orig_templates: list[Template]) -> np.ndarray:
        """F444W neighbour-subtracted residual map: images[0] - Σ_j model_j.

        Each high-res template (unit-sum) is scaled by ``template_norm`` (its
        detection-band flux) and subtracted from the F444W image. Built once and
        reused across bands; for each source we add its own model back before the
        aperture sum, so the tcor_H denominator is the real neighbour-subtracted
        F444W aperture flux rather than the (noise-level) template sum. The map is
        written to disk for over-subtraction diagnosis.
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
        ``template_norm`` are defined against that unit-sum template; PSF curve-
        of-growth fractions (for ``apcor_from_psf`` sources) enter ONLY the
        correction factors ``apcor1``/``totcor1``.

        Templates are unit-sum normalised; ``orig_t.template_norm`` holds the
        pre-normalisation detection-band sum, converting aperture fractions to
        real flux units. All quantities below in real (image) flux units:

        ap_b_corr = template_norm * apB_corr         — low-res aperture flux, correction side
        ap_f_corr = template_norm * apF_corr         — high-res aperture flux, correction side
        apcor1   = ap_f_corr / ap_b_corr             — shape correction (low-res→high-res)
        ap_f_data = aper(F444W_residual + model_i, r_phi)   — REAL neighbour-subtracted
                                                    F444W aperture flux (= template_norm*apF_book
                                                    + residual-in-aper)
        tcor_H   = ftot / ap_f_data   (if a catalog total is supplied, else 1.0)
                                                    — correction to the catalog total
          The tcor_H denominator is the neighbour-subtracted F444W aperture flux
          measured on data (not the noise-level template sum). The F444W residual
          map (images[0] − Σ models) is built once and saved to
          ``f444w_template_residual.fits`` for over-subtraction diagnosis; the
          per-source ``ap_f_data`` is written to the ``apf_data_{idx}`` column.

        ap_model     = fl * aper(H*K, r_phi)         (model flux in aperture, low-res)
        res_sum      = Σ_aperture(residual)          (residual within the aperture disk)
        res_seg      = Σ_segmap(residual)            (diagnostic only)
        ap_flux      = ap_model + res_sum            (observed aperture flux)
        ap_flux_corr = ap_model * apcor1 * tcor_H + res_sum   (Estimator 3 total)

        apcor1 and tcor_H are kept as separate factors and columns and are never
        algebraically collapsed. Per parent id, ap_model is accumulated over any
        multi-component templates; the corrections and residual are computed
        once.

        Writes ap_model_{idx}, apcor1_{idx}, tcor_{idx}, apcor_{idx} (=product),
        res_sum_{idx}, res_seg_{idx}, ap_flux_{idx}, ap_flux_corr_{idx}.
        """
        cfg = self.config
        id_to_row = {int(i): k for k, i in enumerate(cat["id"])}
        r_img_pix = self._resolve_image_ap_radius_pix(idx, cfg)

        if r_orig_pix is None:
            pscale_img = self._pixel_scale_arcsec(self.wcs[idx] if self.wcs is not None else None)
            pscale_ref = self._pixel_scale_arcsec(self.wcs[0] if self.wcs is not None else None)
            r_orig_pix = r_img_pix * pscale_img / pscale_ref if (pscale_img and pscale_ref) else r_img_pix

        # Catalog-anchored low-SNR tcor_H denominator (Weaver+ super catalog). When
        # f444w_totcor_col + f444w_aper_col are present, the faint F444W aperture flux
        # is predicted from the catalog color-aperture flux (f_f444w/tot_cor) grown to
        # the band aperture by the source's own curve of growth -- noise-free (Rung 1).
        # r_color = use_aper/2 on the F444W (reference) grid, same units as r_orig_pix.
        nircam_totcor_by_id: dict[int, float] = {}
        r_color_pix_by_id: dict[int, float] = {}
        cat_src = getattr(self, "catalog", None)
        pscale_ref = self._pixel_scale_arcsec(self.wcs[0] if self.wcs is not None else None)
        if (cfg.tcor_lowsnr_psf and cat_src is not None and pscale_ref
                and cfg.f444w_totcor_col and cfg.f444w_aper_col
                and cfg.f444w_totcor_col in cat_src.colnames
                and cfg.f444w_aper_col in cat_src.colnames):
            ids = np.asarray(cat_src["id"]).astype(int)
            tcs = np.asarray(cat_src[cfg.f444w_totcor_col], dtype=float)
            uas = np.asarray(cat_src[cfg.f444w_aper_col], dtype=float)
            for sid, tc, ua in zip(ids, tcs, uas):
                if np.isfinite(tc) and tc > 0 and np.isfinite(ua) and ua > 0:
                    nircam_totcor_by_id[int(sid)] = float(tc)
                    r_color_pix_by_id[int(sid)] = 0.5 * float(ua) / pscale_ref
            print(f"  Low-SNR tcor_H: catalog-anchored denominator from "
                  f"'{cfg.f444w_totcor_col}'/'{cfg.f444w_aper_col}' ({len(nircam_totcor_by_id)} sources)")

        for name in (
            f"ap_model_{idx}",
            f"apcor_{idx}",
            f"apcor1_{idx}",
            f"totcor1_{idx}",
            f"tcor_{idx}",
            f"apf_data_{idx}",
            f"aper_rphi_{idx}",
            f"tcor_w_{idx}",
            f"aper_pred_{idx}",
            f"res_sum_{idx}",
            f"res_seg_{idx}",
            f"ap_flux_{idx}",
            f"ap_flux_est2_{idx}",
            f"ap_flux_corr_{idx}",
        ):
            if name not in cat.colnames:
                cat[name] = cfg.bad_value

        orig_by_id = {t.id: t for t in orig_templates} if orig_templates else {}
        use_tcor = f444w_totals is not None

        # F444W neighbour-subtracted residual map (built once, reused across
        # bands): the tcor_H denominator becomes the REAL neighbour-subtracted
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
            f"{'with tcor_H' if use_tcor else 'apcor1 only'})"
        )

        # Phase A: point-source aperture-to-total from the PSF curve of growth for
        # ``apcor_from_psf`` (faint / bright+compact) templates, whose shape is
        # unmeasurable so the footprint-truncated template under-counts the total.
        # apF = EE(PSF_hires, r_orig); apB = EE(PSF_band, r_img) -- the matching
        # kernel makes PSF_hires⊗K = PSF_band, so the band PSF gives the convolved
        # curve of growth directly. EE cached per PSF-region id (few distinct).
        psf_hires = self.psfs[0] if (self.psfs is not None and len(self.psfs) > 0) else None
        psf_band = self.psfs[idx] if (self.psfs is not None and len(self.psfs) > idx) else None
        _ee_cache: dict = {}

        # Aperture radius for the BAND PSF EE (apB), in the band PSF's NATIVE pixel
        # scale. psf_hires is on the reference grid so apF uses r_orig_pix directly,
        # but psf_band is on its native grid; in upsample mode the fit grid (r_img_pix)
        # is finer than the band PSF, so measuring apB at r_img_pix would sample the
        # wrong physical radius (-> totcor1 collapses to ~1). Convert via native scales.
        r_band_pix = r_img_pix
        _psn = getattr(self, "_native_pscale", None)
        if _psn and len(_psn) > idx and _psn[0] and _psn[idx]:
            r_band_pix = r_orig_pix * float(_psn[0]) / float(_psn[idx])

        def _psf_ee(psfmap, ra, dec, radius):
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
            key = (id(psfmap), region, float(radius))
            if key not in _ee_cache:
                try:
                    # True-total normalization (docs/aperture_corrections.md
                    # Sec 4.1/5.2): psf_ee_at_radius is stamp-normalized, so
                    # multiply by the region's containment (fraction of the
                    # PSF's true total flux in the stamp). ndarray PSFs -> 1.0.
                    _ee_cache[key] = utils.psf_ee_at_radius(psf, radius) * containment
                except Exception:  # pragma: no cover - degenerate PSF
                    _ee_cache[key] = None
            return _ee_cache[key]

        # Accumulate the model aperture flux per parent id (multi-component
        # templates share an id); the correction factors and the residual are
        # computed once per parent. apcor1 and tcor_H are kept as SEPARATE
        # factors (and columns) and never algebraically collapsed.
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

            # Correction-side convolved aperture fraction. For point-source-like
            # (apcor_from_psf) templates use the band PSF curve of growth so the
            # fraction is over the TRUE total, not the truncated template footprint;
            # this feeds apcor1/totcor1 ONLY, never the bookkeeping above.
            use_psf = bool(getattr(orig_t, "apcor_from_psf", False))
            ra_dec = None
            # Template-path default (apcor_from_psf False, or PSF unavailable below):
            # apB_corr stays the footprint-truncated template fraction, with no
            # stamp-edge extrapolation. Scope cut vs docs/aperture_corrections.md
            # Sec 5.2 bullet 2 -- deferred to Stage 4 (≲1.5% effect per Sec 4.1).
            apB_corr = apB_book
            if use_psf:
                # Cutout-frame position with the cutout-adjusted WCS (the CRPIX is
                # shifted to the cutout in Template.__init__; see the downsample
                # convention in templates.py). position_original is the full-image
                # frame and would give a wrong sky position -> wrong PSF region.
                pos = orig_t.input_position_cutout
                if getattr(orig_t, "wcs", None) is not None:
                    ra_dec = orig_t.wcs.wcs_pix2world(pos[0], pos[1], 0)
                else:
                    ra_dec = pos
                # Both fractions must come from the PSF, or neither: a mixed
                # template/PSF apcor1 would not be a clean curve-of-growth ratio.
                ee_b = _psf_ee(psf_band, ra_dec[0], ra_dec[1], r_band_pix)
                ee_f = _psf_ee(psf_hires, ra_dec[0], ra_dec[1], r_orig_pix)
                if (ee_b is not None and ee_b > 0) and (ee_f is not None and ee_f > 0):
                    apB_corr = float(ee_b)
                else:
                    use_psf = False  # PSF unavailable -> template path for this source

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
            # Real-unit template aperture fluxes (template_norm restores image
            # flux units; it cancels in apcor1 but is required so tcor_H is
            # dimensionless). Correction side (apB_corr): PSF EE for
            # apcor_from_psf sources, else the template -- feeds apcor1/totcor1.
            ap_b_corr = template_norm_i * apB_corr  # low-res convolved template flux in aperture

            # High-res aperture fraction, correction side: PSF curve of growth for
            # apcor_from_psf sources (same true-total normalisation as apB_corr
            # above), else the template. Both from the same source so apcor1 is a
            # clean bounded ratio. Feeds apcor1 and the low-SNR blend prediction only.
            apF_corr = None
            if use_psf and ra_dec is not None:
                ee_f = _psf_ee(psf_hires, ra_dec[0], ra_dec[1], r_orig_pix)
                if ee_f is not None and ee_f > 0:
                    apF_corr = float(ee_f)
            if apF_corr is None:
                # Template-path fallback: no stamp-edge extrapolation (same scope
                # cut as apB_corr above; deferred to Stage 4).
                apF_corr = self._aperture_sum_on_template(orig_t, r_orig_pix)
            ap_f_corr = template_norm_i * apF_corr if apF_corr > 0 else 0.0  # high-res template flux in aperture

            # Shape correction: high-res / low-res aperture flux (real units).
            # Uses the template shapes (a clean, bounded ratio ~PSF curve of growth).
            apcor1 = ap_f_corr / ap_b_corr if (ap_b_corr > 0 and ap_f_corr > 0) else 1.0
            # Internal aperture-to-total (design-doc Eq. 7; = IDL totcor). Bounded
            # ~1.2 for apcor_from_psf sources now that apB is over the true total.
            totcor1 = 1.0 / apB_corr if apB_corr > 0 else 1.0

            # tcor_H denominator: the REAL neighbour-subtracted F444W aperture flux
            # (template model_i + residual), measured on data rather than the
            # noise-level template sum. Bookkeeping side -- ALWAYS the fitted
            # original template's own fraction, so aperture-sum linearity holds
            # against the F444W residual map (built from the same templates).
            apF_book = self._aperture_sum_on_template(orig_t, r_orig_pix)
            ap_f_book = template_norm_i * apF_book if apF_book > 0 else 0.0
            ap_f_data = ap_f_book
            if f444w_res is not None:
                ap_f_data = ap_f_book + self._aperture_sum_on_map(f444w_res, orig_t, r_orig_pix)

            # Low-SNR blend of the tcor_H DENOMINATOR (design-doc Eq. 8). The measured
            # neighbour-subtracted F444W aperture flux ap_f_data is noise-dominated for
            # faint sources (~707 px of sky); blend it toward a NOISE-FREE catalog-anchored
            # prediction. Rung 1 (both catalog columns): Fap_pred = color_flux *
            # apF_frac(Rphi)/apF_frac(r_color), color_flux = f_f444w/tot_cor. Rung 2 (only
            # a catalog total): Fap_pred = f_f444w * apF_frac(Rphi) => tcor_H -> 1/apF_frac.
            # apF_frac uses the SAME source profile (PSF for apcor_from_psf, else template).
            # w -> 1 (high snr_seg) keeps ap_f_data; w -> 0 uses the prediction.
            ftot = f444w_totals.get(int(tmpl.id)) if use_tcor else None
            has_ftot = ftot is not None and np.isfinite(ftot)

            aper_rphi = ap_f_data
            tcor_w = 1.0
            aper_pred = float(cfg.bad_value)
            if cfg.tcor_lowsnr_psf and has_ftot:
                sid = int(tmpl.id)
                pred = None
                if sid in nircam_totcor_by_id:
                    r_color = r_color_pix_by_id[sid]
                    if use_psf and ra_dec is not None:
                        apF_frac_color = _psf_ee(psf_hires, ra_dec[0], ra_dec[1], r_color)
                    else:
                        apF_frac_color = self._aperture_sum_on_template(orig_t, r_color)
                    if apF_frac_color and apF_frac_color > 0:
                        color_flux = float(ftot) / nircam_totcor_by_id[sid]
                        pred = color_flux * (apF_corr / apF_frac_color)
                else:
                    # Rung 2: assume template total = catalog total.
                    pred = float(ftot) * apF_corr
                if pred is not None and np.isfinite(pred):
                    aper_pred = float(pred)
                    tcor_w = self._tcor_blend_weight(
                        getattr(orig_t, "snr_seg", float("nan")),
                        cfg.tcor_blend_center * cfg.fit_snrlo_psf,
                        cfg.tcor_blend_width,
                    )
                    # Clamp the measured term at 0: a negative ap_f_data is unphysical
                    # over-subtraction, and blending it with the positive prediction can
                    # cross zero at intermediate w -> tcor_H spike (a residual band tail).
                    # Both terms are then >= 0 (aper_pred > 0), so aper_rphi cannot cross
                    # zero. The raw (unclamped) value stays in the apf_data_{idx} diagnostic.
                    aper_rphi = tcor_w * max(ap_f_data, 0.0) + (1.0 - tcor_w) * aper_pred

            # tcor_H = f_f444w / aper_rphi (blended). Stays 1 without a catalog total.
            tcor_H = 1.0
            if has_ftot:
                tcor_H = float(np.float64(ftot) / np.float64(aper_rphi))

            # Residual within the measurement aperture (disk), added UNSCALED.
            # Same aperture geometry as _aperture_sum_on_template applied to the
            # residual patch, so ap_flux = ap_model + res_sum exactly.
            # TODO(blend residual double-count): for two close sources the aperture
            # disks overlap, so background residual in the shared region is added to
            # both ap_flux_corr values. Full fix would partition the shared residual.
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

            per[row] = dict(apcor1=apcor1, totcor1=totcor1, tcor=tcor_H, apf_data=ap_f_data,
                            aper_rphi=aper_rphi, tcor_w=tcor_w, aper_pred=aper_pred,
                            res_sum=res_sum, res_seg=res_seg)

        # Write per-parent Estimator-3 results. The correction is applied as the
        # explicit product apcor1 * tcor_H (never pre-collapsed).
        for row, d in per.items():
            ap_model = model_acc[row]
            apcor1 = d["apcor1"]
            totcor1 = d["totcor1"]
            tcor_H = d["tcor"]
            res_sum = d["res_sum"]
            cat[f"ap_model_{idx}"][row] = ap_model
            cat[f"apcor1_{idx}"][row] = apcor1
            cat[f"totcor1_{idx}"][row] = totcor1
            cat[f"tcor_{idx}"][row] = tcor_H
            cat[f"apf_data_{idx}"][row] = d["apf_data"]
            cat[f"aper_rphi_{idx}"][row] = d["aper_rphi"]
            cat[f"tcor_w_{idx}"][row] = d["tcor_w"]
            cat[f"aper_pred_{idx}"][row] = d["aper_pred"]
            cat[f"apcor_{idx}"][row] = apcor1 * tcor_H
            cat[f"res_sum_{idx}"][row] = res_sum
            cat[f"res_seg_{idx}"][row] = d["res_seg"]
            cat[f"ap_flux_{idx}"][row] = ap_model + res_sum
            # Estimator 2 (IDL-consistent): model aperture flux scaled to total by
            # the internal template curve of growth (totcor1 = 1/apB), + residual.
            cat[f"ap_flux_est2_{idx}"][row] = ap_model * totcor1 + res_sum
            # Estimator 3: model aperture flux scaled to total by the factored
            # correction, plus the unscaled residual over the aperture disk.
            cat[f"ap_flux_corr_{idx}"][row] = ap_model * apcor1 * tcor_H + res_sum

    def run(self, config: FitConfig | None = None) -> tuple[Table, list[np.ndarray]]:
        """Run photometry on the configured images.

        Returns
        -------
        Table
            Catalog containing flux measurements for each image.
        list of ndarray
            Residual images corresponding to each fitted image.
        SparseFitter
            The fitter instance used for the final fit.
        """
        from .fit import SparseFitter
        from .astro_fit import GlobalAstroFitter
        from .astrometry import AstroCorrect
        import warnings

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

        self.tmpls = Templates(min_size=min_size)
        self.tmpls.extract_templates(
            images[0],
            segmap,
            list(zip(cat["x"], cat["y"])),
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

        astro = AstroCorrect(config)
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

            scenes = None  # initialise; set below if run_scene_solver=True
            # @@@ split scenes here
            # Optional scene-based solver: does not alter legacy path
            if getattr(config, "run_scene_solver", False):
                # Work on a copy of templates to avoid affecting legacy loop
                templates_scene = templates
                scenes, labels = generate_scenes(
                    templates_scene,
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

            else:
                print("Running legacy solver")
                # fitter_cls = (
                #     GlobalAstroFitter
                #     if (config.fit_astrometry_niter > 0 and config.fit_astrometry_joint)
                #     else SparseFitter
                # )
                fitter_cls = SparseFitter
                niter = max(config.fit_astrometry_niter, 1)
                for j in range(niter):
                    print(f"Running iteration {j+1} of {niter}")

                    fitter = fitter_cls(templates, images[ifilt], weights_i, config)
                    fluxes, errs, info = fitter.solve()
                    print(f"Pipeline (residual) memory: {memory():.1f} GB")

                    # if config.fit_astrometry_niter > 0 and not config.fit_astrometry_joint:
                    #     # @@@ this is very expensive. We dont need to form the whole residual image
                    #     # can do it on the stamps only
                    #     res = fitter.residual()
                    #     logger.info("fitting astrometry separately")
                    #     astro.fit(templates, res, fitter.solution)

                    if config.fit_astrometry_niter > 0 and config.fit_astrometry_joint:
                        Templates.apply_template_shifts(templates)

                res = fitter.residual()

                #            print("END of TEMPLATES FITTING")

                # one final flux only solve after astrometry
                # cfg_noshift = _FitConfig(**config.__dict__)
                # cfg_noshift.fit_astrometry_niter = 0
                # templates, fitter = self._add_templates_for_bad_fits(
                #     templates,
                #     tmpls_lo,
                #     psfs[ifilt] if psfs is not None else None,
                #     weights_i,
                #     fitter,
                #     images[ifilt],
                #     fitter_cls,
                #     config,
                # )

                # add soft non-negative priors if fluxes are < 0.0 and resolve.
                # note idx is relative to initial list of templates. But additional templates were added at the end, so idx still works

                # snr = np.divide(fluxes, errs, out=np.zeros_like(errs), where=errs > 0)
                # selneg = snr < config.negative_snr_thresh
                # if np.any(selneg):
                #     logger.info(
                #         f"{selneg.sum()} fluxes are negative, applying soft non-negative prior and resolving."
                #     )
                #     # this updates ata and atb, so we can resolve again
                #     scale = np.clip(-snr, 1.0, 5.0)  # more negative → tighter prior
                #     fitter.add_flux_priors(selneg, mu=0.0, sigma=(errs / scale))

                #            fluxes, errs, info = fitter.solve(config=cfg_noshift)

            fluxes = [t.flux for t in templates]
            errs = [t.err for t in templates]
            err_pred = Templates.predicted_errors(templates, weights_i)

            # calculate a full image residual from the scenes and their slice
            #            res_scene
            # if getattr(config, "run_scene_solver", False):
            #     # sanity check
            #     diff = np.abs(res - res_scene)
            #     maxdiff = np.nanmax(diff)
            #     if maxdiff > 1e-5 * np.nanmax(np.abs(res)):
            #         warnings.warn(f"Scene residual differs from full residual: max diff {maxdiff}")
            #     else:
            #         print(f"Scene residual matches full residual: max diff {maxdiff}")
            # #                res = res_scene
            # print("Done...")

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
) -> tuple[Table, list[np.ndarray], SparseFitter]:
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

    # # EXTREMELY SLOW
    # # block into tiles for faster access
    # store = zarr.storage.MemoryStore()
    # group = zarr.group(store=store)  # container
    # fast = Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE)  # fastest
    # tight = Blosc(cname="zstd", clevel=1, shuffle=Blosc.BITSHUFFLE)  # better ratio, still fast
    # # You can control threads with Blosc(nthreads=<N>) if desired.
    # for i in range(len(images)):
    #     if images[i] is not None:
    #         img = group.create_array(
    #             f"images/{i}",
    #             shape=(images[i].shape),
    #             chunks=(512, 512),
    #             dtype="float32",
    #             compressors=None,  # <- critical
    #             filters=None,  # <- critical
    #             overwrite=True,
    #             fill_value=0.0,
    #         )
    #         img[:] = images[i]
    #         images[i] = img

    #     if weights[i] is not None:
    #         wht = group.create_array(
    #             f"weights/{i}",
    #             shape=(weights[i].shape),
    #             chunks=(512, 512),
    #             dtype="float32",
    #             compressors=None,  # <- critical
    #             filters=None,  # <- critical
    #             overwrite=True,
    #             fill_value=0.0,
    #         )
    #         wht[:] = weights[i]
    #         weights[i] = wht

    # # print(f"Pipeline (blocked storage) memory: {memory():.1f} GB")
