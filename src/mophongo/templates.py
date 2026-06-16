from __future__ import annotations

from typing import Any, Iterable, Iterator, List, Tuple
from copy import deepcopy

import logging
import numpy as np
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from photutils.segmentation import SegmentationImage
from tqdm import tqdm
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates
from astropy.nddata import block_reduce
from astropy.wcs.utils import proj_plane_pixel_scales

from .utils import measure_shape, bin_remap, psf_ee_radius_pix, psf_ee_area_pix
from .psf_map import PSFRegionMap

logger = logging.getLogger(__name__)

import numpy as np
from copy import deepcopy

try:
    from astropy.wcs import WCS, Sip
except Exception:  # if astropy not available / SIP missing
    WCS = None
    Sip = None

__all__ = [
    "AlignedCutout",
    "as_block_reduce",
    "as_block_replicate",
    "scale_wcs_pixel",
]

# ───────────────────────── helpers ─────────────────────────


def _round_half_up(x: float) -> int:
    return int(np.floor(x + 0.5))


def _aligned_bounds_1d(pos: float, size_min: int, align: int) -> tuple[int, int]:
    """
    Return [imin, imax) bounds with:
      • lower bound divisible by `align` (align≥1)
      • length >= size_min
      • length is a multiple of `align`
      • center close to `pos`
    """
    size_min = int(size_min)
    align = int(max(1, align))

    if align == 1:
        imin = int(np.ceil(pos - size_min / 2.0))
        imax = imin + size_min
        return imin, imax

    size_min = max(size_min, align)

    imin0 = int(np.ceil(pos - size_min / 2.0))
    dx_min = imin0 % align
    imin = imin0 - dx_min

    imax_f = pos + max((pos - imin), size_min / 2.0)
    #    print(pos, size_min)
    dx_max = (-imax_f) % align
    imax = int(np.rint(imax_f + dx_max))

    print(imin0, dx_min, imin)
    print(imax_f, dx_max, imax)
    L = imax - imin
    if L <= 0 or (L % align) != 0:
        imax = imin + ((L + align - 1) // align) * align
    return int(imin), int(imax)


def _bbox_from_slices(sl):
    return ((sl[0].start, sl[0].stop - 1), (sl[1].start, sl[1].stop - 1))


def _slices_from_bbox(bbox):
    return (slice(bbox[0], bbox[1] + 1), slice(bbox[2], bbox[3] + 1))


def _block_reduce(arr: np.ndarray, fact: int, func=np.sum) -> np.ndarray:
    """
    Fast 2-D block reduction by integer `fact` (flux-conserving with func=np.sum).
    """
    a = np.asarray(arr, dtype=np.float32, order="C")
    H, W = a.shape
    H2, W2 = (H // fact) * fact, (W // fact) * fact
    if H2 != H or W2 != W:
        a = a[:H2, :W2]
    a = a.reshape(H2 // fact, fact, W2 // fact, fact)
    return func(a, axis=(1, 3), dtype=np.float32)


def _block_replicate(arr: np.ndarray, fact: int, conserve_sum: bool = True) -> np.ndarray:
    """
    Fast 2-D nearest upsampling by integer `fact`. If `conserve_sum` True,
    each pixel is divided by fact**2 (so flux is preserved).
    """
    a = np.asarray(arr, dtype=np.float32, order="C")
    tile = np.ones((fact, fact), dtype=np.float32)
    if conserve_sum:
        tile /= fact * fact
    return np.kron(a, tile)


def scale_wcs_pixel(
    wcs: WCS | None, pixel_scale_factor: float, new_shape: tuple[int, int] | None = None
) -> WCS | None:
    """
    Scale a WCS by a pixel-size factor (>=0), **preserving sky coordinates**.
      pixel_scale_factor > 1  → pixels get larger (downsampling)
      pixel_scale_factor < 1  → pixels get smaller (upsampling)

    cd/cdelt ← cd/cdelt * pixel_scale_factor
    crpix   ← (crpix - 0.5)/pixel_scale_factor + 0.5
    """
    if wcs is None:
        return None
    w2 = deepcopy(wcs)

    f = float(pixel_scale_factor)
    if hasattr(w2.wcs, "cd") and w2.wcs.cd is not None and w2.wcs.cd.size:
        w2.wcs.cd = w2.wcs.cd * f
    else:
        w2.wcs.cdelt = w2.wcs.cdelt * f

    old_crpix = w2.wcs.crpix.copy()
    w2.wcs.crpix = (old_crpix - 0.5) / f + 0.5

    if new_shape is not None:
        try:
            w2.pixel_shape = (int(new_shape[0]), int(new_shape[1]))
        except Exception:
            pass

    if getattr(wcs, "sip", None) is not None and Sip is not None:
        # SIP polynomials evaluated relative to their CRPIX (in pixel units):
        # just shift SIP CRPIX the same way as WCS CRPIX
        off = old_crpix - w2.wcs.crpix
        w2.sip = Sip(wcs.sip.a, wcs.sip.b, wcs.sip.ap, wcs.sip.bp, wcs.sip.crpix - off)

    w2.wcs.set()
    return w2


# ──────────────────────── main class ─────────────────────────


class AlignedCutout:
    """
    Minimal 2-D cutout that:
      • uses *partial* mode only (zero outside the image)
      • `size` is a **minimum**; actual data may be enlarged by `align`
      • lower-left bound is aligned to a multiple of `align` (per axis)
      • shape is a multiple of `align`
      • stores an adjusted WCS (incl. SIP if present)

    Parameters
    ----------
    data : 2D ndarray
    position : (x, y) float — pixel-center coords
    size : (ny, nx) int or scalar
    align : int >= 1
    copy : bool
    fill_value : float
    wcs : astropy.wcs.WCS (optional)
    """

    def __init__(
        self,
        data: np.ndarray,
        position: tuple[float, float],
        size: tuple[int, int] | int,
        *,
        align: int = 1,
        copy: bool = False,
        fill_value: float | int = 0.0,
        wcs: WCS | None = None,
    ):
        arr = np.asarray(data)
        self.align = int(max(1, align))
        self.shape_input = arr.shape  # (ny, nx)

        x, y = float(position[0]), float(position[1])
        if np.isscalar(size):
            ny = nx = int(size)
        else:
            ny, nx = int(size[0]), int(size[1])

        # aligned bounds in ORIGINAL coords
        x0, x1 = _aligned_bounds_1d(x, nx, self.align)
        y0, y1 = _aligned_bounds_1d(y, ny, self.align)
        h = y1 - y0
        w = x1 - x0

        # overlap with source image
        Y0 = max(0, y0)
        X0 = max(0, x0)
        Y1 = min(arr.shape[0], y1)
        X1 = min(arr.shape[1], x1)

        dy = Y0 - y0
        dx = X0 - x0
        yslice_dst = slice(dy, dy + (Y1 - Y0))
        xslice_dst = slice(dx, dx + (X1 - X0))
        yslice_src = slice(Y0, Y1)
        xslice_src = slice(X0, X1)

        fully_inside = (y0 >= 0) and (x0 >= 0) and (y1 <= arr.shape[0]) and (x1 <= arr.shape[1])

        if not fully_inside or copy:
            out = np.zeros((h, w), dtype=arr.dtype)
            if fill_value != 0:
                out[...] = out.dtype.type(fill_value)
            if (Y1 > Y0) and (X1 > X0):
                out[yslice_dst, xslice_dst] = arr[yslice_src, xslice_src]
            self.data = out
        else:
            self.data = arr[y0:y1, x0:x1]

        self.shape = self.data.shape
        self.input_position_original = (x, y)
        self.input_position_cutout = (x - x0, y - y0)

        self.slices_original = (yslice_src, xslice_src)
        self.slices_cutout = (yslice_dst, xslice_dst)

        self.bbox_original = _bbox_from_slices(self.slices_original)
        self.bbox_cutout = _bbox_from_slices(self.slices_cutout)

        self.origin_original = (
            self.slices_original[1].start,
            self.slices_original[0].start,
        )  # (x, y)
        self.origin_cutout = (self.slices_cutout[1].start, self.slices_cutout[0].start)  # (x, y)

        # “true” cutout origin relative to original, including any fill padding
        self._origin_original_true = (
            self.origin_original[0] - self.slices_cutout[1].start,
            self.origin_original[1] - self.slices_cutout[0].start,
        )

        self.position_original = (_round_half_up(x), _round_half_up(y))
        self.position_cutout = (
            _round_half_up(self.input_position_cutout[0]),
            _round_half_up(self.input_position_cutout[1]),
        )

        so, sc = self.slices_original, self.slices_cutout
        self.center_original = (
            0.5 * (so[1].start + so[1].stop - 1),
            0.5 * (so[0].start + so[0].stop - 1),
        )
        self.center_cutout = (
            0.5 * (sc[1].start + sc[1].stop - 1),
            0.5 * (sc[0].start + sc[0].stop - 1),
        )

        # WCS adjusted to the cutout (shift CRPIX, keep SIP consistent)
        if wcs is not None:
            off_xy = np.array(self._origin_original_true, dtype=float)  # (x, y)
            w2 = deepcopy(wcs)
            if getattr(w2, "wcs", None) is not None and getattr(w2.wcs, "crpix", None) is not None:
                w2.wcs.crpix -= off_xy
            try:
                w2.array_shape = self.data.shape
                w2.pixel_shape = self.data.shape
            except Exception:
                pass
            if getattr(wcs, "sip", None) is not None and Sip is not None:
                w2.sip = Sip(wcs.sip.a, wcs.sip.b, wcs.sip.ap, wcs.sip.bp, wcs.sip.crpix - off_xy)
            w2.wcs.set()
            self.wcs = w2
        else:
            self.wcs = None

    # ───────────── array-only helpers (no geometry changes) ─────────────

    def as_block_reduced(self, factor: int, func=np.sum) -> np.ndarray:
        """Return block-reduced self.data by `factor` (trims edges as needed)."""
        if factor < 1 or int(factor) != factor:
            raise ValueError("factor must be a positive integer")
        return _block_reduce(self.data, int(factor), func=func)

    def as_block_replicated(self, factor: int, conserve_sum: bool = True) -> np.ndarray:
        """Return block-replicated self.data by `factor` (nearest upsample)."""
        if factor < 1 or int(factor) != factor:
            raise ValueError("factor must be a positive integer")
        if factor == 1:
            return np.asarray(self.data, dtype=np.float32, order="C")
        return _block_replicate(self.data, int(factor), conserve_sum=conserve_sum)

    # ───────────── geometry-aware resampling (returns new cutouts) ────────────

    def downsample(self, factor: int) -> "AlignedCutout":
        """
        Return a new cutout binned by integer `factor`:
          • flux-conserving (sum)
          • correct position & WCS updates
          • exact only if origin and shape are divisible by `factor`
        """
        f = int(factor)
        if f < 1:
            raise ValueError("factor must be >= 1")
        if f == 1:
            return deepcopy(self)

        H, W = self.shape
        x0, y0 = self.origin_original

        if (x0 % f) or (y0 % f) or (H % f) or (W % f):
            raise ValueError(
                "Downsample requires origin and size divisible by factor "
                f"(origin=({x0},{y0}), shape=({H},{W}), factor={f})."
            )

        data_lo = _block_reduce(self.data, f, func=np.sum)  # float32

        pos_lo = bin_remap(self.input_position_original, f)  # (x, y)
        shape_input_lo = (self.shape_input[0] // f, self.shape_input[1] // f)
        wcs_lo = scale_wcs_pixel(self.wcs, pixel_scale_factor=f, new_shape=shape_input_lo)

        # alignment propagates: new origin = old_origin / f
        align_lo = max(1, self.align // f)

        # build a new cutout on a dummy parent (zeros), then insert data
        dummy = np.zeros(shape_input_lo, dtype=np.float32)
        out = AlignedCutout(
            dummy, tuple(pos_lo), data_lo.shape, align=align_lo, copy=True, wcs=wcs_lo
        )
        out.data[...] = data_lo
        return out

    def upsample(self, factor: int, conserve_sum: bool = True) -> "AlignedCutout":
        """
        Return a new cutout expanded by integer `factor`:
          • uses block replication (optionally flux-conserving)
          • correct position & WCS updates
        """
        f = int(factor)
        if f < 1:
            raise ValueError("factor must be >= 1")
        if f == 1:
            return deepcopy(self)

        data_hi = _block_replicate(self.data, f, conserve_sum=conserve_sum)

        pos_hi = expand_remap(self.input_position_original, f)  # (x, y)
        shape_input_hi = (self.shape_input[0] * f, self.shape_input[1] * f)
        wcs_hi = scale_wcs_pixel(self.wcs, pixel_scale_factor=1.0 / f, new_shape=shape_input_hi)

        align_hi = self.align * f

        dummy = np.zeros(shape_input_hi, dtype=np.float32)
        out = AlignedCutout(
            dummy, tuple(pos_hi), data_hi.shape, align=align_hi, copy=True, wcs=wcs_hi
        )
        out.data[...] = data_hi
        return out


class Template(Cutout2D):
    """Cutout-based template storing slice bookkeeping."""

    FLAG_VALID = 0x01  # 0001: Template is valid
    FLAG_CONVOLVED = 0x02  # 0010: Template has been convolved
    FLAG_SUM_ZERO = 0x04  # 0100: Template sum is zero
    FLAG_HAS_NAN = 0x08  # 1000: Template contains NaN values
    FLAG_OUTSIDE_WEIGHT = 0x10  # 1 0000: Template is outside weight map
    FLAG_SHIFTED = 0x20  # 10 0000: Template has been shifted
    FLAG_PSF_EXTENDED = 0x40  # 100 0000: Template extended with PSF wings
    FLAG_EXTEND_FAILED = 0x80  # 1000 0000: PSF-wing extension attempted but skipped

    def __init__(
        self,
        data: np.ndarray,
        position: tuple[float, float],
        size: tuple[int, int],
        label: int | None = None,
        copy: bool = True,
        wcs: WCS | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            data,
            position,
            size,
            mode="partial",
            fill_value=0.0,
            copy=copy,
            wcs=wcs,
            **kwargs,
        )
        # do not allow writing into a view
        #        if not copy:
        #            self.data.flags.writeable = False

        # basic metadata
        # Store the original data reference
        #        self.base_data = data.copy()

        self.is_dirty = False  # flag to track if data needs to be updated
        # @@@ bug in Cutout2D: shape_input is not set correctly
        self.shape_input = data.shape
        self.shape_original = data.shape
        self.wcs_original = wcs

        # logical
        self.id = label
        self.id_parent = label  # @@@ this is redundant -> remove
        self.id_scene = 1
        self.name = "main"  # component name
        # Diagnostic flags (bitwise)

        self.flag = 0  # bitwise flag for diagnostics
        self.flag |= Template.FLAG_VALID
        self.is_star: bool = False  # set by pipeline from catalog flag_star

        # flux
        self.flux = 0.0
        self.flux_f444w: float = 0.0  # within-segmap F444W flux in image units (pre-normalization sum)
        self.n_pix: int = 0  # segmap pixel count at extraction time
        self.err = 0.0
        self.err_pred = 0.0  # predicted error from weight map and profile
        self.wnorm = 0.0  # weighted norm of the template d * w * d
        self.ee_rlim: float = 0.0
        self.ee_fraction: float = 1.0

        # astrometry
        # record shift from original position here
        # this is the intended shift from base_data to data
        self.to_shift = np.array([0.0, 0.0], dtype=float)  # impending shift
        self.shifted = np.array([0.0, 0.0], dtype=float)  # accumulated shift

    @property
    def bbox(self) -> tuple[int, int, int, int]:  # pragma: no cover - simple alias
        (ymin, ymax), (xmin, xmax) = self.bbox_original
        return int(ymin), int(ymax), int(xmin), int(xmax)

    def pad(
        self,
        padding: Tuple[int, int],
        original_shape: Tuple[int, int],
        *,
        image: np.ndarray | None = None,
        inplace=False,
    ) -> "Template":
        """Create a new Template with padding, maintaining correct original coordinates."""

        # force padding to be even, otherwise unpredictable behavior for cutout
        ony, onx = padding[0] // 2, padding[1] // 2
        ny, nx = self.data.shape

        # Create new Template directly from the original array reference
        # This ensures all coordinates remain consistent with the true original
        if image is None:
            image = np.zeros(self.shape_input, dtype=self.data.dtype)

        new_template = Template(
            data=image,
            position=self.input_position_original,
            size=(ny + ony * 2, nx + onx * 2),
            wcs=self.wcs,  # wcs will be wrong, offset by padding
            label=self.id,
        )

        # Now place the old data in our padded version
        new_template.data[ony : ony + ny, onx : onx + nx] = self.data

        # if inplace is True, update the current instance
        if inplace:
            # overwrite the current attributes with the new one
            self.__dict__.update(new_template.__dict__)

        return new_template

    # ------------------------------------------------------------------
    # centred, even-padding convolution
    # ------------------------------------------------------------------
    def convolve_cutout(
        self,
        kernel: np.ndarray,
        *,
        parent_image: np.ndarray | None = None,
        preserve_dtype: bool = True,
    ) -> "Template":
        """
        Convolve *this* template with a centred ``kernel`` **and return a new
        `Template` that already has the correct, larger geometry**.

        The routine guarantees that the padding applied to the original
        cut-out is **even** – i.e. an integer number of pixels *on both
        sides* – which avoids the odd-size artefacts you saw earlier.

        Parameters
        ----------
        kernel
            2-D, centred convolution kernel.
        parent_image
            Reference to the *full* parent image.  If ``None`` a tiny dummy
            array of zeros (same dtype) is created just to satisfy Cutout2D.
            It is **never** copied, so the memory cost is negligible.
        preserve_dtype
            Cast the result back to ``self.data.dtype`` (default) instead of
            keeping the float64 that `fftconvolve` returns.

        Returns
        -------
        Template
            A *new* template whose ``data`` attribute contains the full
            convolution result and whose spatial metadata (WCS, slices, …)
            is already consistent with the enlarged size.
        """
        # 1. --- full convolution -------------------------------------------------
        full = fftconvolve(self.data, kernel, mode="full")
        if preserve_dtype:
            full = full.astype(self.data.dtype, copy=False)

        ny, nx = full.shape

        if parent_image is None:
            # a 1-byte dummy is enough – Cutout2D only keeps a *view*
            parent_image = np.zeros(self.shape_input, dtype=self.data.dtype)

        # 2. make *sure* the new cut-out is large enough -----------------------
        #     If ny or nx is odd, add 1 so it becomes even (keeps later padding
        #     code happy) *and* ≥ full.shape.
        ny_even = ny if ny % 2 == 0 else ny + 1
        nx_even = nx if nx % 2 == 0 else nx + 1

        # # --------- 3. build a fresh Cutout2D --------------------------------
        new_cut = Template(
            parent_image,  # original full image reference
            position=self.input_position_original,  # note (x, y)
            size=(ny_even, nx_even),  # (ny, nx)
            wcs=self.wcs,  # note wcs origin is wrong
            label=self.id,
            copy=False,  # do not copy the data, we are replacing later
        )

        # copy the convolution result into the enlarged cut-out
        # account for the extra pixel
        # 4.  centre `full` inside the (possibly larger) even array -------------
        y0 = (ny_even - ny) // 2  # shift is 0 or 1
        x0 = (nx_even - nx) // 2
        data = np.zeros(new_cut.data.shape, dtype=self.data.dtype)
        data[y0 : y0 + ny, x0 : x0 + nx] = full
        new_cut.data = data
        #        new_cut.base_data = data  # also store it in base data
        new_cut.flag |= Template.FLAG_CONVOLVED  # mark as convolved

        # Propagate area + extension provenance unconditionally (not gated on
        # s > 0, unlike flux_f444w below): FLAG_SUM_ZERO templates must keep
        # their n_pix and extension flags so downstream bookkeeping is intact.
        new_cut.n_pix = self.n_pix
        new_cut.flag |= self.flag & (Template.FLAG_PSF_EXTENDED | Template.FLAG_EXTEND_FAILED)

        # Renormalize to unit sum so downstream code can treat templates uniformly.
        # Kernel.sum() is not guaranteed to equal 1 (numerical matching-kernel
        # construction), which otherwise biases the fitted amplitudes and aperture
        # corrections by 1/kernel.sum(). Propagate the un-normalization factor
        # unchanged: PSF matching preserves total source flux.
        s = float(new_cut.data.sum())
        if s > 0:
            new_cut.data /= s
            new_cut.flux_f444w = self.flux_f444w
        else:
            new_cut.flag |= Template.FLAG_SUM_ZERO

        return new_cut

    def downsample_wcs_old(self, image_lo: np.ndarray, wcs_lo, k: int) -> "Template":
        """
        Downsample this template to a lower resolution using the target image and WCS.

        Parameters
        ----------
        image_lo : np.ndarray
            The low-resolution image to extract the template from.
        wcs_lo : astropy.wcs.WCS
            The WCS of the low-resolution image.
        k : int
            Integer downsampling factor.

        Returns
        -------
        Template
            New template extracted from the low-res image using the correct WCS.
        """
        # Get the original position in the high-res WCS
        pos = self.input_position_cutout  # needs to be cutout coordinates
        ra, dec = self.wcs.wcs_pix2world(*pos, 0)

        # Convert RA/Dec to pixel coordinates in the low-res WCS
        # note: x_lo, y_lo are now original coordinates in the low-res image
        x_lo, y_lo = wcs_lo.wcs_world2pix(ra, dec, 0)

        # Calculate new size (downsampled)
        height, width = self.data.shape[0] // k, self.data.shape[1] // k
        # print('Original position:', pos)
        # print(f"Downsampling {self.id} from {self.data.shape} to {height, width} at pos ({x_lo}, {y_lo})")
        # print('original data shape:', self.shape_input, image_lo.shape)
        # print(self.wcs)
        # print(wcs_lo)
        #        Create the new template using the low-res image and WCS
        lowres_tmpl = Template(image_lo, (x_lo, y_lo), (height, width), wcs=wcs_lo, label=self.id)

        # Fill the data with block-reduced (averaged) values from the high-res template
        lowres_tmpl.data[:] = block_reduce(self.data, k, func=np.sum)
        return lowres_tmpl

    # block alignment methods currently not used
    @staticmethod
    def block_aligned(
        pos: np.ndarray,  # [x_c, y_c] (float)
        orig_size: np.ndarray,  # [ny, nx] (int)  <-- note reversed vs pos
        block_align: int,
        rfunc: Callable[[np.ndarray], np.ndarray] = np.ceil,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (size_aligned, idx_min_aligned) so that
            idx_min_aligned = rfunc(pos - size_aligned/2)
        and idx_min_aligned % k == 0,
        with the smallest even Δsize per axis.

        Conventions:
        - pos is [x, y]
        - size is [ny, nx]
        """

        # first force size to be minimum block size
        size = np.maximum(np.asarray(orig_size), block_align).astype(np.int64)

        # initial starts (paired with size[::-1] to match x↔nx, y↔ny)
        idx0 = rfunc(pos - size[::-1] / 2.0).astype(np.int64)  # [x0, y0]

        # minimal steps t: idx0 - t ≡ 0 (mod k)  ->  t ≡ idx0 (mod k)
        steps = idx0 % block_align  # [tx, ty]

        # add 2*steps to the paired sizes (map back with [::-1])
        dsize = 2 * steps[::-1]  # [dny, dnx]
        size_new = (size + dsize).astype(np.int64)  # [ny', nx']

        # recompute aligned starts
        idxmin = rfunc(pos - size_new[::-1] / 2.0).astype(np.int64)
        return size_new, idxmin

    # verified for k=2,4 for sizes 4-16
    def downsample(
        self, k: int, image: np.ndarray | None = None, wcs_lo: WCS | None = None
    ) -> "Template":
        """
        Flux-conserving k× downsample aligned to the global hi-res grid.
        Handles negative origins, preserves center-of-pixel convention.
        """
        from copy import deepcopy

        if k == 1:
            return deepcopy(self)

        H, W = self.data.shape

        # Global lower-left of this cutout (integer pixel indices, can be negative)
        # Cutout2D uses (x, y); ensure we keep that order consistent
        x0_hi, y0_hi = map(int, self._origin_original_true)

        # Phase to reach the next k-aligned boundary *inside* this cutout
        dx = (-x0_hi) % k
        dy = (-y0_hi) % k

        # Low-res size from the remaining pixels after phase adjustment
        hlo = H // k
        wlo = W // k
        if hlo <= 0 or wlo <= 0:
            raise ValueError("Cutout too small to downsample with current k/phase.")

        # Hi-res block aligned to k×k boundaries
        hi_aligned = self.data[dy : dy + hlo * k, dx : dx + wlo * k]
        # Flux-conserving reduction
        lo_block = block_reduce(hi_aligned, k, func=np.sum)

        # print(hlo, wlo, k, lo_block.shape, hi_aligned.shape)

        # Map the *center* correctly
        x_lo, y_lo = bin_remap(self.input_position_original, k)
        shape_input = np.array(self.shape_input) // k

        if image is None:
            image = np.zeros(shape_input)
        # Build the low-res Template at the correct fractional center
        low = Template(image, (x_lo, y_lo), (hlo, wlo), wcs=wcs_lo, label=self.id)

        ly, lx = lo_block.shape
        # print(wlo, hlo, low.shape)
        # print(dx, dy, ly, lx)
        low.data[:ly, :lx] = lo_block

        return low


class Templates:
    """Container for source templates."""

    def __init__(self, min_size: int = 8) -> None:
        # Minimum (even) cutout size in pixels. extract_templates enforces it as
        # ``(min_size // 2) * 2``, so an odd value would lose a pixel; callers
        # that size for PSF wings should pass an even floor (see
        # min_size_from_aperture).
        self.min_size = int(min_size)
        self._templates: List[Template] = []

    @staticmethod
    def min_size_from_aperture(
        aperture_diam_arcsec: float, wcs: WCS, margin: float = 1.5
    ) -> int:
        """Smallest even cutout size (pixels) enclosing a photometry aperture.

        Parameters
        ----------
        aperture_diam_arcsec : float
            Photometry aperture diameter in arcsec.
        wcs : astropy.wcs.WCS
            WCS of the high-res (detection) image, for the pixel scale.
        margin : float
            Multiplicative margin on the aperture diameter (default 1.5).

        Returns
        -------
        int
            Even pixel count; rounded up so the (min_size // 2) * 2 floor in
            extract_templates does not drop a pixel.
        """
        # Use the finer axis (smallest arcsec/pixel) so the square cutout floor
        # contains the aperture on both axes when pixels are non-square.
        pscale_arcsec = float(np.min(proj_plane_pixel_scales(wcs))) * 3600.0
        diam_pix = aperture_diam_arcsec / pscale_arcsec
        size = int(np.ceil(diam_pix * margin))
        return size + (size % 2)

    def __len__(self) -> int:
        return len(self._templates)

    def __getitem__(self, idx: int) -> Template:
        return self._templates[idx]

    def __iter__(self) -> Iterator[Template]:
        return iter(self._templates)

    def add_component(
        self,
        parent: Template,
        data: np.ndarray,
        component: str,
        **kwargs: Any,
    ) -> Template | None:
        """Clone ``parent`` and append a new component template.

        Parameters
        ----------
        parent
            The template providing spatial metadata.
        data
            Pixel data for the new component. Must match the shape of
            ``parent.data``.
        component
            Informational tag describing the component type.
        **kwargs
            Additional attributes to set on the cloned template.

        Returns
        -------
        Template | None
            The newly created template or ``None`` if the component was
            discarded due to high similarity with ``parent``.
        """

        arr_parent = parent.data[parent.slices_cutout]
        arr_new = data[parent.slices_cutout]
        norm_p = np.linalg.norm(arr_parent.ravel())
        norm_n = np.linalg.norm(arr_new.ravel())
        if norm_p > 0 and norm_n > 0:
            corr = float(np.dot(arr_parent.ravel(), arr_new.ravel()) / (norm_p * norm_n))
            if corr > 0.999:
                logger.info(
                    "Skipping component %s for source %s due to high similarity (%.3f)",
                    component,
                    parent.id,
                    corr,
                )
                return None

        tmpl = deepcopy(parent)
        tmpl.data = data
        tmpl.component = component
        tmpl.id_parent = parent.id_parent or parent.id
        for key, val in kwargs.items():
            setattr(tmpl, key, val)

        self._templates.append(tmpl)
        return tmpl

    @classmethod
    def from_image(
        cls,
        hires_image: np.ndarray,
        segmap: np.ndarray,
        positions: Iterable[Tuple[float, float]],
        kernel: np.ndarray | None = None,
        extension: np.ndarray | str | None = None,  # 'psf', 'wings', 'both', None
        wcs: WCS | None = None,
    ) -> "Templates":
        obj = cls()
        obj.wcs = wcs

        # Step 1: Extract raw cutouts
        obj.extract_templates(hires_image, segmap, positions, wcs=wcs)

        # if type(extension) == np.ndarray:
        # Extend templates with PSF wings
        # obj.extend_with_psf_wings(extension, inplace=True)

        # Step 2: Convolve with kernel (includes padding)
        if kernel is not None:
            obj.convolve_templates(kernel, inplace=True)

        return obj

    # ------------------------------------------------------------
    # static helpers
    # ------------------------------------------------------------
    @staticmethod
    def apply_template_shifts(templates: Sequence[Template]) -> None:
        """Apply stored ``shift`` values to templates in-place.

        Parameters
        ----------
        templates:
            Sequence of :class:`~mophongo.templates.Template` objects whose
            ``shift`` attribute encodes the ``(dx, dy)`` offset to apply.
        Sign convention:
        Let (dx, dy) be the image→template correction predicted by astrometry,
        i.e. “shift the image by (dx,dy) to match the template.”
        When applied to template, we must shift the template by (-dx,-dy).
        And scipy.ndimage.shift takes shifts in (axis0, axis1) = (y, x) order.
        """
        from scipy.ndimage import shift as nd_shift

        for tmpl in templates:
            #            if not tmpl.is_dirty:  # skip if shift was already applied
            #                continue

            dx, dy = map(float, tmpl.to_shift)
            if abs(dx) < 1e-2 and abs(dy) < 1e-2:
                continue

            # sign convention: image is shifted, so we reverse shift the template
            # positive shift is from image to template, but here we shift the template
            #            x0, y0 = tmpl.input_position_original
            # @@@ isnt it better to shift the data, because that only affects the Atb vector
            tmpl.data = nd_shift(
                tmpl.data,
                (dy, dx),
                order=3,
                mode="constant",
                cval=0.0,
                prefilter=True,
            )
            tmpl.shifted += [dx, dy]  # accumulate in case of iterating
            tmpl.to_shift[:] = 0.0
            tmpl.flag |= Template.FLAG_SHIFTED  # mark as shifted

    @staticmethod
    def _prepare_fft_fast(psf: np.ndarray) -> tuple[np.ndarray, np.ndarray, interp1d]:
        """Return radial profile, EE curve and inverse profile interpolator."""
        y, x = np.indices(psf.shape)
        cy, cx = (np.array(psf.shape) - 1) / 2
        r = np.hypot(y - cy, x - cx)
        r_int = r.astype(int)
        prof_num = np.bincount(r_int.ravel(), psf.ravel())
        prof_den = np.bincount(r_int.ravel())
        prof = prof_num / np.maximum(prof_den, 1)
        rr = np.arange(len(prof))
        ee = np.cumsum(prof * 2 * np.pi * rr)
        if ee[-1] > 0:
            ee /= ee[-1]
        p2r = interp1d(
            prof[::-1],
            rr[::-1],
            bounds_error=False,
            fill_value=(rr.max(), rr.max()),
        )
        return prof, ee, p2r

    @staticmethod
    def _crop_kernel(kern: np.ndarray, rlim: float) -> tuple[np.ndarray, float]:
        """Crop ``kern`` around its centre to ``rlim`` pixels."""
        r = int(np.ceil(rlim))
        cy, cx = (np.array(kern.shape) - 1) / 2
        size_y = min(2 * r + (kern.shape[0] % 2), kern.shape[0])
        size_x = min(2 * r + (kern.shape[1] % 2), kern.shape[1])
        cut = Cutout2D(kern, (cx, cy), (size_y, size_x), mode="trim", copy=False)
        kc = cut.data
        return kc, float(kc.sum())

    @staticmethod
    def prepare_kernel_info(
        templates: list["Template"],
        psf_full: np.ndarray,
        image_770: np.ndarray,
        weight_770: np.ndarray | None,
        *,
        eta: float,
        r_min_pix: float = 1.0,
        r_max_pix: float | None = None,
    ) -> None:
        """Compute quick-flux based kernel crop radius and encircled energy."""
        if not eta:
            return

        prof, ee, p2r = Templates._prepare_fft_fast(psf_full)
        rr = np.arange(len(prof))

        if weight_770 is not None:
            sigma_pix = float(np.median(np.sqrt(1 / weight_770[weight_770 > 0])))
        else:
            sigma_pix = float(np.std(image_770))

        qflux = Templates.quick_flux(templates, image_770)

        for tmpl, Fq in zip(templates, qflux):
            if not np.isfinite(Fq) or Fq <= 0:
                tmpl.ee_rlim = 0.0
                tmpl.ee_fraction = 1.0
                continue

            thresh = float(eta) * sigma_pix / Fq
            thresh = np.clip(thresh, prof.min(), prof.max())
            r_pix = float(p2r(thresh))
            r_pix = max(r_min_pix, r_pix)
            if r_max_pix is not None:
                r_pix = min(r_pix, r_max_pix)
            tmpl.ee_rlim = r_pix
            tmpl.ee_fraction = float(np.interp(r_pix, rr, ee))

    @staticmethod
    def quick_flux(templates: List[Template], image: np.ndarray) -> np.ndarray:
        """Return quick flux estimates based on template data and image."""
        flux = np.zeros(len(templates), dtype=float)
        for i, tmpl in enumerate(templates):
            tt = tmpl.data[tmpl.slices_cutout]
            img = image[tmpl.slices_original]
            ttsqs = np.sum(tt**2)
            flux[i] = np.sum(img * tt) / ttsqs if ttsqs > 0 else 0.0
            tmpl.flux = flux[i]  # Store quick flux in the template for later use
        return flux

    @staticmethod
    def predicted_errors(templates: List[Template], weights: np.ndarray) -> np.ndarray:
        """Return per-source uncertainties ignoring template covariance."""
        pred = np.empty(len(templates), dtype=float)
        for i, tmpl in enumerate(templates):
            w = weights[tmpl.slices_original]
            inverse_epred = np.sqrt(np.sum(w * tmpl.data[tmpl.slices_cutout] ** 2))
            if inverse_epred > 0:
                pred[i] = 1.0 / inverse_epred
            else:  # @@@ need to debug why this happens should never have zero weight
                logger.debug(
                    f"error for template {i}: {inverse_epred} FLAG_SUM_ZERO {tmpl.flag & Template.FLAG_SUM_ZERO}"
                )
                tmpl.flag |= Template.FLAG_SUM_ZERO

            tmpl.err = pred[i]  # Store RMS in the template for later use
        return pred

    def prune_outside_weight(self, weight: np.ndarray, rtol: float = 1e-8) -> List[Template]:
        """Remove templates with no overlap with the provided ``weight`` map.

        A template is discarded if all pixels belonging to its segmentation
        footprint fall on non-positive weight values. The check is performed in
        the original image coordinates using ``tmpl.slices_original``.

        Parameters
        ----------
        weight : np.ndarray
            Weight map aligned with ``self.original_shape``.

        Returns
        -------
        list[Template]
            Remaining templates after pruning.
        """
        norms = []
        for tmpl in self._templates:
            sl = tmpl.slices_original
            data = tmpl.data[tmpl.slices_cutout]
            w = weight[sl]
            wnorm = float(np.sum(data * w * data))
            tmpl.wnorm = wnorm
            norms.append(wnorm)

        atol = rtol * np.median(norms)
        keep = [t for t in self._templates if t.wnorm > atol]

        dropped = len(self._templates) - len(keep)
        if dropped:
            print(f"Pruned {dropped} templates with low L2 norm on weight map.")
        self._templates = keep
        return self._templates

    @property
    def templates(self) -> List[Template]:
        """Return the list of templates."""
        return self._templates

    def extract_templates(
        self,
        hires_image: np.ndarray,
        segmap: np.ndarray,
        positions: Iterable[Tuple[float, float]],
        wcs: WCS | None = None,
    ) -> list[Template]:
        """Extract cutout templates around segmentation regions."""

        self.original_shape = hires_image.shape
        segm = SegmentationImage(segmap)
        templates: list[Template] = []
        ny, nx = hires_image.shape

        for pos in tqdm(positions, desc="Extracting templates"):
            # silently skip invalid positions
            if not np.isfinite(pos).all():
                continue
            x, y = int(round(pos[0])), int(round(pos[1]))
            if y < 0 or y >= ny or x < 0 or x >= nx:
                continue
            label = segm.data[y, x]
            if label == 0:
                continue

            idx = segm.get_index(label)
            bbox = segm.bbox[idx]
            segm.slices[idx]

            # Make bbox symmetric around the center to ensure proper centering
            # enfore minimum size
            height = max(y - bbox.iymin, bbox.iymax - y, self.min_size // 2) * 2
            width = max(x - bbox.ixmin, bbox.ixmax - x, self.min_size // 2) * 2

            # Create template cutout
            cut = Template(hires_image, pos, (height, width), wcs=wcs, label=label)

            # zero out all non segment pixels
            seg_mask = segm.data[cut.slices_original] == label
            cut.data[cut.slices_cutout] *= seg_mask.astype(cut.data.dtype)
            cut.n_pix = int(seg_mask.sum())  # segmap pixel count at extraction

            # sum data should never be zero. There should
            # there should also never be NaNs.
            # Normalize the template so its sum is 1 (if nonzero)
            total = cut.data.sum()
            cut.flux_f444w = float(total)
            if total != 0:
                cut.data /= total
            else:
                cut.flag |= Template.FLAG_SUM_ZERO

            templates.append(cut)

        self._templates = templates
        return templates

    def convolve_templates(
        self,
        kernel: np.ndarray | PSFRegionMap | None,
        inplace: bool = False,
    ) -> list[Template]:
        """Convolve all templates with ``kernel``.

        Parameters
        ----------
        kernel : np.ndarray or PSFRegionMap or None
            Convolution kernel matching the template resolution. If ``None``,
            templates are returned unchanged (aside from optional padding).
            If templates have ``ee_rlim`` set via :meth:`prepare_kernel_info`,
            kernels are cropped to this radius and their ``ee_fraction`` is
            stored on each template.
        inplace : bool, optional
            If ``True``, templates are modified in place and the internal list
            is returned. Otherwise a new list of convolved templates is
            produced.

        Returns
        -------
        list of Template
            Convolved templates.
        """

        if not self._templates:
            raise ValueError("No templates to convolve. Run extract_templates first.")

        tmpls = self._templates
        original_shape = self.original_shape
        dummy_image = np.zeros(original_shape, dtype=np.byte)

        new_templates: list[Template] = []
        for i, tmpl in enumerate(tqdm(tmpls, desc="Convolving templates")):

            # Obtain kernel for this template
            if isinstance(kernel, PSFRegionMap):
                x, y = tmpl.position_original
                if tmpl.wcs is not None:
                    ra, dec = tmpl.wcs.wcs_pix2world(x, y, 0)
                else:
                    ra, dec = x, y
                kern = kernel.get_psf(ra, dec)
            else:
                kern = kernel

            if tmpl.ee_rlim > 0.0 and tmpl.ee_fraction < 1.0:
                kern, tmpl.ee_fraction = Templates._crop_kernel(kern, tmpl.ee_rlim)

            new_tmpl = tmpl.convolve_cutout(kern, parent_image=dummy_image)

            if not inplace:
                new_templates.append(new_tmpl)

        return new_templates if not inplace else self._templates

    def extend_with_psf_wings(
        self,
        psf: np.ndarray | PSFRegionMap,
        *,
        target_ee: float = 0.95,
        inplace: bool = True,
    ) -> List[Template]:
        """Extend segmap-truncated templates with scaled PSF wings.

        A template whose segmap captures only a small fraction of the PSF (small
        ``n_pix``) is a poor shape model because it is missing the PSF wings. For
        every template with ``0 < n_pix < ee_area`` -- where ``ee_area`` is the
        ``target_ee`` encircled-energy area of the detection PSF -- the PSF,
        scaled to the in-segment flux, is pasted into the pixels outside the
        segment (within the ``target_ee`` radius), making the template a faithful
        PSF shape. Templates with an adequate segmap (``n_pix >= ee_area``) and
        empty templates (``n_pix == 0``) are left untouched.

        Flux bookkeeping. ``flux_f444w`` (the within-segmap detection flux that the
        Mode-B aperture-correction loop reads off the original template) is updated
        to the inferred total ``flux_f444w / f_seg``, where ``f_seg`` is the
        fraction of the PSF flux that falls in the segment. This is derived from
        the full (uncropped) PSF, so the EE-crop applied to the pasted pixels --
        which only bounds the added array footprint -- does not bias the
        denominator low. The matching/convolution PSF elsewhere is not cropped by
        this routine.

        Parameters
        ----------
        psf : np.ndarray or PSFRegionMap
            Detection-band PSF (e.g. F444W). For a PSFRegionMap the per-source PSF
            is looked up by sky position, mirroring :meth:`convolve_templates`; the
            size threshold uses the map's first (representative) PSF.
        target_ee : float
            Encircled-energy fraction setting the size threshold and the radius to
            which the pasted PSF is cropped.
        inplace : bool
            If True (default) modify and return the stored templates, avoiding a
            peak-memory copy. Otherwise operate on deep copies.

        Returns
        -------
        list of Template
        """
        is_map = isinstance(psf, PSFRegionMap)
        rep_psf = np.asarray(psf.psfs[0] if is_map else psf, dtype=float)
        ee_area = psf_ee_area_pix(rep_psf, target_ee)

        templates = self._templates if inplace else [deepcopy(t) for t in self._templates]
        for tmpl in tqdm(templates, desc="Extending with PSF wings"):
            # Idempotency: never re-process an already-handled template -- on a
            # second pass (or post-convolution) ``data > 0`` would include the
            # pasted wings and re-inflate flux_f444w.
            if tmpl.flag & (Template.FLAG_PSF_EXTENDED | Template.FLAG_EXTEND_FAILED):
                continue
            if not (0 < tmpl.n_pix < ee_area):
                continue

            # Per-source detection PSF (mirror convolve_templates for a map)
            if is_map:
                x, y = tmpl.position_original
                if tmpl.wcs is not None:
                    ra, dec = tmpl.wcs.wcs_pix2world(x, y, 0)
                else:
                    ra, dec = x, y
                psf_src = psf.get_psf(ra, dec)
            else:
                psf_src = rep_psf
            if psf_src is None:
                tmpl.flag |= Template.FLAG_EXTEND_FAILED
                continue
            psf_src = np.asarray(psf_src, dtype=float)
            psf_total = float(psf_src.sum())
            if psf_total <= 0:
                tmpl.flag |= Template.FLAG_EXTEND_FAILED
                continue

            data = tmpl.data
            ny, nx = data.shape
            xs, ys = tmpl.input_position_cutout  # float source position (x, y)

            # The target-EE disk must fit inside the cutout. Otherwise the pasted
            # wings clip at the edge while flux_f444w is raised to the full total,
            # silently recreating the truncated-template bias this routine fixes.
            # Fail rather than corrupt; correct usage sizes the cutout via
            # min_size_from_aperture / 2 * psf_ee_radius_pix.
            ee_r = psf_ee_radius_pix(psf_src, target_ee)
            if ee_r > min(xs, nx - 1 - xs, ys, ny - 1 - ys):
                tmpl.flag |= Template.FLAG_EXTEND_FAILED
                continue

            # Bilinear-sample the PSF onto the cutout grid at the sub-pixel
            # source position; normalise so it sums to the PSF flux fraction.
            yy, xx = np.mgrid[0:ny, 0:nx]
            pcy = (psf_src.shape[0] - 1) / 2.0
            pcx = (psf_src.shape[1] - 1) / 2.0
            coords = np.array([pcy + (yy - ys), pcx + (xx - xs)])
            psf_cut = map_coordinates(psf_src, coords, order=1, mode="constant", cval=0.0)
            psf_cut /= psf_total

            # Segment footprint captured before pasting.
            seg = data > 0
            f_seg = float(psf_cut[seg].sum())  # PSF fraction in the segment
            if f_seg < 1e-8:  # negligible overlap -> cannot scale reliably
                tmpl.flag |= Template.FLAG_EXTEND_FAILED
                continue

            # Scale so the PSF's in-segment sum matches the template's, then paste
            # the wings outside the segment within the target-EE radius, in place.
            s = float(data[seg].sum()) / f_seg
            within = (xx - xs) ** 2 + (yy - ys) ** 2 <= ee_r**2
            paste = (~seg) & within
            data[paste] += s * psf_cut[paste]

            # Raise flux_f444w to the inferred total (uncropped PSF), so the
            # Mode-B denominator relaxes. Updated on this template object, which
            # is the original when inplace=True.
            tmpl.flux_f444w = tmpl.flux_f444w / f_seg
            tmpl.flag |= Template.FLAG_PSF_EXTENDED

        return templates


# ---------------------------------------------------- obsolete methods -------------------


def _convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve ``image`` with ``kernel`` using direct sliding windows."""
    ky, kx = kernel.shape
    pad_y, pad_x = ky // 2, kx // 2
    pad_before = (pad_y, pad_x)
    pad_after = (ky - 1 - pad_y, kx - 1 - pad_x)
    padded = np.pad(image, (pad_before, pad_after), mode="constant")
    from numpy.lib.stride_tricks import sliding_window_view

    windows = sliding_window_view(padded, kernel.shape)
    return np.einsum("ijkl,kl->ij", windows, kernel)
