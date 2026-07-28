from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np

# full weights need to be calcuate like
# template_var = scipy.signal.fftconvolve(K**2, 1 / wht1, mode='same')  # same shape as template
# Iterate if needed (since A appears in w(x)):
# First fit using weights = wht2
# Compute A (amplitude)
# Recompute weights using full formula
# Refit using updated weights if you want accurate errors
# wht_tot = 1 / (1 / wht2 + A**2 * template_var)
# Pass wht_tot to the fitter.
# Multiple templates: you must apply the same logic to each template independently. This means different pixels may have different total weights for each template, depending on each one's amplitude and support.
# Correlated templates (overlapping) require full covariance accounting; your current implementation approximates this by assuming per-template independence.
# If template noise is negligible, simplify to: weights = wht2 (as in your current default).
# Flux-dependent variance (via A^2) introduces mild nonlinearity; it's safe to fix A from initial fit for a single iteration.


@dataclass
class FitConfig:
    """Configuration options for the scene solver and the photometry pipeline.

    This module holds only the configuration dataclass; the solver itself
    lives in :mod:`mophongo.scene` / :mod:`mophongo.scene_fitter`. The legacy
    ``SparseFitter`` that used to live here was retired -- see
    ``docs/dead_code.md``.
    """

    positivity: bool = True
    bad_value: float = np.nan

    # Astrometry is fit jointly with the fluxes, per scene. This is the single
    # astrometry on/off + iteration knob: 0 disables the shift block entirely
    # (flux-only solve), N > 0 runs N joint refinement passes. The old
    # `fit_astrometry_joint` flag was removed -- it named a "joint vs separate"
    # choice that no longer exists (the separate fit-then-measure-residual path
    # lived in the retired legacy solver), so it could only ever mean on/off and
    # duplicated this field. See docs/dead_code.md.
    fit_astrometry_niter: int = 2  # Number of astrometry refinement passes (0 → disabled)
    # --- astrometry options -------------------------------------------------
    reg_astrom: float = 1e-4
    snr_thresh_astrom: float = 15.0  # 0 → keep all sources
    astrom_isolation_thresh: float = 0.5  # min flux dominance to include in astrometry (0–1); 0.0 = no cut
    astrom_model: str = "gp"  # 'polynomial' or 'gp'
    astrom_centroid: str = "centroid"  # "centroid" (=old) | "correlation"
    astrom_kwargs: dict[str, dict] = field(
        default_factory=lambda: {"poly": {"order": 0}, "gp": {"length_scale": 400}}
    )
    #    astrom_kwargs={'poly': {'order': 2}, 'gp': {'length_scale': 400}}
    multi_resolution_method: str = "upsample"  # 'upsample' or 'downsample'
    # None → derive from astrometric model order in __post_init__
    # Minimum bright sources per scene. If None reverts to (n_poly+1)*(n_poly+2)
    scene_minimum_bright: int = 5

    # Photometry aperture control:
    # - float/int: fixed aperture diameter size (in arcsec or pixels per `aperture_units`)
    # - str: column name in the input catalog for per-source aperture sizes
    # - None: fallback to 1.5 * FWHM (in pixels) measured from template
    aperture_diam: float | np.ndarray | None = None  # image measurement aperture (diameter)
    aperture_catalog: float | str | None = None  # catalog aperture (diameter or table column name)
    aperture_units: str = "arcsec"  # "arcsec" or "pix"
    f444w_col: str | None = None  # catalog column for F444W total flux (enables Yoshi Mode B correction)
    # Catalog columns for the Stage-3b two-step catalog tie (design doc Sec 5.4):
    # f444w_totcor_col = the aperture(color)->total factor (e.g. "tot_cor");
    # currently unused (kept for diagnostics/alternative ties -- the internal
    # Kron total is measured directly from the model, not this catalog column).
    # f444w_aper_col = the color-aperture DIAMETER in arcsec (e.g. "use_aper"),
    # consumed by _add_aperture_photometry as the Kron circular-radius floor
    # (r_floor_pix = 0.5*f444w_aper_col/pscale_ref); without it tcor_int falls
    # back to 1/apF_corr for every source.
    f444w_totcor_col: str | None = None
    f444w_aper_col: str | None = None

    # Template extension beyond the segmap (Estimator-3). Each source's composite
    # template H is a single radial SNR-weighted linear blend of the real
    # detection-image data and a data-anchored PSF model M, applied uniformly
    # over the source's owned stamp (docs/aperture_corrections.md Sec 5.1):
    #   H = w*data + (1-w)*M, w in [0, 1].
    # One core weight (the whole segment, from its in-segment SNR) and one
    # weight per radial halo annulus (from that annulus' own SNR) -- both from
    # the same ``blend_weight`` onset/rolloff (templates.py), so real data wins
    # wherever it has SNR and the PSF takes over smoothly wherever it doesn't.
    # Halo weights are forced monotone non-increasing outward, seeded at the
    # core weight, so a faint core caps its halo. Onset semantics: the core
    # weight saturates at 1 for snr_seg >= 1.5*fit_snrlo_psf; each halo
    # annulus' weight saturates at 1 for its own SNR >= wings_snr_psf.
    # Requires psfs[0]; falls back to truncated templates with a warning if the
    # detection PSF/WCS is absent.
    template_extend_mode: str = "auto"  # "none" | "auto"
    # Core-blend onset (IDL fit_snrlo_psf): the in-segment SNR at which the core
    # weight saturates at 1 (pure data) is 1.5*fit_snrlo_psf; below that it rolls
    # off toward the PSF model. 0 disables the core blend (weight pinned at 1).
    fit_snrlo_psf: float = 10.0
    # Halo-annulus-blend onset: the per-annulus SNR at which that annulus' weight
    # saturates at 1 (pure data); below that it rolls off toward the PSF model.
    wings_snr_psf: float = 3.0
    # Blend-weight rolloff exponent (templates.blend_weight): w = min(1, (snr/thresh)**p).
    template_blend_p: float = 2.0
    # Halo radial-annulus width (arcsec, converted to detection-image pixels via
    # the template WCS) for the per-annulus SNR/weight measurement.
    template_blend_annulus: float = 0.15
    extend_template_ee: float = 0.95  # encircled-energy fraction: PSF reach & max template-size cap
    # --- deprecated PSF-wing extension flags (broken in-place implementation; do not use) ---
    extend_template_segmap: bool = False  # DEPRECATED: old in-place extension, kept False
    extend_template_min_size_margin: float = 1.5  # cutout margin for min_size sizing

    # scene processing
    scene_coupling_thresh: float = 1e-3  # 1% leakage threshold for scene splitting
    scene_max_merge_radius: float = np.inf  # Max distance (px) to merge underfilled scenes (default: inf = no limit)
    generate_scene_catalog: bool = False  # If True, generate scene catalog and exit

    def __post_init__(self):
        # Validate template extension mode (guards typos like "Auto"/"non").
        # Legacy mode names (data/psf/hybrid) collapse to the single auto tree.
        if self.template_extend_mode in {"data", "psf", "hybrid"}:
            self.template_extend_mode = "auto"
        valid_modes = {"none", "auto"}
        if self.template_extend_mode not in valid_modes:
            raise ValueError(
                f"template_extend_mode must be one of {sorted(valid_modes)}, "
                f"got {self.template_extend_mode!r}"
            )
        # Derive scene_minimum_bright from astrometric polynomial order if not provided
        if self.scene_minimum_bright is None:
            try:
                poly_order = int(self.astrom_kwargs.get("poly", {}).get("order", 1))
            except Exception:
                poly_order = 0
            # default to 2x # of Chebyshev terms + 1
            n_poly = (poly_order + 1) * (poly_order + 2)
            self.scene_minimum_bright = n_poly + 1
