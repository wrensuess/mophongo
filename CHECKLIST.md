# Project Checklist

This checklist tracks tasks for building the photometry pipeline using Poetry and pytest.

## Setup
- [x] Initialize repository with `pyproject.toml` and Poetry
- [x] Create base package structure under `src/`
- [x] Add basic test suite with `pytest`

## Dependencies
- [x] Add `numpy`, `scipy`, and `astropy` to project dependencies
- [x] Run `poetry install` to install all dependencies
- [x] Added `nbformat` for generating example notebooks

## assumptions input data
- [x] input data are images + wcs, and weights that are proportional to variance
- [x] input catalog is catalog of sources positions: id, ra, dec
- [x] detection image, and associated segmentation map image, where each pixel can only belong to a source of a certain id.

## Core Modules
- [x] **PSF utilities** (`src/mophongo/psf.py`)
  - [x] `moffat_psf` Generate Moffat PSF images (ellipticity/FWHM/beta parameters).
  - [x] `matching_kernel` to Compute convolution kernels to transform the high‑resolution PSF into the low‑resolution PSF (Fourier domain or direct numerical solution)
  - [x] Added `recenter` option to `psf_matching_kernel` to shift kernels to their centroid
  - [x] Add methods to fit Moffat and Gaussian profiles to existing PSF arrays
  - [x] Added `PSF.delta` for symmetric delta-function PSFs
  - [x] Added `PSF.from_star` constructor for extracting PSFs from images
  - [x] Added `PSF.gaussian_matching_kernel` and `DrizzlePSF.register`
  - [x] Added `matching_kernel_basis` with Gauss–Hermite and multi-Gaussian basis sets
  - [x] Added `CircularApertureProfile` utility for radial profile and curve of growth
  - [x] Implement JWST STDPSF extension utility for STPSF / Webb PSF
  - [x] Implement drizzling PSF
  - [x] Build PSF region map from exposure footprints
  - [x] Add PA-based coarsening option to PSFRegionMap
  - [x] Added spatially varying kernel support in `run` and template convolution
  - [x] Implemented basic `Catalog` for source detection
  - [x] Added configurable detection parameters in `Catalog`
  - [x] Implemented star finder in Catalog
- [x] **Template builder** (`src/mophongo/templates.py`)
  - [x] `extract_templates` to create PSF-matched templates
  - [x] Extract per-object cutouts from the high‑res image using the detection segmentation.
  - [x] Normalize cutouts to unit flux and convolve each with the PSF kernel to produce a template in the low‑res pixel grid.
  - [x] Store bounding box coordinates for later overlap calculations.
  - [x] Introduced Cutout2D-based template extraction and normal matrix helpers
- [x] **Sparse fitter** (`src/mophongo/fit.py`)
  - Build sparse normal matrix AᵀA and vector Aᵀb using the templates and low‑res image (weights from inverse variance).
  - Solve for fluxes with scipy.sparse.linalg.cg (plus optional positivity and residual regularization).
  - Create the modeled low‑res image and residual map.
  - [x] Added GlobalAstroFitter for astrometric correction
  - [x] Added polynomial-based local astrometric correction
  - [x] Added safeguards against singular normal matrices
- [x] Added Gaussian-process-based local astrometric correction
- [x] Introduced `AstroCorrect` for pluggable local astrometry models
- [x] Added static utilities in `AstroCorrect` for applying stored template shifts and building polynomial predictors
- [x] Merged astrometry modules and added `AstroMap` for image-to-image shift mapping
- [x] Removed deprecated `fit_astrometry` flag in `FitConfig`; use `fit_astrometry_niter` only
- [x] Added ILU preconditioner and SuperLU-based flux error estimation with Hutchinson fallback
- [x] Added LSQR-based matrix-free solver (`solve_lo`)
- [ ] Deduplicate templates using weighted overlap cosine similarity
- [x] Consolidated flux and RMS estimation into parent `SparseFitter`
- [x] Added STRtree-based normal matrix builder (`build_normal_tree`)
 - [x] Removed deprecated `fit_astrometry` flag in `FitConfig`; use `fit_astrometry_niter` only
  - [x] Added ILU preconditioner and SuperLU-based flux error estimation with Hutchinson fallback
  - [x] Added LSQR-based matrix-free solver (`solve_lo`)
  - [ ] Deduplicate templates using weighted overlap cosine similarity
  - [x] Consolidated flux and RMS estimation into parent `SparseFitter`
  - [x] Added STRtree-based normal matrix builder (`build_normal_tree`)
  - [x] Added component-wise CG solver using STRtree groups
- [x] Added component-wise solver with shift blocks
- [x] Whitened component solver with sparse Cholesky preconditioner
- [x] Renamed component terminology to scene and centralized whitening in scene solver
- [x] Introduced stateless `SceneFitter` and `Scene` utilities
- [x] Fixed alpha0 scaling and Cholesky whitening in scene solver
- [x] Added `Scene.plot` for scene-level diagnostics
- [x] Adjusted Chebyshev basis to accept [-1,1] inputs and added edge tests
- [x] **Pipeline orchestrator** (`src/mophongo/pipeline.py`)
  - [x] `run` to tie all pieces together
  - [x] don't implement source detection just yet: assume detection + segmentation image + catalog are available.
  - [x] Load or receive arrays for the images, catalog, and PSFs.
  - [x] Call template builder, construct sparse system, solve for fluxes, and return a table of measurements plus residuals.
  - [x] Propagate RMS images as weights to compute flux uncertainties
  - [x] Prune templates lacking weight overlap before convolution
  - [x] Enabled template deduplication after extraction
- [x] Added multi-template second pass for poor-fit sources
- [x] Added integer-factor multi-resolution support with template and kernel downsampling
- [x] Block templates and PSFs before convolution with `block_reduce` and centroid-preserving PSF shifts
- [x] Downsample templates and kernels in the pipeline prior to convolution to avoid per-source PSF rebinning
- [x] Introduced `Pipeline` class to persist images and fit results
- [x] Consolidated catalog matching and flux extraction into helper methods
- [x] Added aperture photometry on model+residual with PSF correction
- [x] Added Mode B aperture correction: corr = f444w_catalog_total / aperture(flux_f444w_i * T_conv_i + residual_conv, r), per-source denominator using within-segmap F444W flux (tmpl.flux_f444w) + scene residual convolved to MIRI resolution
- [x] **Simulation utilities for tests** (`tests/utils.py`)
  - [x] Create fake catalogs and images with Moffat sources of varying size and ellipticity. positions are ra,dec
  - [x] Produce matching high‑res and low‑res PSFs, with low res PSF at least 5x high res PSF.
  - [x] max 50 sources, max 300 x 300 pixel high resolution image
  - [x] Convolve with a kernel derived from different PSFs to obtain the low‑resolution image and add Gaussian noise.
  - [x] Run the pipeline with the known PSFs and verify recovered fluxes agree with input fluxes within ≈5%.
  - [x] Check that the residual image contains only noise (no strong artifacts).
  - [x] Test failure modes (e.g., negative flux regularization) on a subset of sources.
  - [x] Add simulated data utilities in `tests/utils.py`  
  - [x] Create end-to-end tests in `tests/test_pipeline.py`
    
## Testing
- [x] Run `pytest` to ensure all tests pass
- [x] Save diagnostic plot during pipeline test
- [x] Save diagnostic plots for PSF, fitter and template tests
- [x] Save output catalog to disk during pipeline test
- [x] Benchmarked key pipeline steps in `tests/test_benchmark.py`

## TODO
- [ ] scan for bug fixes / robustness improvements
  - [x] align PSF components to fractional template centers
  - [ ] automated way of determining optimal convolution kernels for PSF  
- [ ] storage
  - [ ] best way to store intermediate results
  - [ ] "drop" image
- [ ] templates
  - [ ] test and validate fitting in downsampled space
  - [ ] profiles of low SNR objects -> asymptotically to psf
- [ ] background options
  - [ ] global background fit
  - [ ] background per stamp
- [ ] diagnostics
  - [x] standard diagnostic view of fit result for object
- [ ] validate output catalogs on MIRI data
  - [ ] color color, color mag
  - [ ] SEDs of stars, photo-z
  - [ ] add in residuals in core for improved flux measurements (shift / psf errors)
- [ ] investigate blending in detection image
- [ ] Investigate template extension methods (Moffat fit and PSF dilation)
- [ ] End-to-end test with realistic mosaic data using `make_mosaic_dataset`
- [ ] Profiling speed + memory usage
- [ ] optimizations
  - [x] adaptive kernel size depending on SNR
  - [x] preconditioning matrix
- [ ] scene size guard and template radius limiting
  - [ ] **Test needed first**: run F1800W without `max_template_radius` to confirm whether `scene_coupling_thresh` alone is sufficient to prevent ginormous scenes, or if radius capping is also required.
  - [ ] **Scene size inspection**: after `generate_scenes` in pipeline, compute per-scene template count and pixel footprint. If any scene exceeds a configurable threshold (e.g. `max_scene_templates: int = None` in FitConfig), halt and print a summary table of scene sizes so the user can decide whether to tighten `scene_coupling_thresh` or enable radius capping.
  - [ ] **`max_template_radius` feature** (implemented but not yet validated — discard from codebase until tested):
    - `FitConfig.max_template_radius: float | None = None` — max footprint radius in arcsec (None = unlimited).
    - In `pipeline.py` before `convolve_templates`: `max_radius_pix = config.max_template_radius / pscale_arcsec` (uses `proj_plane_pixel_scales(wcs[0])[0] * 3600`).
    - In `Templates.convolve_templates(kernel, max_radius_pix=None)`: after each `convolve_cutout`, zero pixels beyond radius — `dist = sqrt((y-yc)^2 + (x-xc)^2); new_tmpl.data[dist > max_radius_pix] = 0.0` where `xc, yc = new_tmpl.input_position_cutout`.
    - Also add `FitConfig.scene_max_merge_radius: float = np.inf` which is already wired in pipeline via `getattr`.
- [x] astrometric shift quality improvements
  - [x] `astrom_isolation_thresh` in FitConfig: sources used for astrometry must contribute >= thresh
        fraction of the flux at their own location (coupling-weighted template overlap). Stars excluded
        via `flag_star` catalog column. `merge_small_scenes` now uses bright non-star count so
        star-dominated scenes are merged into neighbors before solve.
  - [ ] Scene with no bright non-star isolated sources skips astrometry with a warning.
        TODO: consider merging such scenes with a neighbor rather than skipping — isolation filtering
        can't be applied at merge time, so all-blended scenes currently fall through to the guard.
- [ ] strong residuals
  - [ ] handle saturated stars in 444 -> catalog pre pass detection
  - [ ] fit as PSF both 444, 770, fit for centroid, mask center
- [ ]  wavelength dependent morphology: only where residuals are significant.
  - [ ] Add point source, if PSF not given start with marginally sampled Gaussian?  
  - [ ] add second bluer band
- [ ] Mode B aperture correction tests needed: isolated source (corr should equal Mode A result), extended source (corr > Mode A), crowded scene (corr > Yoshi's no-subtraction equivalent)
- [x] Fix pre-existing off-by-one in Mode B scene bounding box: closed by the Mode B aperture
        rework. The aperture path uses half-open `slices_original`/`slices_cutout` directly and
        `Scene.model_image` uses `bb[1]-bb[0]+1`; no inclusive-bbox shape math remains in that path.
- [x] stale tests in `tests/test_fit.py` — closed by retiring the legacy `SparseFitter`
        entirely (see `docs/dead_code.md`). `tests/test_fit.py` and
        `tests/test_sparse_cholesky.py` were deleted; the scene solver is now the only
        fitting path. `test_pipeline.py::test_download_rate` remains network-marked.
- [ ] **the pipeline is not reproducible — scene partitioning varies run-to-run.**
        Established 2026-07-28 by direct experiment: `run_770.py` run twice with *identical*
        code, inputs and config (F1500W, `r_trial=1`, 2890 sources) gives different output
        catalogs. All 22 numeric columns differ. Same 2848 templates every time, but the
        scene grouping — and even the scene *count* — changes:
        `404,627,523,696,598` (5) vs `119,310,776,694,949` (5) vs `297,404,519,714,251,663` (6).
        Scene membership sets which normal-matrix block a source is solved in, hence `err_1`
        moves on ~98% of rows. Magnitude on `flux_1` is unbiased and small in the median
        (0.12% bright / 0.38% faint, ~0.02-0.07 sigma), but ~6% of SNR>10 sources and ~5% of
        SNR>50 sources shift by >1 sigma between identical runs — i.e. quoted errors on bright
        sources understate the run-to-run spread.
        Consequences: output catalogs can never be bit-compared, so A/B regression testing
        against a saved run is inconclusive below ~2% (this defeated the attempt to validate
        the SparseFitter retirement); and published fluxes are not reproducible from the
        same inputs.
        NOTE `random_state=0` IS set on the GP astrometry and there is no `np.random` in
        `src/` — reading the code suggests determinism. That inference is empirically false.
        Suspected mechanism (unverified): `scene_coupling_thresh` is a hard cut on ATA
        couplings, so roundoff-level variation (multithreaded FFT/BLAS reduction order;
        NumExpr reports 16 threads) flips couplings near the cut and cascades into different
        connected components / `merge_small_scenes` outcomes.
        First diagnostic to try: call `generate_scenes` twice inside ONE process on the same
        templates. Identical → cause is cross-process (thread count, `PYTHONHASHSEED`, BLAS
        scheduling) and pinning threads may fix it; different → it is inside the partitioning
        code itself.
- [ ] **tune the flux-block ridge — faint sources are biased low.** `SceneFitter.solve`
        uses `flux_reg = 1e-6 * median(A.diagonal())`: adaptive *per scene*, but the same
        absolute value for every source in that scene. Recovered flux goes as
        `d_i / (d_i + reg)`, so a source whose own `ATA[i,i]` sits far below the scene
        median is suppressed. Measured on a 2-template scene:
        `d_i/median = 1e-3 → −0.05%`, `1e-4 → −0.5%`, `1e-5 → −4.8%`, `1e-6 → −33%`.
        Same bug *class* as the F1800W `reg_astrom` fix (2026-03-27), one level down:
        that one was absolute across the survey, this one is absolute within a scene.
        Extended Estimator-3 templates spread `ΣT²` thin, so 1e-3–1e-4 ratios are
        plausible in real scenes. Likely fix: make the ridge relative per source
        (`reg_i ∝ d_i`) rather than a scene-wide constant.
        NOTE: `FitConfig.reg` was deleted in the SparseFitter retirement (it had zero
        readers — only the legacy solver ever read it). If the ridge becomes tunable,
        reintroduce it deliberately as a live knob rather than restoring a dead field.
        NOTE: the test that guarded this class (`test_regularization_does_not_bias_flux`)
        lived in the deleted `tests/test_fit.py`; recover it from `8192f91` when fixing.
- [ ] restore scene-solver coverage lost with `tests/test_fit.py` (see `docs/dead_code.md`):
        the regularization-bias guard above, the null-shift invariant (joint astrometry
        must not invent shifts on an aligned image), and `Templates.predicted_errors`,
        which is live in production writing the `err_pred_{idx}` catalog column but now
        has no test at all.
- [ ] refactoring for readibility and modularity
  - [ ] split off PSF map / drizzle PSF / PSFs module, make submodule
  - [ ] split off real data as submodule?
  - [ ] other code review, misc refactoring, consolidation
  - [ ] remove unused modules, orphan code
