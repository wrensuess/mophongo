# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Package Management
- **Install dependencies**: `poetry install`
- **Alternative installation**: `pip install -e .`
- **Run tests**: `pytest` (standard pytest runner)
- **Individual tests**: `pytest tests/test_<module>.py` or `pytest tests/test_<module>.py::test_function`

### Project Structure
This is a Python package for astronomical photometry processing, organized as:
- `src/mophongo/` - Main package source code
- `tests/` - Test suite (pytest-based)
- `examples/` - Jupyter notebooks demonstrating usage
- `data/` - Test data and PSF files for astronomical processing
- `legacy/` - Legacy IDL/Pro code for reference

## Architecture Overview

### Core Pipeline Flow
The photometry pipeline follows this sequence:
1. **Template Extraction** (`templates.py`) - Extract source templates from high-resolution detection image
2. **PSF Handling** (`psf.py`, `psf_map.py`) - Manage point spread functions and spatially-varying PSF maps
3. **Convolution/Matching** - Match PSFs between images using kernels
4. **Scene Solving** (`scene.py`, `scene_fitter.py`) - Partition sources into
   independent scenes and solve each for fluxes (+ joint astrometry) via sparse
   normal equations. `fit.py` holds only the `FitConfig` dataclass.
5. **Astrometric Correction** (`astrometry.py`) - Shift-field models applied to templates

### Key Classes and Components

**Templates System** (`templates.py`):
- `Template` - Individual source template (extends `astropy.nddata.Cutout2D`)
- `Templates` - Collection manager with extraction, convolution, and downsampling

**Fitting Framework**:
- `FitConfig` (`fit.py`) - Configuration dataclass controlling fitting behavior.
  This is all `fit.py` contains; the legacy `SparseFitter`/`GlobalAstroFitter`
  solvers were retired (see `docs/dead_code.md`).
- `Scene` (`scene.py`) - A group of templates coupled through overlap, solved
  as one independent block; `generate_scenes()` partitions the field.
- `SceneFitter` (`scene_fitter.py`) - Stateless solver for a single scene's
  joint flux + astrometry system. This is the only solver.

**PSF Management** (`psf.py`, `psf_map.py`):
- `PSF` - Point spread function with analytic and array-based creation
- `PSFRegionMap` - Spatially-varying PSF lookup system
- `DrizzlePSF` - JWST/HST-specific PSF handling with coordinate mapping

**Pipeline Orchestrator** (`pipeline.py`):
- `run()` - Main entry point taking images, segmap, catalog, PSFs
- Handles multi-resolution processing and template matching
- Memory-efficient processing with progress tracking

### Important Implementation Details

**Multi-Resolution Support**:
- Automatic binning factor detection from WCS (`utils.bin_factor_from_wcs`)
- Template and PSF downsampling for resolution matching
- Upsampling of lower-resolution images when needed

**Sparse Matrix Optimization**:
- Per-scene normal equations, Jacobi-whitened and solved with `spsolve`
- Positivity constraint via `config.positivity`
- Regularization is internal, not configurable: the flux block gets an adaptive
  ridge (`1e-6 * median(diag A)`) and the shift block `config.reg_astrom`

**Astrometric Refinement**:
- Iterative astrometry fitting with configurable passes (`fit_astrometry_niter`);
  `0` disables astrometry entirely. This is the only astrometry on/off knob —
  the old `fit_astrometry_joint` flag was removed, since the "separate" mode it
  named lived in the retired legacy solver.
- SNR-based source selection for astrometry

**Memory Management**:
- Template pruning based on weight maps
- In-place operations where possible
- Memory usage tracking throughout pipeline

### Testing Patterns
- Tests use `utils.make_simple_data()` for synthetic datasets
- Visual diagnostics saved to tmp directories for debugging
- Pytest with module-level skipping for data-dependent tests
- Integration tests with full pipeline workflows

### Data Handling
- FITS-based I/O with Astropy integration
- Segmentation maps for source identification
- Coordinate transformations via WCS
- Support for both synthetic and real astronomical data

## Development Notes

**Template System**: Templates maintain both original and cutout coordinate systems with careful slice bookkeeping for sparse matrix construction.

**PSF Matching**: Kernel computation uses Fourier-domain matching with Tukey windowing for stable deconvolution.

**Flux Errors**: `SceneFitter._flux_errors` takes `1/sqrt(diag)` of the *whitened*
normal matrix, so errors do not reflect off-diagonal covariance between blended
sources. The covariance-based mode (`config.fit_covariances`) was never
implemented on the scene solver and the field was removed.

**Multi-Component Fitting**: Not implemented on the scene solver. The
`multi_tmpl_*` config fields and their helper (`_add_templates_for_bad_fits`)
were removed as dead code; see `docs/dead_code.md`.