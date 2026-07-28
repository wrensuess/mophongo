# Test-suite cleanup plan

> **⚠️ SUPERSEDED (2026-07-24) — historical record, do not act on this document.**
>
> The cleanup it describes was carried out, and the legacy `SparseFitter` was
> subsequently retired entirely (`docs/dead_code.md`). Anything below that
> discusses `SparseFitter`, `GlobalAstroFitter`, `solve_method`,
> `build_normal_matrix`, or `tests/test_fit.py` refers to code that **no longer
> exists** — those decisions were resolved by deletion, not by the fixes
> proposed here. Sections A6/B3 in particular are moot.
>
> Kept because it records *why* several surviving `xfail`s exist and what the
> suite is meant to protect. For current state see `docs/dead_code.md`.

Status: **guide only — no code changed yet.** Produced after reading the current
source (`src/mophongo/`) and every file in `tests/`, and running the suite.

## What this suite is actually protecting

The scientific output of mophongo is: **accurate fit fluxes, good astrometry, and
small residuals.** Everything else — scene splitting, PSF matching, template
construction, downsampling — exists to make those three things correct. Aperture
corrections are a small post-processing step at the very end; they matter, but
they are the tail, not the body.

The tests should be weighted and ordered accordingly. This document organizes the
target suite **sequentially, in the order the pipeline actually runs**, and tags
each stage by how critical it is to the three outputs above:

- **CRITICAL** — directly measures flux accuracy, astrometry, or residual size.
- **CORE** — machinery whose correctness those outputs depend on.
- **SUPPORTING** — helpers and post-processing; needed, but low blast radius.

## Snapshot of the current state

Collection is clean: **144 tests collected, 0 import errors.** Running everything
except the three known time-sinks (`test_benchmark.py`, `test_jwst_psf.py`, and
the live-network `test_pipeline.py::test_download_rate`):

```
30 failed, 105 passed, 5 skipped, 5 deselected
```

The 30 failures fall into four buckets:

| Bucket | Meaning | Count |
|---|---|---|
| **B1 — real source regression the test correctly catches** | the pipeline entry point is broken | 6 |
| **B2 — stale test calling a renamed/removed API** | source moved, test didn't | ~16 |
| **B3 — pre-broken by an earlier design change** | `astrom_basis_order` removed from `FitConfig` | 4 |
| **B4 — data-dependent / dead** | missing FITS, `return` at top of test | ~4 |

The most important thing the failure list tells us: **the highest-value test in
the whole suite is currently red.** `test_pipeline_flux_recovery` — the one that
asks "did we get the fluxes back?" — fails because the pipeline entry point
itself is broken (B1). That is the opposite of where a healthy suite's failures
should be.

Conversely, the aperture-correction suite (`test_pipeline_aperture.py`, 1476
lines) is the *healthiest* part of the suite — current, thorough, and green — but
it's also the least critical. Effort has pooled at the wrong end.

---

# Part A — What the suite should contain, in pipeline order

Each stage below corresponds to a step in `Pipeline.run`. ✅ marks coverage that
already exists and should be kept; **(gap)** marks something missing or
inadequate.

## A0. Cross-cutting numerical helpers — `utils.py`  · CORE · ❌ no tests today

70 KB of helpers underlie every stage below, with **no `test_utils.py`**.

- **(gap) EE round-trip:** `psf_ee_at_radius(psf, psf_ee_radius_pix(psf, f)) ≈ f`.
  *Why:* both use true-total (not stamp-sum) normalization and must stay mutually
  consistent — everything downstream that talks about enclosed energy depends on
  it. *How:* Moffat/Gaussian PSF, sweep `f ∈ {0.5,0.8,0.95}`.
- **(gap) `bin_factor_from_wcs`:** exact integer for 2×/4× WCS, raises on 1.5×.
  *Why:* silent mis-binning corrupts all multi-resolution flux. *How:* build WCS
  pairs at known scale ratios. (Docstring says tol 0.02 but code uses 0.001 —
  pin it.)
- **(gap) `psf_stamp_containment` ∈ (0,1]** for a normalized stamp (disk, not box).
- *Latent bug to guard or delete:* `rebin_wcs` references an undefined `n`
  (`factor = 2**n`) → `NameError` if called.

## A1. Detection & catalog — `catalog.py`  · SUPPORTING

Inputs to the fit: segmentation + measured catalog columns.

- **(gap/fix) `Catalog.run` on synthetic data** produces finite `segment_flux`,
  `kron_flux`, `kron_radius`, `r50`, `sharpness`, `snr`, with
  `snr == segment_flux/segment_fluxerr`. *Why:* these feed template sizing and the
  aperture stage. Today `test_catalog` fails reading an external FITS
  (`OSError: No SIMPLE card`) and its companion `test_deblend_sources` is a no-op
  (`return` on line 1). *How:* rebuild both on `make_simple_data`.
- ✅ Deblenders (`test_deblender.py`, `test_photutils_deblend.py`) — small, real,
  passing. Keep.

## A2. Template extraction — `templates.py`  · CORE · ✅ strong

Templates are the model basis; if they're wrong, every flux is wrong.

- ✅ Unit-sum after extraction; `template_norm × H == composite`; positivity.
- ✅ EE-radius vs analytic Gaussian; ownership disjointness; halo-blend weight
  monotonicity; NaN handling; `apcor_from_psf` SNR gate
  (`test_template_extension.py`, `test_templates.py` — high quality).

## A3. PSF matching & kernels — `psf.py`, `psf_map.py`  · CORE

The kernel is what makes a hi-res template predict a lo-res image; a bad kernel
biases flux directly.

- ✅ **Matching-kernel flux conservation + target-PSF reproduction**
  (`test_psf.py`) — this is the load-bearing one. Keep.
- ✅ Moffat/Gaussian normalization, effective-PSF, PSF basis, aperture profile.
- ✅ `PSFRegionMap`: region count, containment round-trip, missing-column default,
  disk-fraction containment (`test_psf_map.py` — good, just calls the renamed
  `lookup_key`).
- **(gap) resolve_key / containment share one region:** assert the PSF and its
  containment resolve from the same region, and NaN/None ra,dec → region 0.

## A4. Convolution & downsampling — `templates.py`  · CORE

- ✅ `template_norm` unchanged by `convolve_cutout` even when `kernel.sum() ≠ 1`
  (flux conservation under convolution).
- **(gap) downsample flux conservation** across k=2,3,4. *Why:* multi-resolution
  is a core feature; `test_downsample.py` was meant to cover this but targets a
  `bin2d_mean` that doesn't exist, so it's dead. *How:* test the real API
  (`Template.downsample` / `downsample_psf`), assert summed flux conserved.

## A5. Scene generation — `scene.py`  · CORE · ➖ only indirect coverage

Scenes decompose the global solve into blocks. The critical property is that the
decomposition **doesn't change the answer**.

- **(gap) `generate_scenes` partitioning:** overlapping templates land in one
  scene, isolated ones don't; a star (`is_star`) is excluded from the
  bright/astrometry set. *Why:* wrong partitioning silently biases crowded fluxes.
  There is no `test_scene.py` today. *How:* a handful of placed templates with
  known overlaps.
- ✅ (covered indirectly in A6 via scene-vs-global equivalence).

## A6. Sparse fitting — flux solve — `fit.py`, `scene_fitter.py`  · **CRITICAL** · ⚠️ intent good, all red

**This is the heart of the suite.** "Did we recover the true fluxes?" The unit
tests here have the right intent but every one calls a removed method.

- ✅(after repair) **Flux recovery on a known system** — fitted flux ≈ analytic /
  injected value.
- ✅(after repair) **Scene-solve == global-solve equivalence** — decomposition
  invariance (mirrored in `test_fit.py` and `test_scene_fitter.py`). *Keep both;
  this is exactly the property A5 relies on.*
- ✅(after repair) `build_normal_tree` vs loop equivalence; zero-weight template
  dropped; bright-source mask.
- **(gap) Regularization does not bias flux:** isolated source, fitted flux ≈
  `b/d` independent of `flux_reg`. *Why:* commit `9d2ed2d` fixed exactly this bias
  and nothing guards it. 
- **(gap/decision) Flux-error orientation:** `SparseFitter._flux_errors` returns
  `sqrt(diag)·covar_power` (with an unreachable `1/sqrt(diag)` below), while
  `SceneFitter._flux_errors` returns `1/sqrt(diag)` — inverse answers on the same
  system. Pin a single isolated source's error against analytic `1/sqrt(d)` to
  force the question of which is correct.
- **(gap) `solve_method` guard:** `solve()` dispatches `else → self.solve_all`,
  which isn't defined — any non-scene method raises `AttributeError`. One line to
  assert supported methods solve and unsupported ones fail cleanly.

## A7. Astrometry refinement — `astrometry.py`, `astro_fit.py`  · **CRITICAL** · ⚠️ API drift + one dead file

- ✅(after repair) **Recovers a known injected shift**; polynomial/GP astrometry
  reduces residual; shift-uncertainty scales with PSF width
  (`test_astrometry.py` — well designed, failing only on
  `astrometry.py:459` API drift).
- **(gap) Astrometry doesn't bias flux:** run with astrometry on vs off on a clean
  scene, assert recovered fluxes agree. *Why:* the joint flux+shift solve is where
  flux bias historically crept in.
- **Decision needed — `test_astro_fit.py` (`GlobalAstroFitter`):** fails on the
  removed `FitConfig(astrom_basis_order=...)` param, and one "test" has **no
  assertions** (prints + saves a PNG, references stale `fitter.alpha/beta`).
  Either `GlobalAstroFitter` is still a supported path (re-pin the config API, add
  real asserts) or it's superseded by the scene solver (delete the file). Your
  call — it's architecture, not a mechanical fix.

## A8. Residuals — `Pipeline.run` end-to-end  · **CRITICAL** · ❌ not asserted anywhere

The third headline output, and **nothing currently asserts it.**

- **(gap) Residual is at the noise floor after a fit.** *Why:* small structured
  residuals at source positions are the canonical symptom of a flux/PSF/kernel
  bug; this is the most sensitive single end-to-end check we can have. *How:* full
  `Pipeline.run` on `make_simple_data`; assert per-pixel χ² ≈ 1 over the image and
  no significant flux left inside source segments (residual segment sums consistent
  with noise). `test_pipeline_flux_recovery` already builds everything needed — it
  just plots instead of asserting.
- **(gap) End-to-end flux recovery, as an assertion.** Same run: recovered vs true
  flux within tolerance, across hi-res and lo-res bands and across the crowding
  range in the synthetic field. This is the suite's keystone test and today it is
  both broken (B1) and assertion-light.

## A9. Flux bookkeeping & aperture corrections — `pipeline._add_aperture_photometry`  · SUPPORTING · ✅ healthy

Post-processing. Keep the existing `test_pipeline_aperture.py` — it's current and
thorough — but it is the tail of the suite, not its center. Existing coverage
(all ✅): aperture-sum linearity, estimator definitions, catalog-tie algebra,
truncation cancellation, faint-limit identities, Stage-4c crowding flatness, Kron
construction vs photutils, native-pixel-scale band EE.

- **(gap, cheap) Correction-magnitude sanity guard:** point-source
  `tcor_int ≈ 1/EE(r_ap)` and all correction products O(1)–O(2), never ~30×. One
  line that would catch the historical failure class wholesale.
- **(gap) Side-effect containment:** `_build_f444w_residual` writes
  `f444w_template_residual.fits` into CWD (stray copies already sit in the repo
  root). Tests should `chdir(tmp_path)`; ideally the diagnostic becomes opt-in.

## What should NOT be in the suite

- **No live-network tests** — `test_download_rate` exercises MAST/astroquery, not
  mophongo, and hangs the run.
- **No unconditional downloads in the default run** — JWST/webbpsf/S3 fetches go
  behind an opt-in marker.
- **No timing/benchmark code mixed with correctness tests** — benchmarks live in a
  separate, deselected, marked file.
- **No assertion-free "diagnostic" tests** — they pass even when the science is
  wrong. Plot helpers are fine; a *test* must assert.

---

# Part B — Changes needed to make the suite match Part A

Nothing here is applied yet. Ordered by leverage on the three headline outputs.

## B1. Un-break the keystone: point the pipeline tests at the real entry point (tests-only)

Six integration tests — including the keystone flux-recovery test — call the
module-level `pipeline.run()` wrapper, which is **dead and broken**: it forwards
`wht_images=` into `Pipeline.__init__()`, which only accepts `weights=` →
`TypeError` on every call (`src/mophongo/pipeline.py:1938`). Nothing in `src/`
uses this wrapper; production and the aperture tests all construct
`Pipeline(...).run()` directly. `Pipeline.run()` returns a **2-tuple**
`(table, residuals)` and never sets `self.fitter` (`pipeline.py:1753`), but these
stale tests unpack a 3-tuple and read a `fitter` that doesn't exist.

Breaks: `test_pipeline_flux_recovery`, `test_pipeline_deduplicates_templates`,
`test_pipeline_multitemplate_pass`, `test_pipeline_prunes_templates_with_zero_weight`,
`test_plot_result`, `test_pipeline_class_attributes`.

**Fix (tests only — no source change):** rewrite each of the six to construct
`Pipeline(images, segmap, ...).run()` directly and unpack the real 2-tuple
`(table, residuals)`. Drop the `fitter` unpack/assertions (the fitter is not
exposed on the pipeline). In `test_pipeline_class_attributes`, also replace
`assert pl.residuals == residuals` (raises on array comparison) with an identity
check. The broken module wrapper is left untouched — the tests simply stop using
it, matching how the code is actually run. These tests are legitimate and are the
keystone (they unblock A6/A8) — keep them, just repaired.

## B2. Repair stale tests calling renamed/removed APIs (mechanical)

| Test file | Broken call | Current API | Tests affected |
|---|---|---|---|
| `test_fit.py` | `fitter.build_normal_matrix()` | method is now `build_normal()` / `build_normal_tree()`; `build_normal_matrix` is a module-level free function (`fit.py:1643`), not a method | 7 (`test_flux_recovery`, `test_lsqr_lo_matches_cg`, `test_ata_symmetry`, `test_zero_weight_template_dropped`, `test_flux_errors_regularized`, `test_build_normal_tree_matches_loop`, `test_bright_source_detection`) + `build_normal_matrix_new` ref |
| `test_psf_map.py` | `regmap.lookup_key(...)` | `resolve_key(...)` (`psf_map.py:583`) | `test_lookup`, `test_pa_coarsening` |
| `test_scene_fitter.py` | `SceneFitter()` / `Scene(...)` ctor | constructor signature changed (`SimpleNamespace` unpack error) | `test_scene_fitter_flux_only`, `_with_shift_block`, `test_solve_flux_and_shifts_matches_dense`, `test_scene_graph_and_residuals` |
| `test_scene_plot.py` | `Scene.plot()` missing arg | `plot()` now requires an argument | `test_scene_plot` |
| `test_astrometry.py` | `astrometry.py:459` attr + `SimpleNamespace` unpack | astrometry return/attr changed | `test_polynomial_astrometry_reduces_residual`, `test_gp_astrometry_returns_models`, `test_astromap_recovers_shift`, `test_apply_template_shifts_uses_shift_field` |

**Keep-and-repair** — the intent (flux equivalence, astrometry recovery) is
exactly the CORE/CRITICAL coverage from Part A; only the call sites are stale.

## B3. Delete `test_astro_fit.py` — `GlobalAstroFitter` is a dead path

Confirmed: `GlobalAstroFitter` is imported in `pipeline.py:1312` but **never
instantiated**; production astrometry uses `AstroCorrect` + scene shifts. The file
tests a superseded solver, fails on the removed `FitConfig(astrom_basis_order=...)`
parameter, and includes an assertion-free "test". **Delete the file.**

## B4. Delete / repair dead and inappropriate tests

- **Delete** `test_pipeline.py::test_download_rate` — live MAST call, tests
  astroquery not mophongo, hangs the suite. (The "pointless slow download test.")
- **Delete** `test_downsample.py` — targets non-existent `bin2d_mean`, never runs;
  replace with the real downsample flux-conservation test (A4).
- **Delete** `test_example.py::test_addition` (`assert 1+1==2`) + its `__main__`
  scratch.
- **Repair** `test_catalog.py::test_deblend_sources` — `return` on line 1 makes it
  a no-op; rebuild on `make_simple_data` (A1).
- **Clean up** `tests/utils.py`: `make_testdata` (hardcoded `/Users/ivo/...`) and
  `check_project` (`from astropy.nddate import ...` — an import that can't succeed)
  are personal scratch, not shared helpers.
- **Gitignore + redirect** the stray FITS the suite/pipeline drop in the repo root
  (`f444w_template_residual.fits`, `img_*_tile.fits`, `img_uncompressed.fits`).

## B5. Add marker infrastructure so the default run is fast and green

There is no `conftest.py` and no marker config today; slow/external handling is
ad-hoc (`pytest.skip`, `skipif(1, ...)`, data-conditional skips, one unguarded
network test).

- Register markers in `pyproject.toml`/`pytest.ini`: `slow`, `network`,
  `needs_data`, `benchmark`.
- Mark `test_benchmark.py` `benchmark`; the download/external tests `network` /
  `needs_data`; default `addopts` to deselect them.
- Replace the `skipif(1, ...)` idioms with real markers so `pytest -m slow` can
  actually run them on demand instead of them being permanently dead
  (`test_realistic_pipeline`, `test_psf::test_drizzle_psf`,
  `test_psf_map::test_psf_region_map_from_file`).

## B6. Add the missing coverage (Part A gaps), in priority order

1. **A8 — residual-at-noise-floor + asserted end-to-end flux recovery.** The
   single highest-value addition; the keystone of the three headline outputs.
2. **A6 — regularization-no-bias, flux-error orientation, `solve_method` guard.**
3. **A7 — astrometry-doesn't-bias-flux.**
4. **A0 — `test_utils.py`** (EE round-trip, `bin_factor_from_wcs`, containment).
5. **A5 — `test_scene.py`** partitioning + star exclusion.
6. **A4 — template downsample flux conservation** (replacing `test_downsample.py`).
7. **A3 — resolve_key/containment shared-region.**
8. **A9 — aperture correction-magnitude sanity guard + side-effect containment.**

## Recommended file organization (sequential, pipeline order)

Reorder the files so the suite reads top-to-bottom in execution order. A numeric
prefix makes the order explicit and groups the stage's unit + integration tests:

| Prefix | File | Stage | Priority |
|---|---|---|---|
| `test_00_utils.py` | *(new)* | numerical helpers | CORE |
| `test_10_catalog.py` | ← `test_catalog.py`, `test_deblender.py`, `test_photutils_deblend.py` | detection/catalog | SUPPORTING |
| `test_20_templates.py` | ← `test_templates.py`, `test_template_extension.py` | extraction | CORE |
| `test_30_psf.py` | ← `test_psf.py`, `test_basis.py`, `test_aperture_profile.py`, `test_psf_map.py` | PSF & kernels | CORE |
| `test_40_convolve_downsample.py` | ← new downsample test | convolution/downsample | CORE |
| `test_50_scene.py` | *(new)* + `test_scene_fitter.py` | scene split | CORE |
| `test_60_fit.py` | ← `test_fit.py`, `test_sparse_cholesky.py` | **flux solve** | **CRITICAL** |
| `test_70_astrometry.py` | ← `test_astrometry.py` (+ `test_astro_fit.py` pending B3) | astrometry | **CRITICAL** |
| `test_80_pipeline.py` | ← `test_pipeline*.py` (flux recovery + **residuals**) | end-to-end | **CRITICAL** |
| `test_90_aperture.py` | ← `test_pipeline_aperture.py` | corrections | SUPPORTING |
| `test_99_benchmark.py` | ← `test_benchmark.py` (marked) | timing | — |

Renaming is optional cosmetics; the point is that the **critical mass of the
suite should sit in stages 60–80 (flux, astrometry, residuals)**, which is exactly
where the failures and gaps are concentrated today.

## Suggested sequencing of the work

1. **B1** — un-break the pipeline entry point; unblocks the keystone flux/residual
   tests. Highest priority.
2. **B2 + B4** — mechanical rename repairs + deletions; turns most of the 30 red
   back to green with no design decisions.
3. **B6 #1–#3** — add the CRITICAL gaps (residuals, flux-bias guards) while that
   code is fresh.
4. **B3 / A7** — your call on `GlobalAstroFitter`.
5. **B5** — marker infrastructure; make the default run fast.
6. **B6 #4–#8** + the sequential file reorg.

After B1–B5 the default `pytest` run should be fully green and fast. B6 then puts
real assertions behind the three outputs that actually define whether mophongo
works.
