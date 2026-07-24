# Dead code inventory & removal workflow

Confirmed-dead, broken-and-unreachable, or redundant code surfaced during the
2026-07 dead-code audit (test-suite cleanup + a three-way static/duplication/
garden-path sweep). Each entry notes **where** it lives, **why** it's dead, and
the **cleanup**.

Entries are grouped into **removal tiers** (see the workflow at the bottom):

- **Tier A — safe deletes.** Zero live callers, off the critical/production
  path. Delete freely; only a parked `xfail` test (if any) needs to go with it.
- **Tier B — decide first.** Redundant, but reachable via a config flag or a
  passing test. Needs a one-line decision (keep which copy?) before removal.
- **Tier C — critical-path duplication.** Do **not** touch without sign-off;
  documented here as drift risk, not as a delete.
- **Test-only modules.** Not in production, but a test still imports them.
- **Not dead — flagged.** Stale docs / deferred features.

The critical/production path is: `Pipeline.run()` → scene solver
(`generate_scenes`, `SceneFitter`, `Scene` in `scene.py`/`scene_fitter.py`) →
`Templates`, `PSF`/`PSFRegionMap`, `Catalog`, `AstroCorrect`. `SparseFitter`
(`fit.py`) is the *legacy* solver path — still imported and tested, still
reachable when `config.run_scene_solver=False`, so treat it as live-but-legacy.

Tests that would exercise broken code are parked as `xfail` so the defect stays
visible without breaking the suite; fixing/removing the code lets those flip to
real tests (or be deleted alongside).

---

## Status — 2026-07-24

**Tier A round 1 executed** (branch `apcor-estimator3`, uncommitted). Each
deletion ran the test suite (`pytest`, 146 passed / 2 skipped / 7 xfailed, 25 s)
green before continuing:

| File | Removed | Lines |
|---|---|---|
| `astro_fit.py` | whole module + its import in `pipeline.py` | −256 |
| `fit.py` | OBSOLETE banner + `assemble_scene_system_old`/`build_scene_tree_old`/`merge_small_scenes_old`/`solve_scene_shifts` (kept `build_normal_matrix`, it's tested) | −329 |
| `psf.py` | `NEffectivePSF`, shadowed `to_header`/`get_slice_wcs`, commented interp block | −191 |
| `templates.py` | `downsample_wcs_old`, `block_aligned` | −76 |
| `catalog.py` | first `_mean_downsample`, first `_expand_remap`, `_sigma_clip`, `calibrate_ivar_with_bg_median` | −135 |
| `astrometry.py` | `measure_template_shifts_old` | −62 |
| `pipeline.py` | commented Zarr block | −38 |
| `CLAUDE.md` | stale `local_astrometry.py` → `astrometry.py` pointer | — |

**Held out of round 1** (entangled with parked xfail tests or a behavior change —
need a decision): `utils.rebin_wcs` (+ its xfail), `SparseFitter.solve_lo`/
`solve_all`/`bright_mask` (+ their xfails), the `_flux_errors` 1293→1294 swap
(a real behavior fix on the legacy path), and `write_wcs_csv`'s `continue`
short-circuit (behavior change; also see its ride-along helpers below).

**Tier A round 2 executed** (same branch, tests green after each file). Deleted
the rescan's zero-caller symbols across `utils.py`, `psf.py`, `psf_map.py`,
`scene.py`, `templates.py`, `catalog.py`, `sim_data.py`, `photutils_deblend.py`,
`astrometry.py` — **kept** `PSF.fit_moffat`/`fit_gaussian` (useful public API) and
`Scene.augment_templates` (stub). Cluster 1 (PSF/basis prototypes) archived in git
(revival pointers in the *Round 2* subsection). See that subsection for the full
resolved list.

**Combined Tier A (rounds 1+2): ~1,833 lines removed**, all 146 tests green,
all modules import clean. Nothing committed yet.

---

## Tier A — safe deletes (zero callers, off critical path)

### ✅ Round 1 — done (see Status table above)

### fit.py — the entire "OBSOLETE BELOW" tail (~400 lines)

`fit.py:1313-1314` carries the author's own banner
`# ---- OBSOLETE BELOW ----` (twice). Everything under it is dead:

- **`assemble_scene_system_old`** (`fit.py:1317`) — no callers.
- **`build_scene_tree_old`** (`fit.py:1451`) — no callers.
- **`merge_small_scenes_old`** (`fit.py:1495`) — no callers.
- **`solve_scene_shifts`** (`fit.py:1559`) — defined *after* a `return` inside
  `merge_small_scenes_old`; a free function, not a method, so unreachable as
  `fitter.solve_scene_shifts(...)`. (Previously logged.)
- **`build_normal_matrix`** (`fit.py:1643`) — a free function taking `self`,
  dedented to module scope. `SparseFitter.build_normal` dispatches to
  `self.build_normal_matrix()` when `config.normal == "loop"`, but nothing binds
  it to the class → `AttributeError` on that path. Default `config.normal="tree"`
  hides the bug. Only reachable via tests calling `build_normal_matrix(fitter)`
  directly. Cleanup: delete, or re-attach as a real method **and** fix the
  `"loop"` dispatch if the O(n²) loop builder is still wanted for small-N debug.

Verified: `grep` finds no live call sites for any of these (only a comment in
`test_fit.py:292` names `merge_small_scenes_old`).

### fit.py — `SparseFitter._flux_errors` dead branch (`fit.py:1293-1294`)

`return np.sqrt(diag) * covar_power` immediately followed by an unreachable
`return 1/np.sqrt(diag)`. The first collapses to `0.0` for isolated sources; the
second (below it) is the physically-correct line. Off the default path — the
scene solver (`SceneFitter._flux_errors`) supplies the reported `err_N` columns.
**Note:** the surviving line may itself be a live bug, not just dead code — flag
to the user before choosing which `return` to keep.
Parked test: `test_fit.py::test_flux_error_orientation_matches_analytic`.

### fit.py — referenced-but-undefined solve methods

- **`SparseFitter.solve_lo`** — `solve()`'s docstring lists a `'lo'` method that
  doesn't exist. Drop `'lo'` from the options, or implement it.
  Parked test: `test_fit.py::test_lsqr_lo_matches_cg`.
- **`SparseFitter.solve_all`** — `solve()` dispatches `else -> self.solve_all()`
  for any `solve_method != "scene"`, so `"all"`/`"lo"` raise `AttributeError`.
  Parked test: `test_fit.py::test_solve_method_all_not_supported`.
- **`SparseFitter.bright_mask`** — computed SNR but the assignment is commented
  out (~`fit.py:751`); the only other reference is inside dead code.
  Parked test: `test_fit.py::test_bright_source_detection`.

### templates.py — versioned leftovers

- **`Template.downsample_wcs_old`** (`templates.py:619`, ~39 lines) — superseded
  by `Template.downsample` (`:696`); zero callers; still holds leftover debug
  `print()` lines (647-651). Delete.
- **`Template.block_aligned`** (`templates.py:661`) — preceded by the author's
  own `# block alignment methods currently not used` (`:659`); zero callers.
  Delete.

### psf.py — `class NEffectivePSF` (`psf.py:1265-1421`, 156 lines)

Never instantiated anywhere in `src/`, `tests/`, or `examples/`; the only
reference is a commented line `psf.py:835`. An abandoned rewrite of
`EffectivePSF` (`:506`) whose remote-download path was never finished. Delete the
whole class (confirm it isn't a mid-migration keep).

### psf.py — shadowed WCS-helper copies

`psf.py` imports `to_header` and `get_slice_wcs` from `.utils`, then **redefines
both** later in the same module with byte-identical bodies
(`psf.py:751` `to_header`, `:774` `get_slice_wcs`), silently shadowing the
import. Delete the psf.py copies; keep the imports.
(`DrizzlePSF.read_wcs_csv` at `psf.py:858` has *diverged* from `utils.read_wcs_csv`
at `utils.py:1230` — it handles masked cells and returns an extra dict; reconcile
into one implementation rather than plain-deleting — this half is Tier B.)

### psf.py — commented-out interpolation block (`psf.py:669-686`, 18 lines)

A prior nearest-grid-point bilinear interpolation, commented out, immediately
followed by its live "grid-agnostic robust interpolation" replacement. Delete.

### astrometry.py — `measure_template_shifts_old` (`astrometry.py:178`)

Superseded by `measure_template_shifts` (`:117`, the one actually called at
`:339`); zero callers. Delete.

### catalog.py — doubled/dead helpers

- **`_mean_downsample`** defined **twice** (`catalog.py:80` and `:383`); Python
  keeps the later, so `:80-88` is permanently unreachable. Delete the first.
  (Both reimplement `astropy.nddata.block_reduce(arr, fact, func=np.mean)`, which
  the file already imports and uses at `:273`, `:505` — see Tier B.)
- **`_expand_remap`** defined **twice**, byte-identical (`catalog.py:457` and
  `:472`), with a stray duplicate import block wedged between them (numpy,
  block_reduce, Table, mad_std, detect_sources, SourceCatalog, minimum_filter —
  all already imported at file top). Delete the first def + the stray imports.
- **`_sigma_clip`** (`catalog.py:392`) — never called; reimplements
  `astropy.stats.sigma_clip` (the file already uses `mad_std` from
  `astropy.stats`). Delete.
- **`calibrate_ivar_with_bg_median`** (`catalog.py:233`) — never called (only a
  commented line at `:708`), **and** broken: line 283 reads `bgmask` before it's
  assigned (`:319`) → `UnboundLocalError` if ever called, and that result is
  overwritten at `:324` anyway. Delete (or salvage its median-detrending into the
  live `get_bg_and_ivar` at `:136` if that feature is wanted).

### utils.py — `rebin_wcs` (`utils.py:262`)

Body does `factor = 2**n` referencing an undefined `n` → `NameError` on any call.
Never called in `src/`. A correct equivalent already exists as
`templates.scale_wcs_pixel` (`templates.py:115`, actively used). Delete, or make
it a `2**n` wrapper around `scale_wcs_pixel`.
Parked test: `test_utils.py::test_rebin_wcs_is_broken` (xfail).

### utils.py — `write_wcs_csv` short-circuit (`utils.py:2157`)

An unconditional `continue` at the top of the loop body (right after the
`print(...)`) makes the function skip every iteration and always write an **empty**
CSV. Classic "disabled to test something" leftover. Only caller is
`examples/uds.ipynb`. Remove the `continue` (or confirm it was deliberate).

### pipeline.py — commented-out Zarr block (`pipeline.py:1945-1983`, ~37 lines)

Module-level `run()` wrapper does `return pipeline.run()` followed by a large
commented-out Zarr chunked-storage experiment (self-labeled `# EXTREMELY SLOW`).
Unreachable *and* commented; `zarr` isn't imported anywhere. Delete.

### astro_fit.py — whole module dead (256 lines)

`GlobalAstroFitter` is imported once (`pipeline.py:1312`) but never instantiated;
production astrometry uses `AstroCorrect` + scene shifts. The module's test
(`test_astro_fit.py`) is already deleted. Remove the class/module and the stale
import.

### Round 2 — rescan additions (resolved 2026-07-24: delete all except the two kept below)

A static-unused rescan of the modules the first sweep didn't finish surfaced
these zero-caller symbols. All verified with `grep -rn` across
`src/ tests/ examples/` as definition-only (or, where noted, the only other hit
is a comment). **None are wired to any config path** — confirmed fully orphaned,
not default-off-but-reachable. All ~1 year old (July 2025 commits).

**Kept live (not deleted):**
- `psf.py` `PSF.fit_moffat` / `PSF.fit_gaussian` — thin public wrappers that
  delegate to the live `PSF._fit_profile`; the natural interactive API for
  fitting a profile to a PSF. Zero callers, but cheap and useful to keep.
- `scene.py` `Scene.augment_templates` — intentional `return None` placeholder in
  the scene-graph-stub family; a feature to build, not dead code.

**Cluster 1 — PSF/kernel/basis research prototypes: deleted from the tree,
archived in git.** These were exploratory alternative bases; the pipeline settled
on `cheb_basis` (scene solver) and the live `regularized_lstsq_kernel`/
`matching_kernel` path (PSF matching), so none are reachable. They carry
conditional future value (revisit if the Chebyshev color basis is too stiff or
PSF matching needs different regularization), and git preserves them intact.
**All exist at commit `9b47342` (HEAD before the deletion commit); recover any via
`git show 9b47342:src/mophongo/<file>` or the origin commit below.**

| Symbol | File | Origin commit |
|---|---|---|
| `regularized_lstsq_kernel_central` | `utils.py` | `52e9aa7` psf generation update |
| `regularized_pixel_kernel_central` | `utils.py` | `52e9aa7` psf generation update |
| `positive_monotone_radial_bspline` | `utils.py` | `52e9aa7` psf generation update |
| `zernike_basis` | `utils.py` | `b07cd1b` Revert PSF map defaults / global fwhm |
| `starlet_basis` | `utils.py` | `b07cd1b` Revert PSF map defaults / global fwhm |
| `eigen_psf_basis` | `utils.py` | `b07cd1b` Revert PSF map defaults / global fwhm |
| `psf_matching_kernel_basis` | `psf.py` | `d43dcb4` Add Fourier basis PSF matching |

**Deleted — obsolete-by-construction / superseded (no future value):**

| File | Symbol | Note |
|---|---|---|
| `astrometry.py` | `make_gradients`, `basis_matrix` | built for the now-deleted `GlobalAstroFitter` (keep `n_terms`/`cheb_basis` — used) |
| `scene.py` | `_bbox_overlap`, `Scene._overlaps` | superseded by the live STRtree coupling partition |
| `templates.py` | `AlignedCutout.as_block_reduced`, `as_block_replicated` | superseded by `Template.downsample`/`_block_reduce` |
| `psf_map.py` | `PSFRegionMap.lookup_key_slow` | reference twin of live `lookup_key` |
| `sim_data.py` | `Frame` | dataclass unused even inside `sim_data.py` |
| `psf.py` | `jwst_header`, `jwst_probe_headers` | trivial MAST header-fetch pair; only a commented example call |

**Deleted — planned-but-dropped small features (trivially recreatable):**

| File | Symbol | Note |
|---|---|---|
| `catalog.py` | `noise_equalised_image` | 1-line `data*sqrt(weight)`; live detection uses `detect_sources` |
| `catalog.py` | `detect_peaks` | alt detection wrapper; **corrects** the old "justified wrapper" note — zero callers |
| `catalog.py` | `vet_by_chi2` | χ² star filter; only ref is a commented-out call |
| `catalog.py` | `CatConfig` | aperture-config stub, never instantiated (docstring mislabels it "SparseFitter") |

**Deleted — dead within the vendored deblender** (unreachable from its own
`deblend_sources` entry point; separate from the whole-module test-only decision):
`photutils_deblend.py` `_steepest_descent_labels`, `deblend_sources_lutz`,
`deblend_sources_color`.

**Ride-along (unchanged):** if `write_wcs_csv` (Tier A, held) is *deleted* rather
than fixed, its ~8 header-parsing helpers (`mast_url_for_filename`,
`extract_dataset_from_comments`, `mjdref_from_header`, `pick_exptime`,
`open_remote_sci_header`, `cd_from_header`, `row_from_header`, `output_csv_path`)
become dead too and should go in the same commit.

---

## Held items — resolutions

The four items originally held out of Tier A (each tied to a parked `xfail` test
or a behavior change). Resolved one at a time with the user.

### Item 1 — `utils.rebin_wcs`: **deleted** (2026-07-24)

Broken on every call (`factor = 2**n` referenced an undefined `n`; docstring
described a power-of-two `n` interface at odds with the `factor` signature), zero
callers, superseded by the live `templates.scale_wcs_pixel`. Deleted the function,
its parked xfail `test_utils.py::test_rebin_wcs_latent_name_error`, and the now-
unused import. **Recover from `9b47342:src/mophongo/utils.py` if ever needed** —
but `scale_wcs_pixel` is the maintained equivalent.

### Item 2 — non-scene `SparseFitter` solve path: **retired to scene-only** (2026-07-24)

`SparseFitter.solve()` only ever implemented `solve_method="scene"` (the default;
notebooks show the method evolved `'ata'` → `'scene'`). The `'all'`/`'lo'` paths
were never built, and the `bright_mask`/`snr` machinery in `__init__` fed only a
commented-out assignment. No live code set `solve_method` to anything but
`"scene"`. Changes made:

- **Deleted** the dead `snr`/`flux_est`/`err_est` block in `SparseFitter.__init__`
  (was `fit.py:741-751`) — it computed a per-construction `quick_flux` +
  `predicted_errors` whose only consumer, `self.orig_bright = ...`, was commented
  out. Removing it also drops a wasteful computation on every construction.
- **`solve()` now fails loudly** for any non-`"scene"` method (`raise ValueError`)
  instead of dispatching to the non-existent `self.solve_all` (which raised
  `AttributeError`). The `solve_method` field comment updated to "only 'scene'".
- **Tests:** deleted the `solve_lo` and `bright_mask` xfails
  (`test_lsqr_lo_matches_cg`, `test_bright_source_detection`); **converted**
  `test_solve_method_all_not_supported` from an xfail into a real passing guard
  that asserts the `ValueError`.

**If the non-scene / linear-operator solve is ever wanted back:** the last tree
where `solve()` branched to `solve_all`, plus the `snr`/`bright_mask` scaffolding
and the original three xfail tests, is commit **`9b47342`** — see
`9b47342:src/mophongo/fit.py` (`solve`, `__init__`) and
`9b47342:tests/test_fit.py` (`test_lsqr_lo_matches_cg`,
`test_bright_source_detection`). Note the `'lo'`/`'all'` methods were *never
implemented* even there, so "recovery" means writing them fresh; the retired
scaffolding only shows the intended call shape. The historical `'ata'` solve
predates the scene refactor — see earlier history if that path is wanted.

### Item 3 — `_flux_errors` 1293→1294 swap: **superseded** (2026-07-24)

Not fixed. `SparseFitter._flux_errors` is legacy-path-only (the production
`SceneFitter._flux_errors` already uses the correct `1/√diag`), and the decision
to **retire `SparseFitter`** (below) means this code is being deleted, not
repaired. Its parked xfail `test_flux_error_orientation_matches_analytic` will be
removed with `test_fit.py`.

### Item 4 — `write_wcs_csv` `continue` short-circuit: *still pending*

Independent of the fitter (it's a `utils.py` FITS-header CSV helper). Unresolved;
revisit after the retirement.

---

## Planned: retire the legacy `SparseFitter`

**Decision (2026-07-24):** the scene solver (`SceneFitter`/`Scene`, default
`run_scene_solver=True`) is the only solver needed. `SparseFitter` is the original
photometry solver that itself grew an internal scene-partition path
(`solve_scene`), so it is a near-complete duplicate of the standalone scene
module. Retiring it removes the largest duplication in the codebase (Tier C).

Scope this as **its own tier / its own commit**, with a real regression run — it
touches the public API, the pipeline, and four test files, so it is bigger than
the Tier A/B deletions.

### The one hard constraint

`FitConfig` **lives in `fit.py`** (`fit.py:49`) and is imported by `__init__.py`,
`pipeline.py`, `astrometry.py`, `scene.py`, `scene_fitter.py`. So **`fit.py` is not
deleted** — it is stripped down to `FitConfig` (+ any genuinely shared helper).
Either keep `FitConfig` in `fit.py` (least churn) or move it to a new `config.py`
and update those 5 import sites. Recommend keeping it in `fit.py` for now.

### What gets removed

- **`fit.py`**: the `SparseFitter` class and its embedded scene machinery —
  `build_scene_tree_from_normal`, `merge_small_scenes`, `make_basis_per_scene`,
  `assemble_scene_system_self_AB`, `summarize_scenes`, `_solve_scenes_with_shifts`,
  `solve`, `solve_scene`, `_flux_errors`, `build_normal_tree`, `predicted_errors`,
  `quick_flux` wrappers, etc. Free function `build_normal_matrix` goes too (only
  tests use it). **Keep `FitConfig`.**
- **`__init__.py`**: drop `SparseFitter` from imports and `__all__`.
- **`pipeline.py`**: remove the `run_scene_solver=False` legacy branch
  (`else: "Running legacy solver"`, ~`:1650-1712`), the `from .fit import
  SparseFitter` import (`:1311`), and the stale `SparseFitter` type hints /
  return-type annotations (`:199,203,229,1308,1928` — several already wrong, since
  `run()` returns a 2-tuple). Audit the extend/refit helper at `:199`
  (`fitter: SparseFitter`) — determine whether the scene path uses it and, if so,
  retype it to `SceneFitter`; otherwise remove it.
- **Tier C follow-on**: the pipeline legacy-branch scaffolding (already logged in
  Tier C) is deleted as part of this.

### Tests to update (this is most of the work)

- **`test_fit.py`** — almost entirely `SparseFitter`/`build_normal_matrix`
  exercises; delete the file (or keep only whatever still tests a retained helper).
- **`test_benchmark.py`** — `SparseFitter` benchmark (already `benchmark`-marked /
  deselected); delete or repoint to `SceneFitter`.
- **`test_astrometry.py`** (`:78,:127`) — real astrometry tests built on
  `SparseFitter`; **repoint** to the scene solver, don't just delete.
- **`test_scene_fitter.py`** (`:160`) — uses `SparseFitter` to build a normal
  matrix for comparison; repoint to `scene_fitter.build_normal`.

### Suggested order (each step green before the next)

1. Repoint `test_astrometry.py` and `test_scene_fitter.py` onto the scene solver;
   confirm green. (Do this *first* so the safety net doesn't depend on the code
   being removed.)
2. Delete `test_fit.py` and `test_benchmark.py` (or reduce to retained helpers).
3. Remove the pipeline legacy branch + `SparseFitter` import/type-hints.
4. Remove `SparseFitter` (and the free `build_normal_matrix`) from `fit.py`,
   keeping `FitConfig`.
5. Update `__init__.py`.
6. Full suite + one real-data scene-solver pipeline run to confirm no regression.

### Recovery

The full `SparseFitter` implementation is preserved in git at the pre-retirement
commit; `git show <that-commit>:src/mophongo/fit.py` restores it. Record the exact
hash in the retirement commit message.

---

## Tier B — decide first (redundant, but reachable / duplicated)

These are duplicates where one copy is redundant but both are currently
reachable, or where consolidation needs a one-line "keep which?" decision.

### Small bbox / slice-intersection helpers duplicated across 4 files

- `utils.py:44` `intersection` == `fit.py:754` `SparseFitter._intersection`
  (byte-identical).
- `fit.py:772` `SparseFitter._slice_intersection` == `scene_fitter.py:22`
  `_slice_intersection` (byte-identical).
- `pipeline.py:672` `_intersect_slices` is a related *extended* variant (maps
  into two local frames) — not a pure duplicate; leave.

Cleanup: keep one copy of each identical helper in `utils.py`, import elsewhere.
Low risk, but `scene_fitter.py` is on the critical path, so verify imports.

### Direct convolution copy-pasted into two modules

`utils.py:493` `convolve2d` and `templates.py:1591` `_convolve2d` are
byte-identical sliding-window `einsum` convolutions (`sim_data.py:17` imports the
templates copy; utils uses its own). The direct form may be a deliberate
small-kernel performance choice vs `scipy.signal.fftconvolve`, but it should live
in **one** shared helper. Consolidate into `utils.py`.

### `_mean_downsample` / `_block_reduce` vs `astropy.nddata.block_reduce`

- `catalog.py` `_mean_downsample` (Tier A delete of the dup) and
  `templates.py:90/103` `_block_reduce`/`_block_replicate` reimplement
  `astropy.nddata.block_reduce`/`block_replicate`, which **both files already
  import and use elsewhere** (`templates.py:16,656,727`; `catalog.py:30,273,505`).
  The hand-rolled versions force `float32` and truncate the remainder instead of
  padding. Decision needed: is the truncate/dtype behavior intentional? If yes,
  document it + consolidate into `utils.py`; if no, replace with `block_reduce`.

### pipeline.py — `_aperture_sum_on_template` / `_aperture_sum_on_map`

`pipeline.py:447` and `:461` differ only in which array they photometer; both
build the same `CircularAperture` and call `aperture_photometry`. Collapse into
one function taking the source array as a parameter. Low risk.

### catalog.py — `get_bg_and_ivar` (live) vs `calibrate_ivar_with_bg_median` (Tier A)

Listed under Tier A for deletion, but if the median-detrending in the dead copy
is wanted, the decision is "fold into `get_bg_and_ivar`" instead of delete.

---

## Tier C — critical-path duplication (report only; do NOT touch without sign-off)

> **Resolution (2026-07-24): retire the legacy `SparseFitter`.** The user
> confirmed the scene solver (`SceneFitter`/`Scene`, the default) is the only
> solver needed. `SparseFitter` and its embedded copy of the scene machinery are
> the "legacy" duplication described below; removing `SparseFitter` collapses this
> whole item. See **"Planned: retire the legacy SparseFitter"** below for scope.
> The section below is retained as the analysis that motivated the decision.

### The scene solver is implemented **three times**

The joint flux+astrometry scene-partition/whiten/solve pipeline exists as three
parallel copies:

1. **`fit.py`** — embedded in `SparseFitter`: `build_scene_tree_from_normal`
   (`:328`), `merge_small_scenes` (`:425`), `make_basis_per_scene` (`:536`),
   `assemble_scene_system_self_AB` (`:584`), `summarize_scenes` (`:661`),
   `_solve_scenes_with_shifts` (`:954`), `solve_scene` (`:1130`). Reachable via
   the **default** `FitConfig.solve_method="scene"` when
   `pipeline.run_scene_solver=False`.
2. **`scene.py`** — scene-local calling convention: `build_scene_tree_from_normal`
   (`:72`), `merge_small_scenes` (`:169`), `make_scene_basis` (`:286`),
   `assemble_scene_system_AB` (`:359`), `summarize_scenes` (`:479`),
   `generate_scenes` (`:498`), `Scene` (`:593`). Reachable via
   `pipeline.run_scene_solver=True`.
3. **`scene_fitter.py`** — `SceneFitter` (`:115`), `build_normal` (`:34`, whose
   docstring says *"Stateless clone of SparseFitter.build_normal_tree"*). The
   solve backend `Scene` calls into.

The `build_scene_tree_from_normal` / `merge_small_scenes` / `summarize_scenes`
bodies were confirmed near-line-for-line identical between `fit.py` and
`scene.py`, **diverging only in defaults** — e.g. `coupling_thresh` 0.03 vs 0.01
(and `scene.py`'s inline comment still says "3% threshold", a stale copy
artifact), and `minimum_bright` computed vs hardcoded `10`.

This is the single largest drift risk in the codebase: fixes to the actively
developed `scene.py`/`scene_fitter.py` (recent Estimator-3 work) do **not**
propagate to `fit.py`'s copy, which is still the default-config path. **Do not
unify silently** — the diverged thresholds may be intentional. Recommended (for
the user to decide): pick `scene.py`/`scene_fitter.py` as canonical and have
`SparseFitter.solve_scene` delegate to it, or explicitly retire the legacy
`fit.py` scene path.

### pipeline.py — legacy-solver `else` branch scaffolding (`pipeline.py:1652-1724`)

Inside `else: print("Running legacy solver")` (reachable when
`run_scene_solver=False`, exercised by `test_template_extension.py`):
dead `fitter_cls` selection now hardcoded to `SparseFitter` (`:1652-1656`), a
separate non-joint astrometry call marked `# @@@ this is very expensive`
(`:1666-1671`), a flux-only re-solve calling the commented-out
`_add_templates_for_bad_fits` (`:1680-1692`), a soft non-negative-prior re-solve
(`:1697-1705`), and a commented scene-vs-full-residual check (`:1713-1724`).
Stale scaffolding, but on a tested legacy path — flag before deleting.

### `build_normal_matrix` (`fit.py:1643`)

Broken bound-method dispatch — see Tier A (it lives under the OBSOLETE banner).
Reachable only via `config.normal="loop"`, which raises `AttributeError`; the
default `"tree"` path is fine. Tier A delete, but the `"loop"` config option must
be dropped or the method re-bound as part of the same change.

---

## Test-only modules (not production; remove module **and** its test together)

Each is commented out of `src/mophongo/__init__.py` and imported **only** by its
own test — production uses the library equivalent directly.

| Module | Lines | Only importer | Notes |
|---|---|---|---|
| `deblender.py` | 438 | `tests/test_deblender.py` | Custom symmetry/hybrid deblend; prod uses `photutils.segmentation.deblend_sources` |
| `photutils_deblend.py` | 658 | `tests/test_photutils_deblend.py` | Vendored fork of photutils deblend + "compactness" option |
| `sim_data.py` | 209 | `tests/test_realistic_pipeline.py` | Synthetic mosaic generator |
| `jwst_psf.py` | 510 | `tests/test_jwst_psf.py` | On the "exclude when validating" list (needs download) |

These are intentional experimental/vendored forks, not accidental duplication —
but they're ~1800 lines of dead weight at the package level. Removal is a
product decision (archive vs delete), and each removal must drop its test file
too. If kept, document *why* in a header comment.

---

## Not dead — flagged for awareness

- **Stale doc pointer:** `CLAUDE.md` references `local_astrometry.py`, which was
  deleted (git history; alongside `fft.py`, `kernels.py`). Update CLAUDE.md.
- **`self.fit` / `Pipeline.plot_result`** — `run()` never populates `self.fit`
  (append commented at `pipeline.py:1745`), so `plot_result()` raises
  `IndexError`. Left as-is per prior decision; `test_plot_result` is xfail.
- **`Scene.create_scene_graph` / `overlay_scene_graph` / `add_residuals`** —
  unimplemented plot helpers, **wanted eventually**. A feature to build, not dead
  code. Parked: `test_scene_fitter.py::test_scene_graph_helpers_are_unimplemented`.
- **`scene.py` redundant imports** — numpy/scipy.sparse/`FitConfig` imported
  twice at top (`:6/14`, `:7/15`, `:10/17`) plus a third stray import block at
  `:278`. Harmless, but evidence of unreviewed file-merges; tidy opportunistically.
- Justified library "reimplementations" (leave as-is): `get_wcs_pscale`
  (caching wrapper, `utils.py:1158`), `bin_factor_from_wcs` (adds integer-ratio
  validation, `:227`), `fit_psf_stamp` (closed-form 2-param lstsq, `catalog.py:423`),
  and the flux-normalized `gaussian`/`moffat`/`elliptical_*` profile generators
  (`utils.py:316-436` — `astropy.modeling` doesn't offer the total-flux→amplitude
  convenience).
  **Correction:** `detect_peaks` (`catalog.py:406`) was previously listed here as
  a used wrapper, but the rescan found it has **zero callers** — it's a Round-2
  Tier-A delete candidate, not a keep.

---

## Removal workflow (how to remove without touching critical infrastructure)

Do this as a **dedicated branch**, one tier at a time, tests green between steps.

1. **Baseline.** On a clean branch off `apcor-estimator3`, run the agreed test
   set (excluding the known-parked ones: benchmark/speed, jwst_psf/download,
   astro_fit) and record it green. This is the regression oracle.

2. **Tier A, in small commits.** Delete the zero-caller items above, grouped by
   file. After each file's deletions:
   - `grep` the deleted symbol name across `src/ tests/ examples/` to confirm no
     surviving reference (comments/notebooks included).
   - Run the test set. Any parked `xfail` whose target you just deleted should be
     deleted in the *same* commit (or flipped to a real test if you fixed rather
     than removed).
   - Update `__init__.py`/`CLAUDE.md` if a public name or a doc pointer changes.
   Suggested commit order (lowest coupling first): `astro_fit.py` module +
   `pipeline.py:1312` import → `fit.py` OBSOLETE tail → `psf.py` `NEffectivePSF` +
   shadowed helpers + commented block → `templates.py` `_old`/`block_aligned` →
   `catalog.py` doubled/dead helpers → `astrometry.py` `_old` →
   `utils.py` `rebin_wcs`/`write_wcs_csv` → `pipeline.py` Zarr block.

3. **Tier B, one decision each.** For every Tier-B item, get the user's
   "keep which copy?" answer, then consolidate into the single canonical location
   (usually `utils.py`) and replace call sites with an import. Re-run tests. These
   touch `scene_fitter.py` (critical path) for the slice-helper — verify imports
   resolve and the scene solver still runs on a real example.

4. **Tier C — do not delete in this pass.** Leave the scene-solver triplication
   and the legacy `else` branch alone until the user decides the canonical solver
   and whether to retire the legacy `SparseFitter` scene path. When that decision
   lands, it's a *refactor* (delegate one to the other), not a delete, and needs a
   full pipeline regression run on real data — not just unit tests.

5. **Test-only modules — product decision.** Only after the user says
   archive/delete: remove the module and its test file in the same commit, and
   drop the commented import from `__init__.py`.

6. **Final sweep.** After Tier A+B: run the full suite, run one real-data /
   example pipeline (scene solver **and** legacy path) to confirm nothing on
   either reachable branch regressed, and re-grep for any newly-orphaned symbols
   the deletions exposed.

**Guardrails.** Never delete anything still imported by a *passing* test without
also handling that test. Never touch the scene solver (`scene.py`,
`scene_fitter.py`), `Pipeline.run`, `Templates`, `PSF`, `Catalog`, or
`AstroCorrect` internals in Tier A/B. Keep each commit to one concern so any
regression bisects cleanly.
