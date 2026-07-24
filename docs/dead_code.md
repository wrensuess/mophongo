# Dead code inventory

Confirmed-dead or broken-and-unreachable code paths surfaced during the
2026-07 test-suite cleanup. These are removal/repair candidates for a dedicated
pass — **not** things that affect a default scene-solver run. Each entry notes
where it lives, why it's dead, and the cleanup.

Tests that would exercise these are currently parked as `xfail` so the defects
stay visible without breaking the suite; fixing/removing the code lets those
xfails flip to real passing tests (or be deleted alongside).

## fit.py

- **`SparseFitter._flux_errors` — returns `0.0` for isolated sources.**
  The independent-pixel branch returns `sqrt(diag)*covar_power`, which collapses
  to `0.0` when the off-diagonal coupling is zero (an isolated source); the
  physically-correct `1/sqrt(diag)` line sits just below it, unreachable.
  Off the default path — the scene solver (`SceneFitter._flux_errors`, correct
  `1/sqrt(diag)`) provides the reported `err_N` columns. Cleanup: delete the dead
  branch and use `1/sqrt(diag)`, or remove the method if nothing else calls it.
  Parked test: `test_fit.py::test_flux_error_orientation_matches_analytic`.

- **`SparseFitter.solve_lo` — referenced, never defined.**
  `solve()`'s docstring lists a `'lo'` (linear-operator) method but no `solve_lo`
  exists on the class. Cleanup: drop `'lo'` from the options, or implement it.
  Parked test: `test_fit.py::test_lsqr_lo_matches_cg`.

- **`SparseFitter.bright_mask` — never assigned.**
  `__init__` computes the SNR but the `self.bright_mask = ...` assignment is
  commented out (~`fit.py:751`); the only other reference is inside dead code.
  Cleanup: assign it, or remove the attribute and its parked test.
  Parked test: `test_fit.py::test_bright_source_detection`.

- **`SparseFitter.solve_all` — referenced, never defined.**
  `solve()` dispatches `else -> self.solve_all(config)` for any
  `solve_method != "scene"`, so `"all"`/`"lo"` raise `AttributeError`.
  See the note below on whether to implement vs. remove.
  Parked test: `test_fit.py::test_solve_method_all_not_supported`.

- **`solve_scene_shifts` (free-function tail) — unreachable.**
  Defined after a `return` inside `merge_small_scenes_old` (~`fit.py:1557`); not a
  class method. Cleanup: delete.

## utils.py

- **`rebin_wcs` — `NameError` on any call.**
  Body does `factor = 2**n` referencing an undefined `n`. Never called anywhere
  in `src/`. Cleanup: delete, or fix to use the `factor` argument.
  Parked test: `test_utils.py::test_rebin_wcs_is_broken` (xfail).

## pipeline.py

- **`_add_templates_for_bad_fits` — feature commented out.**
  The multi-template "bad fit" recovery call is commented out in `run()`
  (~`pipeline.py:1683-1692`), so `multi_tmpl_*` config has no effect and
  `test_pipeline_multitemplate_pass` no longer tests the feature it names.
  Cleanup: re-wire it, or delete the method + `multi_tmpl_*` config.

## astro_fit.py

- **`GlobalAstroFitter` — dead class.**
  Imported in `pipeline.py:1312` but never instantiated; production astrometry
  uses `AstroCorrect` + scene shifts. Its test file (`test_astro_fit.py`) has
  already been deleted. Cleanup: remove the class and the stale import.

---

## Deferred, NOT dead (do not remove)

- **`self.fit` / `Pipeline.plot_result`** — `run()` never populates `self.fit`
  (append commented at `pipeline.py:1745`), so `plot_result()` raises
  `IndexError`. Left as-is per decision; `test_plot_result` is xfail.
- **`Scene.create_scene_graph` / `overlay_scene_graph` / `add_residuals`** —
  scene-graph plot helpers are unimplemented but **wanted eventually**. Treat as
  a feature to build, not dead code. Parked test:
  `test_scene_fitter.py::test_scene_graph_helpers_are_unimplemented` (xfail).
