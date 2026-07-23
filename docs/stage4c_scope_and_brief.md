# Stage 4c — unmask the Estimator-3 catalog-tie denominator

Scoping note + implementer brief. Written 2026-07-17 after Stage 4b landed
(commit a4e0521). Code + review cycle deferred to a later session; this
document is the cold-start handoff for a Sonnet implementer.

---

## 1. Verified diagnosis (do not re-derive — reproduced numerically on the real F1500W run)

Stage 4b moved the aperture-to-total CORRECTIONS (totcor1, apcor1, and the
shape part of tcor_int) onto the source's partially-unmasked model: data over
the owned support + the noiseless PSF component M extended over the full stamp,
so a close neighbour's ownership boundary no longer clips the correction. That
eliminated the crowding tail in totcor1 (now flat vs nearest-neighbour).

But the **Estimator-3 catalog tie is still computed on the ownership-MASKED
fitted template**, so est3cat re-inherits exactly the crowding artifact 4b
removed everywhere else.

The two quantities that carry it:
- `template_norm` = Sigma(H) over the owned support (the masked F444W total),
  used in the tcor_int denominator.
- `f444w_ktot` = `_model_kron` Kron flux measured on the masked model stamp
  (`orig_t.data * orig_t.template_norm`), divided by the PSF EE.

Key algebra (pipeline.py:1096-1111, 1174-1187): in est3cat, f444w_ktot
CANCELS (`tcor_int * s_cat = ftot/(template_norm * apF_book)`), so
```
est3cat ~= flux_1 * (c_det/c_b) * ftot / template_norm  + res_sum
```
i.e. est3cat/est1 ~= ftot/template_norm = (catalog F444W total)/(our masked
template F444W total). A bug in the f444w_ktot Kron calc therefore does NOT
move est3cat (it only moves est3int and the panel-e f444w_ktot diagnostic);
the est3cat crowding leak is driven by `template_norm` being masked.

Numerical proof (24-26 mag, median normalized to the isolated bin,
nearest-neighbour bins [<0.6" | 0.6-1.2" | 1.2-2.4" | isolated]):
```
Stage-4 totcor1  (MASKED correction, the original bug): 1.40 / 1.24 / 1.07 / 1.0
Stage-4b totcor1 (UNMASKED, 4b fixed it)              : 0.99 / 0.99 / 1.00 / 1.0   <- FLAT
Stage-4b est3cat/est1 (MASKED tie, bug survives)      : 1.49 / 1.19 / 1.07 / 1.0
```
The tie's crowding signature (1.49x crowded/isolated) matches the pre-4b
masked-totcor1 signature (1.40x) in shape and magnitude -- the same masking
mechanism, in the code 4b did not touch. Code-level confirmation: `_model_kron`
(pipeline.py:476-556) reads `orig_t.data * template_norm` (masked) with no
`flux_beyond` term; the tcor_int/f444w_ktot block (pipeline.py:1063-1101) uses
`template_norm` and `apF_book` (masked), unlike the apF_corr/apB_corr block
(1003-1031) which uses the 4b `flux_beyond_*`.

A second, smaller, SEPARATE effect underlies it (do not try to "fix" this one):
even isolated, the catalog `f_f444w` runs ~10-16% above IDL's F444W total
(f_f444w/idl_Ff444w ~ 1.10-1.16, roughly mag-independent), while our internal
total sits close to IDL. That is a catalog-vs-IDL total-flux DEFINITION
difference, external to mophongo; on its own it pushes isolated-faint est3cat
slightly FAINTER than est1 (est3cat/est1 ~ 0.87-0.93 isolated). Stage 4c should
remove the crowding leak and leave this definitional residual visible.

## 2. The fix (recommended approach)

Use the **unmasked-model F444W total** in the est3 tie denominator, exactly
consistent with Stage 4b. The unmasked total is ALREADY computed and stored as
a per-template scalar: `template_norm + flux_beyond_stamp`.

Why that is the right total: `flux_beyond_stamp = A_src*(1/c_det - f_cut)`,
`f_cut = Sigma(psf_cut over ext_psf)`. In the faint pure-PSF limit
`template_norm = A_src*f_cut`, so `template_norm + flux_beyond_stamp =
A_src/c_det` = the source's containment-corrected TRUE total, independent of how
the ownership mask clips the support. Substituting it for `template_norm` in the
tie denominator makes est3cat crowding-independent (the leak vanishes) while
leaving isolated sources ~unchanged (flux_beyond_stamp is small there).

The core change: in the tcor_int denominator (pipeline.py:1092,
`denom = template_norm_i * apF_book`) use the unmasked total
`(template_norm_i + flux_beyond_stamp_i)` in place of `template_norm_i`. Per the
D1 ruling (Sec 3) this is done consistently across the WHOLE Estimator-3 system
-- `f444w_ktot` is also unmasked (Sec 3 D2) so est3int and the panel-e
diagnostic lose the crowding dependence too, not just est3cat.

## 3. Design decisions -- RULINGS (user, 2026-07-17)

(D1) **RESOLVED: fix the WHOLE Estimator-3 system.** Unmask both the tcor_int
denominator (`template_norm`) AND `f444w_ktot` (the Kron), so est3int and the
panel-e f444w_ktot diagnostic lose the crowding dependence too, and tcor_int is
internally consistent (unmasked Kron numerator / unmasked denominator).

(D2) **RESOLVED (user-confirmed, option a): scalar top-up for the unmasked Kron.** Keep the
fast masked Kron in `_model_kron`, then add ONE more per-template scalar
computed at extraction time -- the PSF-shaped model flux inside the Kron
aperture but OUTSIDE the owned support -- exactly analogous to the existing
`flux_beyond_aper` (which does this for the measurement aperture). The
alternative (rebuild H_corr = H + A_src*psf_cut per source for the Kron) is
rejected: more faithful for bright extended crowded sources but costs a PSF
resample + photutils call per source, and the crowding leak lives in the
faint/PSF-dominated population where the Kron radius already floors to the
color-aperture circle (the `apcor_from_psf` shortcut), so the top-up is
essentially exact there. The top-up keeps the Kron CONVENTION intact
(f444w_ktot stays comparable to the catalog and IDL), which the curve-of-growth
total would not.
  - IMPLEMENTATION NOTE: the top-up scalar is the PSF aperture-delta at the KRON
    radius, not the measurement-aperture radius, so `flux_beyond_aper` cannot be
    reused directly -- but the Kron radius (`r_kron_circ`) is only known inside
    `_add_aperture_photometry` (from `_model_kron`), while the PSF geometry
    (psf_cut, ext_psf, A_src) only lives in `_extended_composite`. Resolve this:
    either (i) store enough per-template geometry to evaluate the delta at an
    arbitrary radius later (e.g. the cumulative PSF-outside-support profile, a
    small 1-D array -- weigh memory), or (ii) compute the delta in
    `_add_aperture_photometry` by re-sampling the detection PSF there (the PSF
    region lookup already exists in `_extended_composite` and can be factored
    out). For the faint/floored population (i) collapses to a single value at
    the floor radius. The implementer should pick the cheaper consistent route
    and flag it for review.

(D3) **RESOLVED: accept the residual.** Do the close-pair budget analysis (as
the 4b science review did) to quantify it, but the ownership-vs-catalog-deblend
difference is partly intrinsic -- putting the flux on the CATALOG scale is
itself the choice to accept the catalog's deblending, so some per-pair
difference is expected and acceptable, not a further bug to chase.

(D4) **RESOLVED: est1/est2 essentially unchanged.** They do not use the tie, so
they must match to within run-to-run nondeterminism (the ~sub-percent per-source
jitter seen between the july16/july17 runs), NOT necessarily byte-identical.
Guard with a test at that tolerance.

## 4. Implementer brief

Anchor: commit a4e0521 (Stage 4b), branch apcor-estimator3, clean tree.
Python: /opt/anaconda3/envs/mophongo/bin/python. NO commits. Touch
src/mophongo/pipeline.py, tests/test_pipeline_aperture.py, and
src/mophongo/templates.py (D2 needs a new stored per-template scalar/profile).
Read CLAUDE.md and docs/aperture_corrections.md Sec 5.1/5.4/6 first.

Core change (D1 = whole system, D2 = scalar top-up):
- The internal F444W total system uses the UNMASKED model throughout:
  - tcor_int denominator: `template_norm + flux_beyond_stamp` (both already
    stored) in place of `template_norm`.
  - `f444w_ktot`: masked Kron flux + the PSF-shaped top-up at the Kron radius
    (per D2), divided by ee_kron as now. In the faint pure-PSF limit this must
    reproduce A_src/c_det (the true total), same as template_norm +
    flux_beyond_stamp.
- Output columns unchanged (same schema as Stage 4b). ap_model/ap_flux/est1/
  est2 unchanged to nondeterminism tolerance (D4).

Acceptance tests (each must fail against the current masked code -- revert-verify):
1. **Crowding flatness of est3cat**: faint target + bright neighbour at several
   separations (reuse the Stage-4b `test_crowding_regression_...` ownership
   scene) -> est3cat/est1 within ~3% of the isolated value across all
   separations (currently 1.49x at <0.6").
2. **Crowding flatness of est3int and f444w_ktot** (D1 = whole system): same
   scene -> est3int/est1 and f444w_ktot lose their crowding dependence too.
3. **Isolated invariance**: for an isolated source (flux_beyond ~ small), the
   whole est3 system changes negligibly -- the fix must not move isolated ties.
4. **est1/est2 unchanged** to nondeterminism tolerance (D4).
5. **Faint-limit identity**: the unmasked total equals A_src/c_det in the pure-
   PSF limit; tcor_int and f444w_ktot reproduce their closed forms with it.

Scoped gate (the ONLY test command):
```
/opt/anaconda3/envs/mophongo/bin/python -m pytest tests/test_pipeline_aperture.py \
  tests/test_templates.py tests/test_template_extension.py tests/test_psf_map.py -q
```
Expect all pass except the 2 pre-existing test_psf_map failures.

Review cycle (per project standing rule): Sonnet code review + Opus science
review before commit; the science review also rules on D3 (residual budget).

## 5. Expected outcome

est3cat/est1 flat vs crowding (crowded ~0.93 instead of 1.49, i.e. the crowded
faint population stops scattering bright). The remaining est3cat-vs-est1
difference should collapse to the isolated residual (~0.87-0.93 faint), which is
the catalog-vs-IDL definitional offset -- a SCIENCE question about which F444W
total system to anchor to, NOT a code bug, for the user/team to decide
separately.

## 6. Run-side

No run_770.py or geojson changes. After the fix lands, a validation run
(version bump only) grades est3cat/est1 vs nearest-neighbour distance (should be
flat) and confirms est1/est2 unchanged.
