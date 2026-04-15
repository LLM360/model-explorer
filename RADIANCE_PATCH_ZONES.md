# Radiance Patch Zones

This file defines where Radiance may patch the vendored Model Explorer tree for
v1.

Radiance vendors the approved upstream Model Explorer snapshot from
`https://github.com/LLM360/model-explorer` at `vendor/model_explorer/`.

Radiance owns only the documented patch zones below; everything else stays
upstream-owned by default as the local ownership model for v1.

## Allowed Patch Zones

- `src/server/**` for integrated Flask server wiring, route registration, and
  artifact-serving integration.
- `src/ui/**` for Angular shell composition, Radiance workspace surfaces, and
  event wiring that stays aligned with upstream structure.
- `src/example_node_data_providers/**` only as reference material when upstream
  overlay hooks need to be mirrored in first-party code.
- Adjacent docs in the vendored tree when a local patch needs an explanatory
  note.

## Discouraged Patch Zones

- `src/ui/src/components/visualizer/**` and other low-level rendering internals.
- packaging, release, and CI files under `.github/`, `ci/`, and test-only
  screenshot baselines unless a later issue owns that change explicitly.
- example adapters or demos unless the change is required to preserve a local
  integration contract.

## Hard Rules

- Extend the Angular shell and Flask server directly; do not create a parallel
  React shell, separate admin UI, or alternate backend service.
- Prefer additive hooks, route registration, and shell composition over deep
  rewrites of upstream behavior.
- Low-level renderer internals stay off-limits unless overlay or event
  requirements force a documented exception in the owning issue and PR.
- Keep all Radiance-specific behavior discoverable through narrow, documented
  patch points.

## Expected Downstream Use

- `#22` extends the vendored Flask server into the integrated Radiance
  application shell.
- `#25` uses upstream graph load and event hooks to inject overlays without a
  page reload.
- `#26` composes the final Radiance workspace inside the Angular shell.

Any future change outside these expectations should open a dedicated issue
before code lands.
