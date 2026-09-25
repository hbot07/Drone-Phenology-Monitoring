/**
 * param-docs.js — single source of truth for parameter documentation.
 *
 * Both home.html (inline field help + info tooltips) and parameter-guide.html
 * read from this object, so the field hint and the guide can never drift apart.
 *
 * Each entry keyed by the parameter's DOM id (without the "nr-" prefix where
 * convenient) or a stable doc id. Fields:
 *   label    — human name
 *   step     — which pipeline step it belongs to
 *   short    — one-line hint (shown in the info tooltip and field hint)
 *   why      — why it matters / what it affects
 *   how      — how to choose a value
 *   default  — the default value
 *   range    — valid range or options
 */
window.PARAM_DOCS = {
  // ─────────────────────────── Step 1: Detection ───────────────────────────
  "model": {
    label: "Detector model",
    step: "Step 1 — Crown detection",
    short: "The Detectree2 model variant used to delineate tree crowns.",
    why: "Different models are trained on different canopy types. The default (250312_flexi) is a flexible multi-site model that works across most drone surveys.",
    how: "Use the default unless you have a site-specific model. urban_canopy suits street trees; tropical_dense suits closed tropical canopy.",
    default: "250312_flexi",
    range: "250312_flexi | urban_canopy | tropical_dense",
  },
  "tile_width": {
    label: "Tile width (m)",
    step: "Step 1 — Crown detection",
    short: "Width of each inference tile in metres.",
    why: "The orthomosaic is split into tiles small enough for the detector to process. Larger tiles are faster but use more GPU memory; smaller tiles catch fine detail but take longer.",
    how: "25 m works well for 5 cm/px drone imagery. Increase for sparse canopy, decrease if you hit GPU-memory errors.",
    default: "25",
    range: "10–200 m",
  },
  "tile_height": {
    label: "Tile height (m)",
    step: "Step 1 — Crown detection",
    short: "Height of each inference tile in metres.",
    why: "Same role as tile width, for the vertical dimension. Keep it equal to width for square tiles.",
    how: "Match tile width (25 m) unless your survey is strongly rectangular.",
    default: "25",
    range: "10–200 m",
  },
  "tile_buffer": {
    label: "Tile buffer (m)",
    step: "Step 1 — Crown detection",
    short: "Overlap between adjacent tiles to reduce edge artefacts.",
    why: "Crowns straddling a tile boundary can be cut in half. The buffer adds overlap so each crown is seen whole in at least one tile, then duplicates are merged.",
    how: "Typically 50–60% of tile size. With 25 m tiles, 15 m is a good default.",
    default: "15",
    range: "0–100 m",
  },
  "iou_threshold": {
    label: "IoU threshold",
    step: "Step 1 — Crown detection",
    short: "Intersection-over-union threshold used to remove duplicate crowns.",
    why: "When two overlapping detections describe the same tree (often from tile overlap), the one with lower IoU overlap is dropped. Higher values keep more near-duplicates; lower values merge more aggressively.",
    how: "0.7 is a safe default. Lower it if you see doubled crowns; raise it if distinct nearby trees are being merged.",
    default: "0.7",
    range: "0.1–1.0",
  },

  // ─────────────────────────── Step 2: Tracking ────────────────────────────
  "base_threshold_tag": {
    label: "Base threshold (%)",
    step: "Step 2 — Crown tracking",
    short: "Confidence level for crown matching across observation dates.",
    why: "Controls which detections are used when building temporal chains that link the same tree across dates. Lower = denser (more crowns, more noise); higher = sparser (fewer, more confident crowns).",
    how: "45 is the default. Lower to 35 if too few trees are tracked; raise to 55 if noisy detections are polluting chains.",
    default: "45",
    range: "0–100",
  },
  "align_threshold_tag": {
    label: "Align threshold (%)",
    step: "Step 2 — Crown tracking",
    short: "Confidence for the anchor crowns used to align dates.",
    why: "Alignment uses only very confident crowns as anchors to register one date against another. Higher values give more stable alignment but fewer anchors.",
    how: "65 is the default and works for most surveys. Raise it if alignment drifts; lower it if there are too few anchors on sparse sites.",
    default: "65",
    range: "0–100",
  },
  "skip_viz": {
    label: "Skip visualizations",
    step: "Step 2 — Crown tracking",
    short: "Skip generating per-chain and consensus visualization images.",
    why: "The tracking step can render diagnostic images of every chain. These are useful for debugging but slow and disk-heavy.",
    how: "Leave checked (skip) for first runs. Uncheck only when you need to inspect tracking quality visually.",
    default: "checked (skip)",
    range: "on/off",
  },

  // ────────────────────────── Step 3: Phenology ────────────────────────────
  "w_veg_amp": {
    label: "w_veg_amp — vegetation amplitude",
    step: "Step 3 — Phenology analysis",
    short: "Weight on peak-to-trough vegetation range in the deciduousness score.",
    why: "The deciduousness score (DS) combines four seasonal signals. This weight controls how much the overall swing in vegetation greenness contributes. Strongly deciduous trees have a large amplitude.",
    how: "0.35 default. Raise it to emphasise trees with big seasonal green-up/green-down; lower it if amplitude is noisy on your site.",
    default: "0.35",
    range: "0–2",
  },
  "w_depth": {
    label: "w_depth — seasonal depth",
    step: "Step 3 — Phenology analysis",
    short: "Weight on dormancy depth relative to peak greenness.",
    why: "Captures how far the canopy falls at its most dormant. Trees that go fully bare score high on depth.",
    how: "0.30 default. Raise it to separate fully-deciduous from semi-deciduous trees.",
    default: "0.30",
    range: "0–2",
  },
  "w_gcc_amp": {
    label: "w_gcc_amp — GCC amplitude",
    step: "Step 3 — Phenology analysis",
    short: "Weight on green chromatic coordinate amplitude.",
    why: "GCC is a colour-ratio greenness index robust to lighting changes. Its seasonal amplitude is another deciduousness cue, complementary to raw vegetation fraction.",
    how: "0.25 default. Useful weight to keep non-zero on sites with variable illumination between dates.",
    default: "0.25",
    range: "0–2",
  },
  "w_tex": {
    label: "w_tex — texture",
    step: "Step 3 — Phenology analysis",
    short: "Weight on Laplacian-variance texture amplitude.",
    why: "Bare canopies look texturally different from full ones (branches vs leaves). Texture amplitude across the season adds an independent signal to the score.",
    how: "0.10 default — a small contribution. Raise cautiously; texture is the noisiest of the four signals.",
    default: "0.10",
    range: "0–2",
  },

  // ────────────────────────── Step 4a: COG tiling ──────────────────────────
  "zoom": {
    label: "Max COG zoom",
    step: "Step 4a — COG tiling",
    short: "Highest zoom level in the tile pyramid.",
    why: "Determines how far a user can zoom into the interactive viewer. z22 corresponds to native 5 cm/px resolution; higher wastes space, lower loses detail.",
    how: "22 matches typical drone GSD. Lower to 20 to save disk if fine detail isn't needed.",
    default: "22",
    range: "14–24",
  },
  "tilesize": {
    label: "Tile size (px)",
    step: "Step 4a — COG tiling",
    short: "Width and height of each PNG map tile.",
    why: "Standard web map tiles are 256×256. Larger tiles mean fewer HTTP requests but heavier individual loads.",
    how: "256 is the safe standard and matches Leaflet defaults. Change only if you know your viewer needs it.",
    default: "256",
    range: "64–512 (step 64)",
  },
  "underlay_om": {
    label: "Underlay OM",
    step: "Step 4a — COG tiling",
    short: "Which observation month loads by default in the viewer.",
    why: "The interactive viewer shows one orthomosaic as the base layer. This picks whether that's the earliest or the most recent date.",
    how: "'last' (most recent) is usually the clearest base. Use 'first' if the earliest date is the reference survey.",
    default: "last",
    range: "last | first",
  },

  // ─────────────────────────── Advanced ────────────────────────────────────
  "exclude": {
    label: "Exclude orthomosaics",
    step: "Advanced",
    short: "Observation months to skip from the whole pipeline.",
    why: "Badly misaligned, cloud-covered, or corrupted dates hurt tracking and phenology. Excluding them is cleaner than letting them pollute chains.",
    how: "Only exclude dates you've confirmed are problematic. Every exclusion shortens the temporal series.",
    default: "none",
    range: "any uploaded OM",
  },
};
