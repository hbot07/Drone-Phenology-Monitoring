from __future__ import annotations

"""Match-case configuration presets.

These encode the case-specific gates and scoring used during edge selection.
The design is intentionally explicit so it is easy to audit and tune.

Notation:
- Let $g_p$ and $g_c$ be prev/curr polygons.
- IoU: $\mathrm{IoU}(g_p,g_c)=\frac{|g_p\cap g_c|}{|g_p\cup g_c|}$
- Overlap fractions:
  - $\mathrm{ov}_p=\frac{|g_p\cap g_c|}{|g_p|}$
  - $\mathrm{ov}_c=\frac{|g_p\cap g_c|}{|g_c|}$

The selector enforces per-case minimum thresholds over these quantities and a
weighted similarity score.
"""

from .matching import strict_aligned_configs, ultra_relaxed_case_configs

__all__ = ["ultra_relaxed_case_configs", "strict_aligned_configs"]
