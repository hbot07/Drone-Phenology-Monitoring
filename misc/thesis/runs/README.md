# Thesis Runs

This folder stores reproducible analysis runs used while writing the thesis paper.

The idea is:

- keep code that extracts paper-ready tables/figures here;
- write outputs into named run folders;
- never edit pipeline outputs by hand;
- record what data and pipeline output each result came from.

## Current LHC Run

Generate initial LHC thesis artifacts from the available `input/input_om_lhc` data and existing `output/lhc_pipeline_fixed` pipeline output:

```bash
bash misc/thesis/runs/run_lhc_current.sh
```

Main outputs are written to:

```text
misc/thesis/runs/lhc_current/
```

This run is useful for drafting tables and deciding figures, but it is not the final full thesis run unless the paper explicitly uses the 8-orthomosaic LHC subset and excludes the problematic 2025-12-09 orthomosaic.
