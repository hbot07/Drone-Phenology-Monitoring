# Local STAC Sentinel-2 RF Results

Generated locally from Microsoft Planetary Computer Sentinel-2 L2A, without GEE.

Feature extraction command used:

```bash
/Users/hbot07/VS\ Code/Drone-Phenology-Monitoring/.venv/bin/python -u \
  python/03_extract_sentinel2_stac_features.py \
  --year 2025 \
  --geometry-mode buffer \
  --buffer-meters 20 \
  --max-items-per-season 4 \
  --label-filter label_esd \
  --out-csv exports/stac_s2_features_2025_buffer20_items4_label_esd.csv
```

Main feature table:

```text
exports/stac_s2_features_2025_buffer20_items4_label_esd.csv
```

Summary table:

```text
outputs/local_rf_stac/metrics_summary.csv
```

Best local signal so far:

```text
label_acacia random split:
  accuracy          0.883
  balanced accuracy 0.798
  macro-F1          0.798
```

Important caution:

```text
Acacia leave-area-out is uneven:
  SV_S1 balanced accuracy 0.344
  SV_S3 balanced accuracy 0.708
  SV_S4 balanced accuracy 0.550
```

Deciduous vs rest is moderate on random split:

```text
accuracy          0.709
balanced accuracy 0.606
macro-F1          0.604
```

ESD multiclass is not yet strong enough as a local-only result:

```text
random split balanced accuracy 0.512
SIT leave-area-out balanced accuracy 0.248
A3 leave-area-out balanced accuracy 0.298
A4 leave-area-out balanced accuracy 0.256
```

Recommendation: use these local results as a working baseline, but run the GEE route next for fuller seasonal composites and faster geometry/experiment sweeps.

## Follow-up Acacia-Specific Run

Because Acacia was the strongest local target, a second STAC feature table was extracted with all Acacia-usable crowns:

```text
exports/stac_s2_features_2025_buffer20_items4_label_acacia.csv
```

Results:

```text
outputs/local_rf_stac_acacia_full/metrics_summary.csv
outputs/local_rf_stac_acacia_full_threshold/metrics_summary.csv
```

Hard 0.5 RF decision:

```text
random split:  accuracy 0.873, balanced accuracy 0.764, macro-F1 0.770
SV_S1 holdout: accuracy 0.486, balanced accuracy 0.481, macro-F1 0.461
SV_S3 holdout: accuracy 0.560, balanced accuracy 0.643, macro-F1 0.560
SV_S4 holdout: accuracy 0.357, balanced accuracy 0.513, macro-F1 0.344
```

CV threshold tuning improved random split balanced accuracy to 0.888 at threshold 0.15, but did not solve leave-area-out transfer. This suggests the main problem is spatial/domain shift, not just an overly conservative probability cutoff.
