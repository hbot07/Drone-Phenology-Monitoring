# Collecting Drone Imagery

This guide covers the collection habits that matter for the phenology pipeline. The detailed flight-planning and drone-operation reference remains Paul Pop's end-to-end manual:

- https://github.com/paulvpop/drone-mapping-end-to-end-workflow/blob/main/user_manual.md

Use that manual for the mechanics of creating flight paths, exporting missions, uploading them to the drone or controller, handling multi-battery flights, exporting logs, and understanding the general photogrammetry workflow. This project adds one extra requirement: repeatability across dates. The drone imagery is not collected just to make one good map; it is collected so the same tree crowns can be compared through time.

## Goal

Each visit should produce a photo set that can become an orthomosaic aligned closely enough for crown detection, crown tracking, and crown-level phenology measurements. Small field differences are often amplified later. A slightly different footprint, a lower overlap, harsh shadows, or a wind-disturbed canopy can make the orthomosaic harder to align and can reduce the quality of crown matching.

## Before The Flight

Reuse the saved mission whenever possible. Avoid redrawing the polygon or changing flight settings unless there is a clear reason. If something must change, record the change in the field notes for that date.

Check these items before flying:

1. The mission footprint covers the full monitoring area plus a small buffer.
2. The altitude is the same as the previous monitoring flights for that site.
3. Forward and side overlap are unchanged.
4. Camera angle is nadir unless a specific campaign requires otherwise.
5. The planned speed is not unusually fast for the light and wind conditions.
6. Batteries, controller storage, and SD card capacity are sufficient for the whole mission.
7. The drone clock, controller clock, and field notebook date agree.

For LHC, SIT, and Sanjay Van, keep site and date naming consistent from collection onward. The later scripts assume that dates and site names can be interpreted reliably, so it is worth fixing names early instead of cleaning them repeatedly later.

## During The Flight

Try to keep the following conditions as similar as practical from one visit to the next:

1. Flight footprint.
2. Altitude.
3. Overlap.
4. Speed.
5. Time of day.
6. Sun and shadow conditions.
7. Wind conditions.

Do not discard notes about imperfect flights. If the flight was interrupted, if a battery swap changed the timing, if the wind picked up, or if part of the area was reflown, write it down. These notes are useful later when one orthomosaic or crown-detection output looks different from the rest of the time series.

## After The Flight

Keep each flight date in its own image folder. Do not mix dates, sites, spots, or partial reflights unless the folder name and notes make that explicit.

Recommended records to keep:

1. Raw image folder.
2. Mission file or saved route.
3. Flight logs.
4. Notes about weather, shadows, interruptions, or unusual site conditions.
5. Any manual changes to the mission settings.

The orthomosaic scripts in `misc/ODM/` expect a folder of images from one flight/date. The single-date folder is passed to `misc/ODM/make_om.ps1` or to one of the batch helpers such as `misc/ODM/lhc_sit_make_oms.ps1` and `misc/ODM/sv_make_oms.ps1`.

## Naming And Handoff

Use names that keep the site and date visible. For cleaned analysis orthomosaics, the current conventions are documented in `misc/ODM/ODM_QUICKSTART.md`:

1. `input/input_om_lhc`: `lhc_DD-MM-YY.tif`
2. `input/input_om_sit`: `sit_DD-MM-YY.tif`
3. `input/input_om_sv/spot_X`: `sv_spotX_DD-MM-YY.tif`

Those names are not required at raw-image collection time, but the raw folders should be clear enough that the orthomosaic and pipeline stages can map them into those cleaned names without guessing.

## Why This Matters

The rest of the project depends on temporal comparability. If the imagery is collected consistently, the downstream steps can focus on biological change. If the imagery is inconsistent, the pipeline may spend its effort explaining technical differences: alignment drift, missing crowns, crown boundary shifts, or noisy phenology signals.

## Troubleshooting

1. If one date later produces a poor orthomosaic, check the field notes for wind, shadow, interrupted flight paths, or mixed image folders.
2. If crown detection is weak on one date, compare the image sharpness, shadows, and overlap against stronger dates.
3. If tracking fails between dates, check whether the flight footprint or altitude changed enough to alter the visible crown geometry.
4. If the date is uncertain, preserve that uncertainty in the filename or notes rather than inventing a date. The SIT workflow has used `dateUnknown` or `dateNotConfirmed` naming for this kind of case.
