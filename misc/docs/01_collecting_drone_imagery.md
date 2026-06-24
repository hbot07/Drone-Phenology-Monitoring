# Collect Drone Imagery

Main drone-mapping reference:

- https://github.com/paulvpop/drone-mapping-end-to-end-workflow/blob/main/user_manual.md

For repeated monitoring, the priority is consistency: same footprint, altitude, camera angle, overlap, and time-of-day window whenever possible.

## Output

For each site/date, produce one raw image folder with:

1. Original drone images.
2. Mission file or route information.
3. Flight logs if available.
4. Short field notes.

Downstream processing expects one checked orthomosaic per site/date.

## Plan A Site

Record these before the first flight:

1. Site code.
2. Boundary or flight polygon.
3. Spots/subplots if the site is split.
4. Flight altitude.
5. Forward and side overlap.
6. Camera angle, usually nadir.
7. Revisit interval.
8. Preferred flight time.
9. Restrictions, permissions, obstacles, and no-fly areas.
10. Storage location for images, missions, logs, and orthomosaics.

Use short codes such as `site_a`, `north_block`, or `forest_spot1`.

## Build A Repeatable Flight Path

Create one saved mission per site and reuse it. Avoid redrawing the polygon each visit.

Check that:

1. The polygon covers the monitoring area plus a buffer.
2. Overlap is high enough for tree canopy reconstruction.
3. Ground resolution is good enough to see individual crowns.
4. The drone can finish safely with available batteries.
5. The route avoids obstacles and restricted areas.
6. Takeoff and landing are practical for repeat visits.

## Flight Settings

| Setting | Use |
|---|---|
| Camera angle | Nadir unless the campaign specifies otherwise. |
| Footprint | Same saved mission each visit. |
| Altitude | Same altitude each visit. |
| Forward overlap | High enough for canopy reconstruction. |
| Side overlap | High enough to avoid gaps and striping. |
| Speed | Conservative enough to avoid blur. |
| Time of day | Similar window when possible. |
| Image format | Original files from the drone. |

## Before Flying

Check:

1. Saved mission is available.
2. Batteries are charged.
3. SD card and controller storage have space.
4. Drone, controller, and notebook dates match.
5. Weather and wind are acceptable.
6. Permissions and access are clear.
7. Folder naming plan is ready.

If altitude, overlap, camera angle, or footprint changes, write it down.

## During Flight

Record anything that may affect the orthomosaic:

1. Pauses or interruptions.
2. Battery swaps.
3. Reflown strips.
4. Strong wind.
5. Harsh shadows or changing clouds.
6. People, vehicles, or moving objects.
7. Missing coverage.
8. Camera-setting changes.

## After Flight

Keep raw folders unmixed by site/date.

Example:

```text
raw_drone_images/
  site_a/
    2026_01_15/
      images/
      notes.txt
      mission_file/
      logs/
    2026_01_29/
      images/
      notes.txt
      mission_file/
      logs/
```

## Field Notes

```text
site: site_a
date: 2026-01-15
operator: <name>
mission file: <mission name or ID>
flight start/end: <time range>
altitude: <value>
forward overlap: <value>
side overlap: <value>
camera angle: nadir
weather/light: <clear/cloudy/shadow notes>
wind: <calm/moderate/strong>
interruptions: <none or details>
reflown sections: <none or details>
coverage concerns: <none or details>
comments: <anything unusual>
```

## Naming

Raw folders can use readable dates such as:

```text
2026_01_15
```

Clean orthomosaics should use one stable convention:

```text
<site>_DD-MM-YY.tif
<site>_DD-MM-YY_dateNotConfirmed.tif
<site>_spot<id>_DD-MM-YY.tif
```

Examples:

```text
site_a_15-01-26.tif
site_a_29-01-26.tif
forest_spot1_10-05-26.tif
```

If the date is uncertain, mark it instead of inventing one.

## Handoff

For orthomosaic processing, provide:

1. Raw image folder.
2. Mission/log files.
3. Field notes.
4. Site/date naming convention.
5. Status: `usable`, `check`, `preview_only`, or `reject`.

## Quick Field Checklist

Before leaving:

1. Image count looks reasonable.
2. Images are copied or backed up.
3. Mission/log files are saved.
4. Notes are written.
5. Date, site, and coverage uncertainty are marked.

## Troubleshooting

1. Poor orthomosaic: check notes for wind, shadows, interruptions, partial reflights, or mixed folders.
2. Weak crown detection: check blur, exposure, shadows, and overlap.
3. Tracking failure: check footprint, altitude, and crown-shape consistency.
4. Missing edge crowns: check whether the flight buffer was too small.
5. Uncertain date: use `dateNotConfirmed` and document it.
