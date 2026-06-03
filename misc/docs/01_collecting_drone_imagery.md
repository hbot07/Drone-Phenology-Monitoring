# Collecting Drone Imagery

For this project, use Paul Pop's manual directly. Our process is the same.

Main reference:

- https://github.com/paulvpop/drone-mapping-end-to-end-workflow/blob/main/user_manual.md

Use that manual for:

1. Creating and exporting flight paths.
2. Uploading flight paths to the drone or controller.
3. Flying the drone.
4. Handling multi-battery missions.
5. Exporting logs.
6. The general WebODM-side orthomosaic workflow.

For this project, the important point is consistency across dates. Since the same area is flown repeatedly for phenology monitoring, try to keep the following as similar as possible from one visit to the next:

1. Flight footprint.
2. Altitude.
3. Overlap.
4. Speed.
5. Time of day, where practical.
6. Weather conditions, especially wind and shadow conditions.

The reason is simple: small inconsistencies at image collection stage show up later as alignment drift, weaker crown matching, and noisier phenology signals.

As a practical rule, keep the saved mission and reuse it rather than redrawing it each time. If you need to change anything, note it down clearly along with the date.

Also keep the mission logs, exported flight files, and any notes about interruptions, battery swaps, or unusual conditions. Those records become useful later if one date looks odd during orthomosaic creation or crown tracking.
