#!/bin/bash
# lhc_sit_upload_oms.sh
# Upload newly created LHC and SIT orthomosaics to the CSE NFS filer.
# Mounts the processed ODM output dirs at /lhc_oms and /sit_oms (read-only).
# Filer destination:
#   LHC: IITD_Drone_data/orthomosaics/LHC/odm_orthophoto<date>.tif
#   SIT: IITD_Drone_data/orthomosaics/SIT/odm_orthophoto<date>.tif
# Skips if the file already exists on the filer.

set -euo pipefail

NFS_SERVER="cohesityiitd.cse.iitd.ac.in:/Aaditeswar"
MOUNT_PT="/mnt/filer"
LHC_LOCAL="/lhc_oms"
SIT_LOCAL="/sit_oms"

echo "=== Mounting NFS ==="
mount -t nfs -o vers=3,nolock "$NFS_SERVER" "$MOUNT_PT"
echo "Mounted OK"

LHC_FILER="$MOUNT_PT/IITD_Drone_data/orthomosaics/LHC"
SIT_FILER="$MOUNT_PT/IITD_Drone_data/orthomosaics/SIT"
mkdir -p "$LHC_FILER" "$SIT_FILER"

upload_om() {
    local src="$1"
    local dest="$2"
    local label="$3"
    if [ -f "$dest" ]; then
        echo "SKIP $label -- already on filer"
    elif [ ! -f "$src" ]; then
        echo "MISSING $label -- source tif not found: $src"
    else
        echo "=== Uploading $label ==="
        rsync -ah --progress "$src" "$dest"
        echo "Done: $label"
    fi
}

# LHC
upload_om "$LHC_LOCAL/LHC_16_03_26/odm_orthophoto.tif"  "$LHC_FILER/odm_orthophoto16_03_26.tif"  "LHC 16_03_26"

# SIT
upload_om "$SIT_LOCAL/SIT_9_11_25/odm_orthophoto.tif"   "$SIT_FILER/odm_orthophoto9_11_25.tif"   "SIT 9_11_25"
upload_om "$SIT_LOCAL/SIT_20_11_25/odm_orthophoto.tif"  "$SIT_FILER/odm_orthophoto20_11_25.tif"  "SIT 20_11_25"
upload_om "$SIT_LOCAL/SIT_28_01_26/odm_orthophoto.tif"  "$SIT_FILER/odm_orthophoto28_01_26.tif"  "SIT 28_01_26"
upload_om "$SIT_LOCAL/SIT_29_1_26/odm_orthophoto.tif"   "$SIT_FILER/odm_orthophoto29_1_26.tif"   "SIT 29_1_26"
upload_om "$SIT_LOCAL/SIT_20_02_26/odm_orthophoto.tif"  "$SIT_FILER/odm_orthophoto20_02_26.tif"  "SIT 20_02_26"
upload_om "$SIT_LOCAL/SIT_16_03_26/odm_orthophoto.tif"  "$SIT_FILER/odm_orthophoto16_03_26.tif"  "SIT 16_03_26"

echo ""
echo "=== All uploads complete ==="
