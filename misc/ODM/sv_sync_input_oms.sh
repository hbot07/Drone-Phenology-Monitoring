#!/bin/bash
# sv_sync_input_oms.sh
# Download all Sanjay Van OMs from filer into /sv_out (mounted to input_om_sv locally).
# Naming convention:
#   New-style date folders (YYMMDD): sv_spotX_DD-MM-YY.tif
#   Old bundle folders (date in name like DD-MM-YY): sv_spotX_DD-MM-YY.tif
#   Old bundle without date (e.g. images_raw_spot_4_investigation): sv_spotX_00-00-0000.tif
# Skips if destination file already exists.

set -euo pipefail

NFS_SERVER="cohesityiitd.cse.iitd.ac.in:/Aaditeswar"
MOUNT_PT="/mnt/filer"
SV_FILER="$MOUNT_PT/Sanjay_Van_Data/ortho_images"
DEST="/sv_out"

echo "=== Mounting NFS ==="
mount -t nfs -o vers=3,nolock "$NFS_SERVER" "$MOUNT_PT"
echo "Mounted OK"
echo ""

copy_om() {
    local src="$1"
    local dest="$2"
    local label="$3"
    if [ -f "$dest" ]; then
        echo "SKIP $label -- already exists"
    elif [ ! -f "$src" ]; then
        echo "MISSING $label -- source not found: $src"
    else
        echo "=== Downloading $label ==="
        rsync -ah --progress "$src" "$dest"
        echo "Done: $label -> $(basename $dest)"
    fi
}

for spot in 1 2 3 4; do
    spot_dir="$SV_FILER/spot_${spot}"
    echo ""
    echo "======== spot_${spot} ========"

    for entry in "$spot_dir"/*/; do
        name=$(basename "$entry")

        # ── New-style: 6-digit YYMMDD folder ──────────────────────────────────
        if echo "$name" | grep -qE '^[0-9]{6}$'; then
            yy="${name:0:2}"
            mm="${name:2:2}"
            dd="${name:4:2}"
            date_str="${dd}-${mm}-${yy}"
            dest_file="$DEST/sv_spot${spot}_${date_str}.tif"
            src_tif="$entry/odm_orthophoto.tif"
            copy_om "$src_tif" "$dest_file" "spot${spot} ${date_str} (new-style)"

        # ── Old bundle with date in name DD-MM-YY ─────────────────────────────
        elif echo "$name" | grep -qE '[0-9]{2}-[0-9]{2}-[0-9]{2}'; then
            date_str=$(echo "$name" | grep -oE '[0-9]{2}-[0-9]{2}-[0-9]{2}' | head -1)
            dest_file="$DEST/sv_spot${spot}_${date_str}.tif"
            src_tif="$entry/odm_orthophoto/odm_orthophoto.tif"
            copy_om "$src_tif" "$dest_file" "spot${spot} ${date_str} (old bundle)"

        # ── Old bundle without parseable date ─────────────────────────────────
        else
            dest_file="$DEST/sv_spot${spot}_00-00-0000.tif"
            src_tif="$entry/odm_orthophoto/odm_orthophoto.tif"
            # fallback: try root level tif
            if [ ! -f "$src_tif" ]; then
                src_tif="$entry/odm_orthophoto.tif"
            fi
            copy_om "$src_tif" "$dest_file" "spot${spot} unknown-date ($name)"
        fi
    done
done

echo ""
echo "=== All SV OMs synced ==="
echo "Files in $DEST:"
ls -lh "$DEST/"
