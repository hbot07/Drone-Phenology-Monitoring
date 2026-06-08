#!/bin/bash
# sv_upload_oms.sh
# Upload all Sanjay Van orthomosaics from E:\Sanjay_van_data\Processed\ODM to filer
# and update sanjay_van_data.csv
# Mounts: /sv_oms = E:\Sanjay_van_data\Processed\ODM (read-only)

set -e
NFS="cohesityiitd.cse.iitd.ac.in:/Aaditeswar"
FILER="/mnt/filer"

echo "=== Mounting NFS ==="
mkdir -p $FILER
mount -t nfs -o nolock,vers=3 $NFS $FILER
echo "Mounted OK"
echo ""

# Upload each spot/date orthomosaic found in /sv_oms
for spot_date_dir in /sv_oms/*/; do
  name=$(basename "$spot_date_dir")
  tif="$spot_date_dir/odm_orthophoto.tif"

  if [ ! -f "$tif" ]; then
    echo "SKIP $name — no odm_orthophoto.tif found"
    continue
  fi

  # Parse spot and date from name like "spot_1_250911"
  spot=$(echo "$name" | sed 's/_[0-9]*$//')
  date=$(echo "$name" | grep -oP '[0-9]{6}$')

  dest="$FILER/Sanjay_Van_Data/ortho_images/$spot/$date"

  # Check if already uploaded
  if [ -f "$dest/odm_orthophoto.tif" ]; then
    echo "SKIP $name — already on filer"
    continue
  fi

  echo "=== Uploading $name ($spot / $date) ==="
  mkdir -p "$dest"
  rsync -avh --progress --partial "$tif" "$dest/"
  echo "Done: $name"
  echo ""
done

echo "=== All uploads complete ==="
