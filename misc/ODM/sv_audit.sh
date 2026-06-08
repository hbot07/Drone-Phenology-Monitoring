#!/bin/bash
mount -t nfs -o nolock,vers=3 cohesityiitd.cse.iitd.ac.in:/Aaditeswar /mnt/filer 2>/dev/null
echo "=== raw_data/images_raw ==="
for spot in /mnt/filer/Sanjay_Van_Data/raw_data/images_raw/*/; do
  sname=$(basename "$spot")
  for d in "$spot"*/; do
    dname=$(basename "$d")
    cnt=$(ls "$d" 2>/dev/null | wc -l)
    echo "  $sname / $dname : $cnt files"
  done
done
echo ""
echo "=== ortho_images ==="
for spot in /mnt/filer/Sanjay_Van_Data/ortho_images/*/; do
  sname=$(basename "$spot")
  for d in "$spot"*/; do
    dname=$(basename "$d")
    tif=$(find "$d" -name "odm_orthophoto.tif" 2>/dev/null | head -1)
    if [ -n "$tif" ]; then
      sz=$(du -m "$tif" | cut -f1)
      echo "  $sname / $dname : orthophoto ${sz}MB"
    else
      echo "  $sname / $dname : no orthophoto"
    fi
  done
done
