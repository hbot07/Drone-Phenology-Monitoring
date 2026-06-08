#!/bin/bash
mount -t nfs -o nolock,vers=3 cohesityiitd.cse.iitd.ac.in:/Aaditeswar /mnt/filer 2>/dev/null
for d in SIT LHC; do
  echo "=== Raw_data/$d ==="
  for f in /mnt/filer/IITD_Drone_data/Raw_data/$d/*/; do
    cnt=$(ls "$f" 2>/dev/null | wc -l)
    echo "  $(basename $f): $cnt files"
  done
done
echo "=== orthomosaics/SIT ==="
ls -lh /mnt/filer/IITD_Drone_data/orthomosaics/SIT/
echo "=== orthomosaics/LHC ==="
ls -lh /mnt/filer/IITD_Drone_data/orthomosaics/LHC/
