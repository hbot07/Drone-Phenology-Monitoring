#!/bin/bash
# Batch transfer script — run inside nfs_mount_img Docker container
# Mounts:
#   /raw_sit  -> D:\Gaurav2\IIT_Delhi_Drone_Data\Raw_data\SIT
#   /raw_lhc  -> D:\Gaurav2\IIT_Delhi_Drone_Data\Raw_data\LHC
#   /om_sit   -> D:\Gaurav2\IIT_Delhi_Drone_Data\Processed\ODM\sit_processed
#   /om_lhc   -> D:\Gaurav2\IIT_Delhi_Drone_Data\Processed\ODM\lhc_processed
#   /om_ws    -> D:\Drone_Phenology_Monitoring\ODM_Output

set -e

echo "==== Mounting NFS ===="
mount -t nfs -o nolock,vers=3 cohesityiitd.cse.iitd.ac.in:/Aaditeswar /mnt/filer
echo "NFS mounted OK"

echo ""
echo "==== SIT Raw data ===="

echo "--- SIT 15_04_26 ---"
mkdir -p /mnt/filer/IITD_Drone_data/Raw_data/SIT/15_04_26
rsync -avh --progress --partial /raw_sit/15_04_26/ /mnt/filer/IITD_Drone_data/Raw_data/SIT/15_04_26/

echo "--- SIT 16_03_26 (from local 16_03_26_80m) ---"
mkdir -p /mnt/filer/IITD_Drone_data/Raw_data/SIT/16_03_26
rsync -avh --progress --partial /raw_sit/16_03_26_80m/ /mnt/filer/IITD_Drone_data/Raw_data/SIT/16_03_26/

echo "--- SIT 17_03_26 (from local 17_03_26_50m — 50m experimental run) ---"
mkdir -p /mnt/filer/IITD_Drone_data/Raw_data/SIT/17_03_26
rsync -avh --progress --partial /raw_sit/17_03_26_50m/ /mnt/filer/IITD_Drone_data/Raw_data/SIT/17_03_26/

echo ""
echo "==== LHC Raw data ===="

echo "--- LHC 09_05_26 ---"
mkdir -p /mnt/filer/IITD_Drone_data/Raw_data/LHC/09_05_26
rsync -avh --progress --partial /raw_lhc/09_05_26/ /mnt/filer/IITD_Drone_data/Raw_data/LHC/09_05_26/

echo "--- LHC 15_04_26 ---"
mkdir -p /mnt/filer/IITD_Drone_data/Raw_data/LHC/15_04_26
rsync -avh --progress --partial /raw_lhc/15_04_26/ /mnt/filer/IITD_Drone_data/Raw_data/LHC/15_04_26/

echo "--- LHC 16_03_26 ---"
mkdir -p /mnt/filer/IITD_Drone_data/Raw_data/LHC/16_03_26
rsync -avh --progress --partial /raw_lhc/16_03_26/ /mnt/filer/IITD_Drone_data/Raw_data/LHC/16_03_26/

echo ""
echo "==== SIT Orthomosaics ===="

echo "--- SIT 07_03_26 ---"
rsync -avh --progress /om_ws/07_03_26/odm_orthophoto.tif /mnt/filer/IITD_Drone_data/orthomosaics/SIT/odm_orthophoto07_03_26.tif

echo "--- SIT 15_04_26 ---"
rsync -avh --progress /om_sit/SIT_15_04_26/sit_15_04_26.tif /mnt/filer/IITD_Drone_data/orthomosaics/SIT/odm_orthophoto15_04_26.tif

echo "--- SIT 09_05_26 ---"
rsync -avh --progress /om_sit/SIT_09_05_26/odm_orthophoto.tif /mnt/filer/IITD_Drone_data/orthomosaics/SIT/odm_orthophoto09_05_26.tif

echo ""
echo "==== LHC Orthomosaics ===="

echo "--- LHC 07_03_26 ---"
rsync -avh --progress /om_ws/LHC_07_03_26/odm_orthophoto.tif /mnt/filer/IITD_Drone_data/orthomosaics/LHC/odm_orthophoto07_03_26.tif

echo "--- LHC 15_04_26 ---"
rsync -avh --progress /om_lhc/LHC_15_04_26/odm_orthophoto.tif /mnt/filer/IITD_Drone_data/orthomosaics/LHC/odm_orthophoto15_04_26.tif

echo "--- LHC 09_05_26 ---"
rsync -avh --progress /om_lhc/LHC_09_05_26/odm_orthophoto.tif /mnt/filer/IITD_Drone_data/orthomosaics/LHC/odm_orthophoto09_05_26.tif

echo ""
echo "==== All transfers complete ===="
