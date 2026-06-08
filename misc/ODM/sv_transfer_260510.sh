#!/bin/bash
set -e
NFS="cohesityiitd.cse.iitd.ac.in:/Aaditeswar"
FILER="/mnt/filer"

echo "=== Mounting NFS ==="
mkdir -p $FILER
mount -t nfs -o nolock,vers=3 $NFS $FILER
echo "Mounted OK"

# --- Raw data 260510 ---
echo ""
echo "=== Raw data spot_1/260510 (111 files) ==="
mkdir -p $FILER/Sanjay_Van_Data/raw_data/images_raw/spot_1/260510
rsync -avh --progress --partial /raw_spot1/ $FILER/Sanjay_Van_Data/raw_data/images_raw/spot_1/260510/

echo ""
echo "=== Raw data spot_2/260510 (86 files) ==="
mkdir -p $FILER/Sanjay_Van_Data/raw_data/images_raw/spot_2/260510
rsync -avh --progress --partial /raw_spot2/ $FILER/Sanjay_Van_Data/raw_data/images_raw/spot_2/260510/

echo ""
echo "=== Raw data spot_3/260510 (60 files) ==="
mkdir -p $FILER/Sanjay_Van_Data/raw_data/images_raw/spot_3/260510
rsync -avh --progress --partial /raw_spot3/ $FILER/Sanjay_Van_Data/raw_data/images_raw/spot_3/260510/

echo ""
echo "=== Raw data spot_4/260510 (60 files) ==="
mkdir -p $FILER/Sanjay_Van_Data/raw_data/images_raw/spot_4/260510
rsync -avh --progress --partial /raw_spot4/ $FILER/Sanjay_Van_Data/raw_data/images_raw/spot_4/260510/

# --- Orthomosaics 260510 ---
echo ""
echo "=== Orthomosaic spot_1/260510 ==="
mkdir -p $FILER/Sanjay_Van_Data/ortho_images/spot_1/260510
rsync -avh --progress --partial /om_spot1/ $FILER/Sanjay_Van_Data/ortho_images/spot_1/260510/

echo ""
echo "=== Orthomosaic spot_2/260510 ==="
mkdir -p $FILER/Sanjay_Van_Data/ortho_images/spot_2/260510
rsync -avh --progress --partial /om_spot2/ $FILER/Sanjay_Van_Data/ortho_images/spot_2/260510/

echo ""
echo "=== Orthomosaic spot_3/260510 ==="
mkdir -p $FILER/Sanjay_Van_Data/ortho_images/spot_3/260510
rsync -avh --progress --partial /om_spot3/ $FILER/Sanjay_Van_Data/ortho_images/spot_3/260510/

echo ""
echo "=== Orthomosaic spot_4/260510 ==="
mkdir -p $FILER/Sanjay_Van_Data/ortho_images/spot_4/260510
rsync -avh --progress --partial /om_spot4/ $FILER/Sanjay_Van_Data/ortho_images/spot_4/260510/

echo ""
echo "=== ALL DONE ==="
