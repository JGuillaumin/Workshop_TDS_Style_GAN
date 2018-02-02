#!/usr/bin/env bash

cd /mnt

# generator
sudo gdrive download 1CH6QWfmMJ1WtHTzjaNo84z8d9QS0BC3D
# cat dataset
sudo gdrive download 1IlZzzbwCSPq03q4XgG0p5wrHFbq5Zenf

sudo mkdir s0
sudo mkdir s1
sudo mkdir s2
sudo mkdir s3

sudo cp generator.h5 s0/
sudo cp generator.h5 s1/
sudo cp generator.h5 s2/
sudo cp generator.h5 s3/


sudo tar -C s0/ -xzf cat_dataset.tar.gz.tar.gz
sudo tar -C s1/ -xzf cat_dataset.tar.gz.tar.gz
sudo tar -C s2/ -xzf cat_dataset.tar.gz.tar.gz
sudo tar -C s3/ -xzf cat_dataset.tar.gz.tar.gz

sudo cp -rp ~/Workshop_TDS_Style_GAN/* /mnt/s0/
sudo cp -rp ~/Workshop_TDS_Style_GAN/* /mnt/s1/
sudo cp -rp ~/Workshop_TDS_Style_GAN/* /mnt/s2/
sudo cp -rp ~/Workshop_TDS_Style_GAN/* /mnt/s3/


cd ~/Workshop_TDS_Style_GAN/

screen -dm -S sess_jupyter bash -c "./_start.sh; exec sh"
exit

