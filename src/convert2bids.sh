#!/bin/bash

parent_dir="/temshil/data/raw"

for dir in "$parent_dir"/*/; do
    [ -d "$dir" ] || continue
    python /temshil/src/bidsconverter.py --in_path $dir --anat_name EPIref --fmri_name SE_EPI_Scan --dwi_name DTI_EPI

done