#!/bin/bash

find /temshil/data/processed -type f -path "*/dwi/*_dwi.nii.gz" ! -path "*/dwi/raw/*" | while read -r dwi; do
    echo "Processing: $dwi"
    python3 /temshil/src/dwi.py --in_path "$dwi"
done