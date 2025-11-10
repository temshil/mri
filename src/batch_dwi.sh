#!/bin/bash

find /temshil/data/processed -type f -path "*/dwi/*dwi.nii.gz" | while read -r dwi; do
    echo "Processing: $dwi"
    python3 /temshil/src/dwi.py --in_path "$dwi"
done