#!/bin/bash

find /temshil/data/processed -type f -path "*/func/*bold.nii.gz" | while read -r fmri; do
    echo "Processing: $fmri"
    python3 /temshil/src/fmri.py --in_path "$fmri" --part 2
done