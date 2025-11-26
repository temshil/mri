#!/bin/bash

find /temshil/data/processed -type f -path "*/func/*_bold.nii.gz" ! -path "*/func/raw/*" | while read -r fmri; do
    echo "Processing: $fmri"
    python3 /temshil/src/fmri.py --in_path "$fmri" --part 2 --rm
done