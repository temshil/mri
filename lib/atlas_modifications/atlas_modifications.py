import nibabel as nib
import numpy as np
import pandas as pd

ara_atlas = nib.load('path/to/ARA_annotationR+2000.nii.gz')
ara_data = ara_atlas.get_fdata()

ara_annot = pd.read_csv(
    "path/to/atlas_modifications/ARA_annotationR+2000.nii.txt",
    sep="\t",
    header=None,
    dtype=str
)

csf_names_path = 'path/to/atlas_modifications/csf.txt'
csf_names_arr = [line.strip().replace(" ", "_") for line in open(csf_names_path, "r", encoding="utf-8")]

csf_labels = []
for line in csf_names_arr:
    l = ara_annot[ara_annot[1].str.contains(fr"^L_.*{line}", na=False, case=False)]
    r = ara_annot[ara_annot[1].str.contains(fr"^R_.*{line}", na=False, case=False)]
    csf_labels.append(float(l.iloc[0,0]))
    csf_labels.append(float(r.iloc[0,0]))

csf_mask = np.isin(ara_data, csf_labels)

mod_atlas = nib.load('path/to/atlas_modifications/annoVolume+2000_rsfMRI.nii.gz')
mod_atlas_data = mod_atlas.get_fdata()

l_cor_sub  = np.isin(mod_atlas_data, 703)
l_str_amyg = np.isin(mod_atlas_data, 278)
r_cor_sub  = np.isin(mod_atlas_data, 2703)
r_str_amyg = np.isin(mod_atlas_data, 2278)

l_cor_cal  = np.isin(mod_atlas_data, 776)
l_corspin_tr  = np.isin(mod_atlas_data, 784)
l_thamrel = np.isin(mod_atlas_data, 896)
l_med_for_bun = np.isin(mod_atlas_data, 991)
l_extrapyr_fib = np.isin(mod_atlas_data, 1000)

r_cor_cal  = np.isin(mod_atlas_data, 2776)
r_corspin_tr  = np.isin(mod_atlas_data, 2784)
r_thamrel = np.isin(mod_atlas_data, 2896)
r_med_for_bun = np.isin(mod_atlas_data, 2991)
r_extrapyr_fib = np.isin(mod_atlas_data, 3000)

l_claustrum  = np.isin(mod_atlas_data, 583)
r_claustrum  = np.isin(mod_atlas_data, 2583)

mod_atlas_data[csf_mask]  = 200

mod_atlas_data[l_cor_cal]  = 200
mod_atlas_data[l_corspin_tr]  = 0
mod_atlas_data[l_thamrel]  = 0
mod_atlas_data[l_med_for_bun]  = 0
mod_atlas_data[l_extrapyr_fib]  = 0

mod_atlas_data[r_cor_cal]  = 200
mod_atlas_data[r_corspin_tr]  = 0
mod_atlas_data[r_thamrel]  = 0
mod_atlas_data[r_med_for_bun]  = 0
mod_atlas_data[r_extrapyr_fib]  = 0


mod_atlas_data[l_claustrum]  = 95
mod_atlas_data[r_claustrum]  = 2095

mod_atlas_data[l_cor_sub]   = 191
mod_atlas_data[l_str_amyg]  = 191
mod_atlas_data[r_cor_sub]   = 2191
mod_atlas_data[r_str_amyg]  = 2191

new_mod_atlas = nib.Nifti1Image(mod_atlas_data, mod_atlas.affine, mod_atlas.header)
nib.save(new_mod_atlas, 'path/to/atlas.nii.gz')
