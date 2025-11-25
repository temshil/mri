import nibabel as nii
import numpy as np
import nrrd        
import requests
from pathlib import Path 
import pandas as pd
import os
import glob
import re
import shutil

labels = pd.read_csv("/temshil/data/lib/atlas_modifications/labels.csv")

url  = "https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/average_template/average_template_50.nrrd"
nrrd_path = Path('/temshil/lib/atlas_modifications/average_template_50.nrrd')

with requests.get(url, stream=True) as r:
    r.raise_for_status()
    with open(nrrd_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
data, header = nrrd.read(nrrd_path)
data_swapped = np.transpose(data, (2,1,0))
data_flipped = np.flip(data_swapped, axis=2) 
data_flipped[data_flipped==1]=0
img = nii.Nifti1Image(data_flipped, affine=np.eye(4))
img_header = img.header.copy()
img_header.set_xyzt_units('mm', 'sec')
img_affine=np.eye(4)
img_affine[:3, :3] *= .05  
img = nii.Nifti1Image(data_flipped, affine=img_affine, header=img_header)
nii_out_path = os.path.join('/temshil/data/lib/', os.path.basename(nrrd_path).split('.')[0]+'.nii.gz')
nii.save(img, nii_out_path)
print("Saved:", nii_out_path)

for str_id in labels['ID']:
    url  = f"https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/structure_masks/structure_masks_50/structure_{str_id}.nrrd"
    nrrd_path = Path(f'/temshil/lib/atlas_modifications/str_nrrd/structure_{str_id}.nrrd')

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(nrrd_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    data, header = nrrd.read(nrrd_path)
    data_swapped = np.transpose(data, (2,1,0))
    data_flipped = np.flip(data_swapped, axis=2) 
    img = nii.Nifti1Image(data_flipped, affine=np.eye(4))
    nii_out_path = os.path.join('/temshil/lib/atlas_modifications/str_nifti', os.path.basename(nrrd_path).split('.')[0]+'.nii.gz')
    nii.save(img, nii_out_path)
    print("Saved:", nii_out_path)

str_nifti_list = glob.glob('/temshil/data/atlas_modifications/str_nifti/*', recursive=True)

shutil.copy(str_nifti_list[1],'/temshil/lib/atlas_modifications/atlas.nii.gz')

pref = [184, 31, 44, 972]
orb = [714, 95, 583]
olf = [698,942]
peri = [541, 922, 895]
amyg = [131,295,319,780,278]
striat = [485,493,275]
wmcsf = [73,1009]

atlas_img = nii.load('/temshil/lib/atlas.nii.gz')
atlas_data = atlas_img.get_fdata()

for str_nifti in str_nifti_list:
    str_img = nii.load(str_nifti)
    str_data = str_img.get_fdata()
    label_id = int(re.search(r'structure_(\d+)', str_nifti).group(1))
    if label_id in pref:
        label_id = 31
    elif label_id in orb:
        label_id = 714
    elif label_id in olf:
        label_id = 698
    elif label_id in peri:
        label_id = 541
    elif label_id in amyg:
        label_id = 131
    elif label_id in striat:
        label_id = 485
    elif label_id in wmcsf:
        label_id = 200
    atlas_data[str_data > 0] = label_id
   

x_size, y_size, z_size = atlas_data.shape

midline = x_size // 2 

left_hemi = atlas_data[:midline, :, :]
right_hemi = atlas_data[midline:, :, :]

mask = (right_hemi > 0) & (right_hemi != 200)
right_hemi[mask] += 2000

combined = np.zeros_like(atlas_data)
combined[:midline, :, :] = left_hemi
combined[midline:, :, :] = right_hemi

atlas_img_upd = nii.Nifti1Image(atlas_data, affine=atlas_img.affine)
nii.save(atlas_img_upd, '/temshil/atlas/atlas.nii.gz')
   

