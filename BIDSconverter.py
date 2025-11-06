import glob
import os
import re
import shutil
import nibabel as nii
import argparse
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(
        description="Move converted files to BIDS.")
    parser.add_argument('--in_path', type=str,
                        help='Input path')
    parser.add_argument('--anat_name', type=str,
                        help='Name of the anatomical sequence')
    parser.add_argument('--fmri_name', type=str,
                        help='Name of the fMRI sequence')
    parser.add_argument('--dwi_name', type=str,
                        help='Name of the DTI sequence')
    return parser.parse_args()

def main():
    args = parse_args()
    in_path = args.in_path
    anat_name = "*" + args.anat_name + "*"
    fmri_name =  "*" + args.fmri_name + "*"
    dwi_name =  "*" + args.dwi_name + "*"
    
    os.chdir(in_path)
    
    subprocess.run(['brkraw', 'tonii', '-b',
                    in_path], check=True)
    
    with open(os.path.join(in_path,"subject"), "r") as file:
        content = file.read()
        lines = content.split("##")
        for line in lines:
            if "$SUBJECT_study_name" in line:
                match = re.search(r'<([A-Za-z0-9]+)_([a-z]+)>', line)
                if match:
                    sub = match.group(1) if match.group(1) else 'sub'
                    ses = match.group(2) if match.group(2) else 'ses'
                else:
                    sub = 'sub'
                    ses = 'ses'
                break
    
    os.makedirs(os.path.join(in_path,f"sub-{sub}",f"ses-{ses}","anat"), exist_ok=True)
    os.makedirs(os.path.join(in_path,f"sub-{sub}",f"ses-{ses}","func"), exist_ok=True)
    os.makedirs(os.path.join(in_path,f"sub-{sub}",f"ses-{ses}","func_rev"), exist_ok=True)
    os.makedirs(os.path.join(in_path,f"sub-{sub}",f"ses-{ses}","dwi"), exist_ok=True)
    os.makedirs(os.path.join(in_path,f"sub-{sub}",f"ses-{ses}","dwi_rev"), exist_ok=True)
    
    
    anat_file_list = glob.glob(os.path.join(in_path, '**', anat_name), recursive=True)
    
    if len(anat_file_list) != 0:
    
        for file in anat_file_list:
            if "json" in file:
                anat_json_copy = os.path.join(in_path, f"sub-{sub}",f"ses-{ses}","anat",f"sub-{sub}_ses-{ses}_T2w.json")
                shutil.copy(file,anat_json_copy)
            elif "nii.gz" in file:
                anat_nii_copy = os.path.join(in_path, f"sub-{sub}",f"ses-{ses}","anat",f"sub-{sub}_ses-{ses}_T2w.nii.gz")
                shutil.copy(file,anat_nii_copy)
    else:
        print('anat data is not found')
    
    fmri_file_list = glob.glob(os.path.join(in_path, '**', fmri_name), recursive=True)
    
    if len(fmri_file_list) != 0:
    
        for file in fmri_file_list:
                if "nii.gz" in file:
                    imgTemp = nii.load(file)
                    if len(imgTemp.shape)==3:
                        if imgTemp.shape == (64,64,30):
                            fmri_rev_nii_copy = os.path.join(in_path, f"sub-{sub}",f"ses-{ses}","func_rev",f"sub-{sub}_ses-{ses}_bold_rev.nii.gz")
                            shutil.copy(file,fmri_rev_nii_copy)
                            fmri_rev_json_copy = os.path.join(in_path, f"sub-{sub}",f"ses-{ses}","func_rev",f"sub-{sub}_ses-{ses}_bold_rev.json")
                            #file.split(".")[0] did not work
                            shutil.copy(re.sub(r'\.nii\.gz$', '', file)+".json",fmri_rev_json_copy)
                        elif imgTemp.shape == (64,64,64):
                            b0map_copy = os.path.join(in_path, f"sub-{sub}",f"ses-{ses}",f"sub-{sub}_ses-{ses}_b0map.nii")
                            shutil.copy(file, b0map_copy)
                    elif len(imgTemp.shape)==4:
                        fmri_nii_copy = os.path.join(in_path, f"sub-{sub}",f"ses-{ses}","func",f"sub-{sub}_ses-{ses}_bold.nii.gz")
                        shutil.copy(file,fmri_nii_copy)
                        fmri_json_copy = os.path.join(in_path, f"sub-{sub}",f"ses-{ses}","func",f"sub-{sub}_ses-{ses}_bold.json")
                        shutil.copy(re.sub(r'\.nii\.gz$', '', file)+".json", fmri_json_copy)
    else:
        print('fmri data is not found')
    
    dwi_file_list = glob.glob(os.path.join(in_path, '**', dwi_name), recursive=True)
    
    if len(dwi_file_list) != 0:
    
        for file in dwi_file_list:
                if "nii.gz" in file:
                    imgTemp = nii.load(file)
                    if len(imgTemp.shape)==3:
                        dwi_rev_nii_copy = os.path.join(in_path, f"sub-{sub}",f"ses-{ses}","dwi_rev",f"sub-{sub}_ses-{ses}_dwi_rev.nii.gz")
                        shutil.copy(file,dwi_rev_nii_copy)
                        dwi_rev_json_copy = os.path.join(in_path, f"sub-{sub}",f"ses-{ses}","dwi_rev",f"sub-{sub}_ses-{ses}_dwi_rev.json")
                        shutil.copy(re.sub(r'\.nii\.gz$', '', file)+".json",dwi_rev_json_copy)
                        dwi_rev_bvec_copy = os.path.join(in_path, f"sub-{sub}",f"ses-{ses}","dwi_rev",f"sub-{sub}_ses-{ses}_dwi_rev.bvec")
                        shutil.copy(re.sub(r'\.nii\.gz$', '', file)+".bvec",dwi_rev_bvec_copy)
                        dwi_rev_bval_copy = os.path.join(in_path, f"sub-{sub}",f"ses-{ses}","dwi_rev",f"sub-{sub}_ses-{ses}_dwi_rev.bval")
                        shutil.copy(re.sub(r'\.nii\.gz$', '', file)+".bval",dwi_rev_bval_copy)
                        
                    elif len(imgTemp.shape) == 4:
                        match = re.search(r'-\d+-(\d+)-', file)
                        if match and match.group(1) == "1":
                            dwi_nii_copy = os.path.join(in_path, f"sub-{sub}",f"ses-{ses}","dwi",f"sub-{sub}_ses-{ses}_dwi.nii.gz")
                            shutil.copy(file,dwi_nii_copy)
                            dwi_json_copy = os.path.join(in_path, f"sub-{sub}",f"ses-{ses}","dwi",f"sub-{sub}_ses-{ses}_dwi.json")
                            shutil.copy(re.sub(r'\.nii\.gz$', '', file)+".json",dwi_json_copy)
                            dwi_bvec_copy = os.path.join(in_path, f"sub-{sub}",f"ses-{ses}","dwi",f"sub-{sub}_ses-{ses}_dwi.bvec")
                            shutil.copy(re.sub(r'\.nii\.gz$', '', file)+".bvec",dwi_bvec_copy)
                            dwi_bval_copy = os.path.join(in_path, f"sub-{sub}",f"ses-{ses}","dwi",f"sub-{sub}_ses-{ses}_dwi.bval")
                            shutil.copy(re.sub(r'\.nii\.gz$', '', file)+".bval",dwi_bval_copy)
    else:
        print('dwi data is not found')

    scanProgram = os.path.join(in_path, "ScanProgram.scanProgram")

    with open(scanProgram, "r") as file:
        content = file.read()
        lines = content.split("##")
        for line in lines:
            if "$PVM_StudyB0Map" in line:
                parts_coma = line.split(",")
                b0_folder = parts_coma[2].strip()
                break
                
    b0map = glob.glob(os.path.join(in_path, b0_folder, '**', 'nifti', '*.nii'), recursive=True)
    b0map_copy = os.path.join(in_path, f"sub-{sub}",f"ses-{ses}",f"sub-{sub}_ses-{ses}_b0map.nii")
    if not os.path.exists(b0map_copy):
        shutil.copy(b0map[0],b0map_copy)

if __name__ == '__main__':
    main()