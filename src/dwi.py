import numpy as np
from dipy.denoise.patch2self import patch2self
from dipy.io.image import load_nifti, save_nifti
import nibabel as nii
import ants
import argparse
import subprocess
import os
import re
import shutil

def denoise(in_path):
    data, affine = load_nifti(in_path)
    bvals = np.loadtxt(in_path.split('.')[0]+'.bval')
    dwi_denoised = patch2self(
        data,
        bvals,
        model="ols",
        shift_intensity=True,
        clip_negative_vals=False,
        b0_threshold=50,
    )
    out_path = in_path.split('.')[0]+'_dn.nii.gz'
    save_nifti(out_path, dwi_denoised, affine)
    return out_path

def changescale(in_path, factor):
    if factor > 1:
        out_path = in_path.split('.')[0]+'_sc.nii.gz'
    else:
        out_path = in_path.split('.')[0]+'_ds.nii.gz'
    img = nii.load(in_path)
    data = img.get_fdata()
    affine = img.affine.copy()
    header = img.header.copy()
    scaled_affine = affine.copy()
    scaled_affine[:3, :3] *= factor  
    scaled_img = nii.Nifti1Image(data, scaled_affine, header=header)
    scaled_img.header.set_xyzt_units(xyz='mm')
    nii.save(scaled_img, out_path)
    return out_path

def biascorrection(in_path, shrink_factor, n_iterations):
    img = ants.image_read(in_path)

    corrected_image = ants.n4_bias_field_correction(
        img,
        shrink_factor=shrink_factor,
        spline_param=20,
        convergence={'iters': n_iterations, 'tol': 1e-7}
    )
    out_path = in_path.split('.')[0]+'_bc.nii.gz'
    corrected_image.to_file(out_path)
    return out_path

def flip_nifti(in_path,axis):
    img = nii.load(in_path)
    data = img.get_fdata()
    flipped_data = np.flip(data, axis=axis) 
    flipped_img = nii.Nifti1Image(flipped_data, img.affine)
    axis_name = str(axis)
    out_path = in_path.split('.')[0]+'_fl_'+axis_name+'.nii.gz'
    nii.save(flipped_img, out_path)
    return out_path
    
def change_header(in_path1,in_path2):
    ref_img = nii.load(in_path1)
    flo_img = nii.load(in_path2)
    data = flo_img.get_fdata()
    flo_img_upd = nii.Nifti1Image(data, flo_img.affine, header=ref_img.header)
    flo_img_upd_out = in_path2.split('.')[0]+'_upd.nii.gz'
    nii.save(flo_img_upd, flo_img_upd_out)
    return(flo_img_upd_out)

def parse_args():
    parser = argparse.ArgumentParser(
        description="DWI analysis pipeline")
    parser.add_argument('--in_path', type=str,
                        help='Input path for NIfTI file')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    in_path = args.in_path
    
    match = re.search(r"sub-(.*?)/ses-(.*?)/", in_path)
    if match:
        sub = match.group(1)
        ses = match.group(2)
    in_path =  os.path.dirname(os.path.dirname(in_path))
    
    dwi_in_path = os.path.join(in_path,'dwi',f'sub-{sub}_ses-{ses}_dwi.nii.gz')
    dwi_rev_in_path = os.path.join(in_path,'dwi_rev',f'sub-{sub}_ses-{ses}_dwi_rev.nii.gz')
    anat_in_path = os.path.join(in_path,'anat',f'sub-{sub}_ses-{ses}_T2w.nii.gz')
    
    backup_dwi = os.path.join(in_path,'dwi','raw')
    backup_dwi_rev = os.path.join(in_path,'dwi_rev','raw')
    
    os.makedirs(backup_dwi, exist_ok=True)
    os.makedirs(backup_dwi_rev, exist_ok=True)
    
    shutil.copy2(dwi_in_path, os.path.join(backup_dwi, os.path.basename(dwi_in_path)))
    shutil.copy2(dwi_rev_in_path, os.path.join(backup_dwi_rev, os.path.basename(dwi_rev_in_path)))
    
    backup_anat = os.path.join(in_path,'anat','raw')
    if not os.path.exists(backup_anat):
        os.makedirs(backup_anat, exist_ok=True)
        shutil.copy2(anat_in_path, os.path.join(backup_anat, os.path.basename(anat_in_path)))
    
    subprocess.run(['fslorient', '-deleteorient',
                    dwi_in_path], check=True)
    
    subprocess.run(['fslorient', '-forceradiological',
                    dwi_in_path], check=True)
    
    subprocess.run(['fslorient', '-deleteorient',
                    anat_in_path], check=True)
    
    subprocess.run(['fslorient', '-forceradiological',
                    anat_in_path], check=True)
    
    subprocess.run(['fslorient', '-deleteorient',
                    dwi_rev_in_path], check=True)
    
    subprocess.run(['fslorient', '-forceradiological',
                    dwi_rev_in_path], check=True)
    
    out_path_dn = denoise(dwi_in_path)
    
    out_path_sc = changescale(out_path_dn, 20)
    out_path_rev_sc = changescale(dwi_rev_in_path, 20)
    
    subprocess.run(['fslroi', out_path_sc,
                    out_path_sc.split('.')[0]+'_AP', '0', '1'], check=True)
    
    
    subprocess.run(['fslmerge', '-t', out_path_sc.split('.')[0]+'_AP_PA.nii.gz',
                    out_path_sc.split('.')[0]+'_AP.nii.gz', 
                    out_path_rev_sc], check=True)
    
    acq_param='/temshil/lib/acq_param.txt'
    
    subprocess.run([
    'topup',
    '--imain='+out_path_sc.split('.')[0]+'_AP_PA.nii.gz',
    '--datain='+acq_param,
    '--config=b02b0.cnf',
    '--out='+out_path_sc.split('.')[0]+'_AP_PA_tu',
    '--warpres=20,16,14,12,10,6,4,4,4',
    '--subsamp=2,2,2,2,2,1,1,1,1',
    '--fwhm=8,6,4,3,3,2,1,0,0',
    '--miter=5,5,5,5,5,10,10,20,20',
    '--lambda=0.005,0.001,0.0001,0.000015,0.000005,0.0000005,0.00000005,0.0000000005,0.00000000001',
    '--ssqlambda=1',
    '--regmod=bending_energy',
    '--estmov=1,1,1,1,1,0,0,0,0',
    '--minmet=0,0,0,0,0,1,1,1,1',
    '--splineorder=3',
    '--numprec=double',
    '--interp=spline',
    '--scale=1'
    ], check=True)
    
    out_path_bc = biascorrection(out_path_sc.split('.')[0]+'_AP.nii.gz',2, [50, 50, 50, 50, 0])
    
    subprocess.run(['bet', out_path_bc,
                    out_path_bc.split('.')[0]+'_bet.nii.gz', 
                    '-f', '0.25',
                    '-r', '50',
                    '-g', '0',
                    '-m', '-R'], check=True)
    
    open(os.path.join(in_path,'dwi','index.txt'), "w").close() 

    for i in range(1, 126):
        with open(os.path.join(in_path,'dwi','index.txt'), "a") as f:
            f.write("1\n")
    
    subprocess.run([
    'eddy_cuda11.0',
    '--imain='+out_path_sc,
    '--mask='+out_path_bc.split('.')[0]+'_bet_mask.nii.gz',
    '--acqp='+acq_param,
    '--index='+os.path.join(in_path,'dwi','index.txt'),
    '--bvecs='+dwi_in_path.split('.')[0]+'.bvec',
    '--bvals='+dwi_in_path.split('.')[0]+'.bval',
    '--topup='+out_path_sc.split('.')[0]+'_AP_PA_tu',
    '--out='+out_path_sc.split('.')[0]+'_tu_ed.nii.gz', 
    # '--nthr=12', The version compiled for GPU can only use 1 CPU thread (i.e. --nthr=1)
    '--nthr=1'
    # '--s2v_niter=2'
    ], check=True)
    
    out_path_ed = out_path_sc.split('.')[0]+'_tu_ed.nii.gz'
    
    subprocess.run(['fslroi', out_path_ed,
                    out_path_ed.split('.')[0]+'_fs', '0', '1'], check=True)
    
    out_path_fs_bc = biascorrection(out_path_ed.split('.')[0]+'_fs.nii.gz', 2, [50, 50, 50, 50, 0])
    
    subprocess.run(['bet', out_path_fs_bc,
                    out_path_fs_bc.split('.')[0]+'_bet.nii.gz', 
                    '-f', '0.3',
                    '-r', '50',
                    '-g', '0',
                    '-m', '-R'], check=True)    
    

    
    out_path_ds_bet = changescale(out_path_fs_bc.split('.')[0]+'_bet.nii.gz', 0.05)
    
    out_path_ds_mask = changescale(out_path_fs_bc.split('.')[0]+'_bet_mask.nii.gz', 0.05)
    
    out_path_ds = changescale(out_path_ed, 0.05)    
    
    anno_path='/temshil/lib/atlas.nii.gz'
    template_path='/temshil/lib/average_template_50.nii.gz'
    
    if not os.path.exists(os.path.join(in_path,'anat','anat2temp_ala.nii.gz')):
    
        out_path_anat_sc = changescale(anat_in_path, 20)
    
        out_path_anat_bc = biascorrection(out_path_anat_sc, 2, [50, 50, 50, 50, 0])
    
        subprocess.run(['bet', out_path_anat_bc,
                    out_path_anat_bc.split('.')[0]+'_bet.nii.gz', 
                    '-f', '0.25',
                    '-r', '50',
                    '-g', '0',
                    '-m', '-R'], check=True)    
    
        out_path_anat_ds = changescale(out_path_anat_bc.split('.')[0]+'_bet.nii.gz', 0.05)
          
    
        subprocess.run(['reg_aladin', '-ref', template_path,
                    '-flo',
                    out_path_anat_ds,
                    '-aff', os.path.join(in_path,'anat','anat2temp_affine.txt'),
                    '-res',os.path.join(in_path,'anat','anat2temp_ala.nii.gz')], check=True)
        
        subprocess.run(['reg_transform', '-ref', template_path,
                    '-invAff', os.path.join(in_path,'anat','anat2temp_affine.txt'),
                    os.path.join(in_path,'anat','temp2anat_invaffine.txt')], check=True)
    
    else:
        out_path_anat_ds = anat_in_path.split('.')[0]+'_sc_bc_bet_ds.nii.gz'
    
    subprocess.run(['reg_aladin', '-ref', out_path_anat_ds,
                    '-flo',
                    out_path_ds_bet,
                    '-aff', os.path.join(in_path,'dwi','dwi2anat_affine.txt'),
                    '-res', os.path.join(in_path,'dwi','dwi2anat_ala.nii.gz')], check=True)
        
    subprocess.run(['reg_transform', '-ref', out_path_anat_ds,
                    '-invAff', os.path.join(in_path,'dwi','dwi2anat_affine.txt'),
                    os.path.join(in_path,'dwi','anat2dwi_invaffine.txt')], check=True)

    
    subprocess.run(['reg_resample', '-ref',
                    out_path_anat_ds,
                    '-flo',anno_path,
                    '-trans', os.path.join(in_path,'anat','temp2anat_invaffine.txt'), 
                    '-inter','0',
                    '-res',os.path.join(in_path,'anat','anno2anat_nii.gz')], check=True)
    
    subprocess.run(['reg_resample', '-ref',
                    out_path_ds_bet,
                    '-flo', os.path.join(in_path,'anat','anno2anat_nii.gz'),
                    '-trans', os.path.join(in_path,'dwi','anat2dwi_invaffine.txt'), 
                    '-inter','0',
                    '-res', os.path.join(in_path,'dwi','anno2dwi.nii.gz')], check=True)
        
    out_atlas = os.path.join(in_path,'dwi','anno2dwi.nii.gz')
    
    basename = os.path.basename(out_path_ds)
    
    out_atlas_upd = change_header(out_path_ds,out_atlas)
    out_mask_upd = change_header(out_path_ds,out_path_ds_mask)
    
    atlas_img = nii.load(out_atlas_upd)
    atlas_data = atlas_img.get_fdata()
    wmcsf_mask = np.isin(atlas_data, 200)
    atlas_data[wmcsf_mask]=0
    atlas_data_rm_wmcsf = nii.Nifti1Image(atlas_data, atlas_img.affine, header=atlas_img.header)
    nii.save(atlas_data_rm_wmcsf, out_atlas_upd)
    
    dsi_output = os.path.join(in_path,'dwi','dsi_studio')
    
    os.makedirs(dsi_output, exist_ok=True)
    
    dsi_studio = '/temshil/dsi_studio/dsi-studio/dsi_studio'
    
    bval = dwi_in_path.split('.')[0]+'.bval'
    bvec = dwi_in_path.split('.')[0]+'.bvec'
    
    subprocess.run([dsi_studio, '--action=src',
                    '--source='+out_path_ds,
                    '--bval='+bval,
                    '--bvec='+bvec,
                    '--output='+dsi_output], check=True)

    subprocess.run([dsi_studio, '--action=rec',
                    '--source='+dsi_output+'/'+basename.split('.')[0]+'.sz',
                    '--cmd="[Step T2][B-table][flip by]+[Step T2][B-table][flip bz]"',
                    '--method=4',
                    '--param0=1.2',
                    '--mask='+out_mask_upd,
                    '--output='+dsi_output], check=True)
                    # '--correct_bias_field=0'

    subprocess.run([dsi_studio, '--action=trk',
                    '--source='+dsi_output+'/'+basename.split('.')[0]+'.gqi.fz',
                    '--turning_angle=60',
                    '--step_size=0.07',
                    '--min_length=0.4',
                    '--max_length=10',
                    '--threshold_index=qa',
                    '--fa_threshold=0.02',
                    '--track_voxel_ratio=1',
                    '--seed_count=1000000',
                    '--connectivity='+out_atlas_upd,
                    '--connectivity_output=matrix,measure',
                    '--connectivity_type=pass,end',
                    '--output='+dsi_output], check=True)