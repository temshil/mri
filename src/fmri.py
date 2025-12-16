import numpy as np
import nibabel as nii
import ants
import argparse
import subprocess
import os
import re
import shutil
from scipy.stats import pearsonr
from scipy.io import savemat
from scipy.ndimage import gaussian_filter
from scipy.signal import stft
import matplotlib.pyplot as plt

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

def plot_spect(melodic_out):
    mix = np.loadtxt(os.path.join(melodic_out,'melodic_mix'))
    n_components = mix.shape[1]

    for ic in range(n_components):
        tc = mix[:, ic]

        fs = 1/1.5
        f, t, Zxx = stft(tc, fs=fs, nperseg=64)

        Sxx = np.abs(Zxx)
        Sxx_dB = 20 * np.log10(Sxx + 1e-8)

        plt.figure(figsize=(8, 4))
        plt.pcolormesh(t, f, Sxx_dB, shading='gouraud', cmap='viridis')
        plt.title(f'Spectrogram of IC {ic+1}')
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        plt.colorbar(label='Power [dB]')
        plt.tight_layout()
        os.makedirs(os.path.join(melodic_out,'spect'), exist_ok=True)
        save_path = os.path.join(melodic_out,'spect', f'{ic+1}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

def wmcsf_mask_maker(anno_path, corr_map_path):  
    anno_img = nii.load(anno_path)
    anno_data = anno_img.get_fdata()
    
    anno_wmcsf_mask = (anno_data == 200).astype(np.uint8)
    
    corr_map = nii.load(corr_map_path)
    corr_map_data = corr_map.get_fdata()
    
    max_val = np.max(corr_map_data)
    thr = 0.5 * max_val
    corr_map_bin = (corr_map_data >= thr).astype(np.uint8)
    
    wmcsf_mask = corr_map_bin * anno_wmcsf_mask
    
    mask_img = nii.Nifti1Image(wmcsf_mask, anno_img.affine, anno_img.header)
    wmcsf_mask_path = os.path.join(os.path.dirname(anno_path), 'wmcsf_mask.nii.gz')
    nii.save(mask_img, wmcsf_mask_path)
    return wmcsf_mask_path

def smooth(in_path):
    img = nii.load(in_path)
    data = img.get_fdata()

    sigma_xy = 2 / 2.355
    sigma_z = 0

    smoothed_data = np.zeros_like(data)

    for t in range(data.shape[-1]):  # loop over timepoints
        smoothed_data[..., t] = gaussian_filter(data[..., t],
                                                sigma=[sigma_xy, sigma_xy, sigma_z])

    smoothed_img = nii.Nifti1Image(smoothed_data, img.affine, img.header)
    out_path = in_path.split('.')[0]+'_sm.nii.gz'
    nii.save(smoothed_img, out_path)
    return out_path

def fisher_z(r):
    r = np.clip(r, -0.999999, 0.999999)
    return np.arctanh(r)

def roi_time_series(fmri_data, atlas_data, roi_labels):
    ts_dict = {}
    for roi in roi_labels:
        mask = atlas_data == roi
        if np.sum(mask) == 0:
            continue
        roi_ts = fmri_data[mask, :].mean(axis=0)
        ts_dict[roi] = roi_ts
    return ts_dict

def roi_to_roi_correlation(ts_dict):
    rois = list(ts_dict.keys())
    n = len(rois)
    R = np.zeros((n, n))
    
    for i, roi1 in enumerate(rois):
        for j, roi2 in enumerate(rois):
            if i <= j:
                r, _ = pearsonr(ts_dict[roi1], ts_dict[roi2])
                R[i, j] = r
                R[j, i] = r
    
    Z = fisher_z(R)
    return R, Z, rois

def roi_to_voxel_map(fmri_masked, ts_dict, mask, target_rois, out_dir, affine):
    os.makedirs(out_dir, exist_ok=True)

    voxel_ts = fmri_masked.reshape(-1, fmri_masked.shape[-1])
    voxel_ts = voxel_ts[mask.flatten() > 0, :]  # apply mask

    for roi in target_rois:
        if roi not in ts_dict:
            print(f"ROI {roi} not found — skipping.")
            continue

        roi_ts = ts_dict[roi]
        roi_ts = roi_ts - np.mean(roi_ts)
        voxel_ts_demeaned = voxel_ts - voxel_ts.mean(axis=1, keepdims=True)

        numerator = np.dot(voxel_ts_demeaned, roi_ts)
        denominator = np.sqrt(np.sum(voxel_ts_demeaned**2, axis=1) * np.sum(roi_ts**2))
        r_values = numerator / denominator

        r_values = np.nan_to_num(r_values, nan=0.0)
        z_values = fisher_z(r_values)

        r_map = np.zeros(mask.shape)
        z_map = np.zeros(mask.shape)
        r_map[mask > 0] = r_values
        z_map[mask > 0] = z_values

        nii.save(nii.Nifti1Image(r_map, affine), os.path.join(out_dir, f'ROI{roi}_Rmap.nii.gz'))
        nii.save(nii.Nifti1Image(z_map, affine), os.path.join(out_dir, f'ROI{roi}_Zmap.nii.gz'))
        print(f"Saved ROI {roi} R and Z maps in {out_dir}")

def plot_matrix(data, labels, out_dir):   
    plt.figure(figsize=(25, 20))
    plt.imshow(data, cmap='coolwarm', aspect='auto',vmin=-1, vmax=1)
    plt.colorbar() 
    save_path = os.path.join(out_dir,'corr_matrix.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

def parse_args():
    parser = argparse.ArgumentParser(
        description="func analysis pipeline")
    parser.add_argument('--in_path', type=str,
                        help='Input path for NIfTI file')
    parser.add_argument('--part', type=str,
                        help='Processing part 1 or 2')
    parser.add_argument('--rm', action="store_true",
      help="Save raw data, cleaned time series, and remove files larger than 100 MB."
  )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    in_path = args.in_path
    part = args.part
    
    match = re.search(r"sub-(.*?)/ses-(.*?)/", in_path)
    if match:
        sub = match.group(1)
        ses = match.group(2)
    in_path =  os.path.dirname(os.path.dirname(in_path))
    
    func_in_path = os.path.join(in_path,'func',f'sub-{sub}_ses-{ses}_bold.nii.gz')
    func_rev_in_path = os.path.join(in_path,'func_rev',f'sub-{sub}_ses-{ses}_bold_rev.nii.gz')
    anat_in_path = os.path.join(in_path,'anat',f'sub-{sub}_ses-{ses}_T2w.nii.gz')
    
    melodic_out = os.path.join(in_path,'func','melodic')
    os.makedirs(melodic_out, exist_ok=True)
    
    if part == "1":    
        
        backup_anat = os.path.join(in_path,'anat','raw')
        if not os.path.exists(backup_anat):
            os.makedirs(backup_anat, exist_ok=True)
            shutil.copy2(anat_in_path, os.path.join(backup_anat, os.path.basename(anat_in_path)))
            
        backup_func = os.path.join(in_path,'func','raw')
        backup_func_rev = os.path.join(in_path,'func_rev','raw')
        
        os.makedirs(backup_func, exist_ok=True)
        os.makedirs(backup_func_rev, exist_ok=True)
    
        shutil.copy2(func_in_path, os.path.join(backup_func, os.path.basename(func_in_path)))
        shutil.copy2(func_rev_in_path, os.path.join(backup_func_rev, os.path.basename(func_rev_in_path)))
        
        subprocess.run(['fslorient', '-deleteorient',
                        func_in_path], check=True)
        
        subprocess.run(['fslorient', '-forceradiological',
                        func_in_path], check=True)
        
        subprocess.run(['fslorient', '-deleteorient',
                        anat_in_path], check=True)
        
        subprocess.run(['fslorient', '-forceradiological',
                        anat_in_path], check=True)
        
        subprocess.run(['fslorient', '-deleteorient',
                        func_rev_in_path], check=True)
        
        subprocess.run(['fslorient', '-forceradiological',
                        func_rev_in_path], check=True)
        
        
        subprocess.run([
        'slicetimer',
        '-i',func_in_path,
        '-o',func_in_path.split('.')[0]+'_stc.nii.gz',
        '-r', '1.5',
        '-d','3',
        '--odd'
        ], check=True)
        
        out_path_sc = changescale(func_in_path.split('.')[0]+'_stc.nii.gz', 20)
        out_path_rev_sc = changescale(func_rev_in_path, 20)
        
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
        
        subprocess.run([
        'applytopup',
        '--imain='+out_path_sc,
        '--datain='+acq_param,
        '--topup='+out_path_sc.split('.')[0]+'_AP_PA_tu',
        '--out='+out_path_sc.split('.')[0]+'_tu',
        '--method=jac',
        '--inindex=1'
        ], check=True)
    
        subprocess.run([
        'mcflirt',
        '-in', out_path_sc.split('.')[0]+'_tu.nii.gz',
        '-out', out_path_sc.split('.')[0]+'_tu_mcflirt.nii.gz',
        '-plots',
        '-report'
        ], check=True)
        
        subprocess.run([
        'fsl_regfilt',
        '-i', out_path_sc.split('.')[0]+'_tu.nii.gz',
        '-o',out_path_sc.split('.')[0]+'_tu_mc.nii.gz',
        '-d',out_path_sc.split('.')[0]+'_tu_mcflirt.nii.gz.par',
        '-f','1,2,3,4,5,6'
        ], check=True)
        
        subprocess.run(['fslroi', out_path_sc.split('.')[0]+'_tu_mc.nii.gz',
                        out_path_sc.split('.')[0]+'_tu_mc_fs', '0', '1'], check=True)
        
        out_path_bc = biascorrection(out_path_sc.split('.')[0]+'_tu_mc_fs.nii.gz', 2, [50, 50, 50, 50, 0])
        
        subprocess.run(['bet', out_path_bc,
                        out_path_bc.split('.')[0]+'_bet.nii.gz', 
                        '-f', '0.3',
                        '-r', '50',
                        '-g', '0',
                        '-m', '-R'], check=True)    
        
        changescale(out_path_bc.split('.')[0]+'_bet.nii.gz', 0.05)
        changescale(out_path_bc.split('.')[0]+'_bet_mask.nii.gz', 0.05)
        
        out_path_ds = changescale(out_path_sc.split('.')[0]+'_tu_mc.nii.gz', 0.05)   
        
        subprocess.run([
        'melodic',
        '-i', out_path_ds,
        '-o', melodic_out,
        '--Oall',
        '--report',
        '--bgthreshold=0.25',
        '--tr=1.5',
        '--nobet'
        ], check=True)
        
        plot_spect(melodic_out)
    
    if part == "2":
        
        func_in_path = os.path.join(in_path,'func',f'sub-{sub}_ses-{ses}_bold_stc_sc_tu_mc_ds.nii.gz')
        func_in_path_mask = os.path.join(in_path,'func',f'sub-{sub}_ses-{ses}_bold_stc_sc_tu_mc_fs_bc_bet_mask_ds.nii.gz')
        func_in_path_bet = os.path.join(in_path,'func',f'sub-{sub}_ses-{ses}_bold_stc_sc_tu_mc_fs_bc_bet_ds.nii.gz')
        
        if  os.path.exists(os.path.join(in_path,'func', 'bad_ics.txt')):
            
            with open(os.path.join(in_path,'func', 'bad_ics.txt'), "r") as f:
                bad_ics = [line.strip() for line in f if line.strip()]  # remove empty lines
    
            bad_ics_joined = ",".join(bad_ics)
                        
            subprocess.run([
            'fsl_regfilt',
            '-i',func_in_path,
            '-o',func_in_path.split('.')[0]+'_ica.nii.gz',
            '-d', os.path.join(melodic_out,'melodic_mix'),
            '-f', bad_ics_joined,
            ], check=True)
            
            func_in_path_ica = func_in_path.split('.')[0]+'_ica.nii.gz'
            
        else:
            func_in_path_ica = func_in_path
        
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
        
        out_atlas = os.path.join(in_path,'func','anno2func.nii.gz')
        
        if not os.path.exists(out_atlas):
            
            subprocess.run(['reg_aladin', '-ref', out_path_anat_ds,
                            '-flo',
                            func_in_path_bet,
                            '-aff', os.path.join(in_path,'func','func2anat_affine.txt'),
                            '-res', os.path.join(in_path,'func','func2anat_ala.nii.gz')], check=True)
                
            subprocess.run(['reg_transform', '-ref', out_path_anat_ds,
                            '-invAff', os.path.join(in_path,'func','func2anat_affine.txt'),
                            os.path.join(in_path,'func','anat2func_invaffine.txt')], check=True)
        
            
            subprocess.run(['reg_resample', '-ref',
                            out_path_anat_ds,
                            '-flo',anno_path,
                            '-trans', os.path.join(in_path,'anat','temp2anat_invaffine.txt'), 
                            '-inter','0',
                            '-res', os.path.join(in_path,'anat','anno2anat.nii.gz')], check=True)
            
            subprocess.run(['reg_resample', '-ref',
                            func_in_path_bet,
                            '-flo', os.path.join(in_path,'anat','anno2anat.nii.gz'),
                            '-trans', os.path.join(in_path,'func','anat2func_invaffine.txt'), 
                            '-inter','0',
                            '-res', os.path.join(in_path,'func','anno2func.nii.gz')], check=True)
                 
                
        fmri_img = nii.load(func_in_path_ica)
        fmri_data = fmri_img.get_fdata()
        atlas_img = nii.load(out_atlas)
        atlas_data = atlas_img.get_fdata().astype(int)
        mask_data = nii.load(func_in_path_mask).get_fdata().astype(bool)
        affine = atlas_img.affine
        
        fmri_masked = fmri_data * mask_data[..., np.newaxis]
    
        roi_labels = np.unique(atlas_data)
        roi_labels = roi_labels[roi_labels > 0]
        ts_dict = roi_time_series(fmri_masked, atlas_data, roi_labels)
            
        roi_to_voxel_map(fmri_masked, ts_dict, mask_data, [31], os.path.dirname(out_atlas), affine)
        
        wmcsf_mask = wmcsf_mask_maker(out_atlas,os.path.join(os.path.dirname(out_atlas),'ROI31_Rmap.nii.gz'))
        
        subprocess.run([
            'fslmeants',
            '-i', func_in_path_ica,
            '-o', os.path.join(in_path,'func','wmcsf_ts.txt'),
            '--label='+wmcsf_mask
            ], check=True)
        
        subprocess.run([
        'fsl_regfilt',
        '-i', func_in_path_ica,
        '-o',func_in_path_ica.split('.')[0]+'_wmcsf.nii.gz',
        '-d', os.path.join(in_path,'func','wmcsf_ts.txt'),
        '-f','1',
        ], check=True)
        
        hp_sigma = 1/(2*0.01*1.5)
        lp_sigma = 1/(2*0.1*1.5)
        #https://www.jiscmail.ac.uk/cgi-bin/webadmin?A2=fsl;fe2fad43.1104
        
        subprocess.run(['fslmaths', func_in_path_ica.split('.')[0]+'_wmcsf.nii.gz',
                        '-bptf',
                        str(hp_sigma),
                        str(lp_sigma), 
                        func_in_path_ica.split('.')[0]+'_wmcsf_bp.nii.gz'], check=True)
        
        wmcsf_mask_atlas = np.isin(atlas_data, 200)
        atlas_data[wmcsf_mask_atlas]=0

        out_dir = os.path.join(in_path,'func','corr')
    
        fmri_img = nii.load(func_in_path_ica.split('.')[0]+'_wmcsf_bp.nii.gz')
        fmri_data = fmri_img.get_fdata()
        fmri_data = fmri_data[:, :, :, 20:-20]
        #https://github.com/CoBrALab/RABIES/blob/master/docs/confound_correction.md
    
        os.makedirs(out_dir, exist_ok=True)
    
        fmri_masked = fmri_data * mask_data[..., np.newaxis]
    
        roi_labels = np.unique(atlas_data)
        roi_labels = roi_labels[roi_labels > 0]
        ts_dict = roi_time_series(fmri_masked, atlas_data, roi_labels)
        
        R, Z, rois = roi_to_roi_correlation(ts_dict)
        savemat(os.path.join(out_dir, 'ROI_correlation_matrices.mat'),
                {'R': R, 'Z': Z, 'ROI_labels': np.array(rois)})
        print("Saved ROI correlation matrices to MATLAB format.")
        
        plot_matrix(R,rois,out_dir)
        
        func_in_path_sm = smooth(func_in_path_ica.split('.')[0]+'_wmcsf_bp.nii.gz')   
        
        selected_rois = [31,1080,131]
    
        fmri_img_sm = nii.load(func_in_path_sm)
        fmri_data_sm = fmri_img_sm.get_fdata()
        fmri_data_sm = fmri_data_sm[:, :, :, 20:-20]
        
        fmri_masked_sm = fmri_data_sm * mask_data[..., np.newaxis]
    
        ts_dict_sm = roi_time_series(fmri_masked_sm, atlas_data, roi_labels)
        
        roi_to_voxel_map(fmri_masked_sm, ts_dict_sm, mask_data, selected_rois, out_dir, affine)

        
        if args.rm:
            limit = 100 * 1024 * 1024 
    
            for file in os.listdir(os.path.join(in_path,'func')):
               if file.endswith("nii.gz") and not file.endswith(("sm.nii.gz", "mc_ds.nii.gz")):
                    full_path = os.path.join(os.path.join(in_path,'func'), file)
                    if os.path.getsize(full_path) > limit:
                        os.remove(full_path)