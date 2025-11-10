import glob
import os
import shutil
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Move converted files to BIDS.")
    parser.add_argument('--in_path', type=str,
                        help='Input path')
    parser.add_argument('--out_path', type=str,
                        help='Output path')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    in_path = args.in_path
    out_path = args.out_path
    
    os.makedirs(out_path, exist_ok=True)
    count = 0
    sub_dirs = [d for d in glob.glob(os.path.join(in_path,'**', "sub-*")) if os.path.isdir(d)]
    
    for in_sub in sub_dirs:
        out_sub = os.path.join(out_path, os.path.basename(in_sub))
        
        if os.path.isdir(os.path.join(os.path.dirname(in_sub),'sub-sub')):
            count += 1
            out_sub_new =  os.path.join(out_path, os.path.basename(in_sub)+str(count))
            shutil.copytree(in_sub, out_sub_new,dirs_exist_ok=True)
        elif os.path.isdir(os.path.join(in_sub,'ses-ses')):
            count += 1
            out_sub_new = os.path.join(out_path, os.path.basename(in_sub)+str(count))
            shutil.copytree(os.path.join(in_sub,'ses-ses'), out_sub_new, dirs_exist_ok=True)
        else:
            shutil.copytree(in_sub, out_sub,dirs_exist_ok=True)
    