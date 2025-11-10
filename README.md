# MRI Processing Pipeline

A complete pipeline for converting raw Bruker data into NIfTI files with automated BIDS formatting and processing fMRI and DWI data.

## Installation

### Prerequisites
- [Docker](https://www.docker.com/)
- [Git Bash](https://git-scm.com/install/windows) (Windows users only)

### Setup Steps

1. **Clone the repository** and navigate to the directory:
   ```bash
   git clone https://github.com/temshil/mri.git
   cd path/to/mri
   ```

2. **Build the Docker image:**
   ```bash
   docker build -t img_name -f Dockerfile .
   ```
   
   The Docker image creates an Ubuntu environment with:
   - NiftyReg
   - FSL
   - DSI Studio
   - Python 3 with required packages

3. **Run the container:**
   ```bash
   docker run -it --name cont_name --gpus all --mount type=bind,source=/path/to/data,target=/temshil/data img_name
   ```
   
   > **Note:** Include `--gpus all` flag to use GPU-accelerated version of eddy function for DWI analysis, significantly decreasing processing time.

## Recommended Workflow

### 1. Setup Data Directories

In the directory bound to the container, create:
- `raw` folder - for your raw Bruker data
- `processed` folder - for output data

### 2. Start the Container

```bash
docker start cont_name
docker attach cont_name
```

> **Windows users:** You may need to add `winpty` before the second line.

### 3. Convert Raw Data to NifTI in BIDS Format

```bash
/temshil/src/conv2bids.sh
python /temshil/src/copybids.py --in_path /temshil/data/raw --out_path /temshil/data/processed
```

## fMRI Processing

### Part 1:

**Launch:**
```bash
/temshil/src/batch_fmri_part1.sh
```

**Steps included:**
1. Slice time correction using `slicetimer`
2. Correction of phase encoding distortions using [topup](https://fsl.fmrib.ox.ac.uk/fsl/docs/diffusion/topup/index.html?h=topup)
3. Estimation of motion confounds using [mcflirt](https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/mcflirt.html?h=mcflir)
4. Regression of motion confounds using `fsl_regfilt`
5. Independent component (IC) analysis using [melodic](https://fsl.fmrib.ox.ac.uk/fsl/docs/resting_state/melodic.html?h=melod)

### Manual Component Selection

After Part 1 completes, manually select ICs for regression. Save selections as a text file (list of numbers) named `bad_ics.txt` in the fMRI data directory.

### Part 2:

**Launch:**
```bash
/temshil/src/batch_fmri_part2.sh
```

**Steps included:**
1. Regression of selected ICs using `fsl_regfilt`
2. Atlas registration using [NiftyReg](https://github.com/KCL-BMEIS/niftyreg)
3. Regression of white matter and CSF time series using `fsl_regfilt`
4. Spatial filtering (2-voxel Gaussian kernel)
5. Frequency bandpassing (0.01-0.1 Hz)
6. Correlation analysis

**Output:** Results stored in `corr` folder containing:
- Matrices with Pearson R and Z-transformed Pearson R values
- Plot of the Pearson R correlation matrix
- Correlation maps for the selected regions of interests (by default, these are amygdalar nuclei, barrel field, and anterior cingulate cortex). 

## DWI Processing

**Launch:**
```bash
/temshil/src/batch_dwi.sh
```

**Steps included:**
1. Correction of phase encoding distortions using [topup](https://fsl.fmrib.ox.ac.uk/fsl/docs/diffusion/topup/index.html?h=topup)
2. Correction of eddy currents using [eddy](https://fsl.fmrib.ox.ac.uk/fsl/docs/diffusion/eddy/index.html?h=eddy)
3. Atlas registration using [NiftyReg](https://github.com/KCL-BMEIS/niftyreg)
4. Tractography using [DSI Studio](https://dsi-studio.labsolver.org/) with optimized parameters for mouse brain

**Output:** Results stored in `dsi_studio` folder containing:
- Correlation matrices
- Text file with graph analysis features

## Acknowledgements

- Dockerfile and Atlas are based on [AIDAmri](https://github.com/maswendt/AIDAmri)
- Analytical flow developed with major contributions from Julius Benson, Jozien Goenze, and Howard Gritton
- DWI analytical pipeline developed with major contributions from Paul Camacho and Brad Sutton
- [Topup parameters](https://github.com/frankyeh/DSI-Studio/blob/b33a41da21f138e9cad47a4bb3b424510e56fd3a/other/topup_param.txt#L4) from Frank Yeh

## License

MIT license