# Wellcounter: Automated High-Throughput Phenotyping for Aquatic Microinvertebrates

## Overview
The **Wellcounter** is an open-source platform designed for **automated high-throughput phenotyping** of aquatic microinvertebrates. It enables large-scale ecological experiments by automating **image acquisition, processing, and analysis**, significantly reducing manual effort in tracking population growth and swimming behavior.

This repository provides resources for **building and using** the Wellcounter, including:
- **Hardware design**: Instructions for assembling the Wellcounter using commercially available and 3D-printed components.
- **Software**: Python-based modules for automated data acquisition, image analysis, and motion tracking.
- **Documentation**: A detailed manual and supplementary materials supporting setup and optimization.

The Wellcounter was introduced in:
> Stelzer, C.P. & Groffman, D. (2025) *Wellcounter: Automated High-Throughput Phenotyping for Aquatic Microinvertebrates*. Methods in Ecology and Evolution.

## Features
- **Automated Imaging**: Captures high-resolution images and videos using a motorized **XY linear guide system**.
- **High-Throughput Analysis**: Handles up to **42 six-well plates** (252 populations) per batch. The recording of one batch takes about 1-1.5 hours per day. Multiple batches are possible
- **Image & Motion Analysis**: Quantifies population size and swimming behavior.
- **Customizable & Open-Source**: Fully programmable and adaptable to different microorganisms.
- **3D-Printed Components**: CAD files for custom-designed adapters and mounts included.

## Getting Started
### 1. Hardware Setup
The Wellcounter integrates a **Basler a2A4504-27g5m digital camera**, a **telecentric lens**, and a **darkfield illumination ring**. Detailed **assembly instructions** are available in the **Wellcounter Manual**.

### 2. Software Installation
The Wellcounter software is written in Python and requires installation via **Conda**.
```bash
conda env create -f wellcount6_pi.yml
conda activate wellcount6_pi
```

### 3. Data Acquisition & Analysis
- **Automated Imaging**: The `wc_record_experiment.py` script controls the Wellcounter for video acquisition.
- **Image Analysis**: The `wc_analyze_one_sample.py` script detects and quantifies microorganisms in a recorded movie.
- **Motion Analysis**: The motion module tracks swimming behavior over time.
- **Optimization Module**: Fine-tunes imaging parameters using a training dataset.

## Documentation & Tutorials
- **[Wellcounter Manual](https://github.com/cpstelzer/wellcounter)**: Step-by-step guide to assembling and using the system.
- **Example Datasets**:
  - Training & test data available via Zenodo:
    10.5281/zenodo.14833208
    10.5281/zenodo.14833370
    10.5281/zenodo.14833381
    

## Citation
If you use the Wellcounter in your research, please cite:
> Stelzer, C.P. & Groffman, D. (2025) *Wellcounter: Automated High-Throughput Phenotyping for Aquatic Microinvertebrates*. Methods in Ecology and Evolution.

## License
This project is released under an **open-source license**, allowing modifications and contributions from the scientific community.

---
For more details, visit the [Wellcounter repository](https://github.com/cpstelzer/wellcounter).

