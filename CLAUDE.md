# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is a Halogen Hackathon project focused on brain injury analysis with three machine learning tasks: prediction, prescription, and inference. The project analyzes focal brain injury patterns using anatomical lesion maps to predict outcomes, prescribe treatments, and infer functional brain organization.

## Development Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

## Key Dependencies
- nibabel: NIfTI neuroimaging file handling
- scikit-learn: Machine learning models
- pandas: Data manipulation
- numpy: Numerical computations
- joblib: Model serialization
- tqdm: Progress bars

## Core Architecture
The main implementation is in `src/tasks.py` which contains:
- `_read_in_data()`: Loads lesion maps from ZIP archives and matches with CSV targets
- `task1_fit()`: Predictive modeling for deficit severity (regression)
- `task2_fit()`: Prescriptive modeling for treatment response (classification)
- `task3()`: Inference of anatomical maps for deficit and treatment regions

## Data Structure
- Lesion data: 4119 binary maps (91×109×91 voxels) in NIfTI format
- Targets: CSV with clinical scores, treatment assignments, and outcomes
- Data path: `data/participant-accessible/` (lesions.zip and tasks.csv)

## Running Tasks
Execute all three tasks:
```bash
python src/tasks.py --savepath /path/to/output
```

Models are saved as:
- `task1_model.pkl`: Regression model
- `task2_model.pkl`: Classification model
- `deficit_map.nii.gz`: Anatomical deficit map
- `treatment_map.nii.gz`: Treatment response map

## Current Implementation Status
The existing code uses dummy models (DummyRegressor/DummyClassifier) and random maps as placeholders. Real implementations need to be developed for each task.