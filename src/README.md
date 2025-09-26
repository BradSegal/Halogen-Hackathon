Welcome to the Holothon!

We have three related tasks to be completed today spanning *prediction*, *prescription* and *inference* in the context of focal brain injury.

We are investigating the functional consequences of focal brain injury and its response to treatment under the assumption of sensitivity to anatomically distributed effects.

The challenge is to devise an algorithmic approach to predicting outcomes, prescribing treatments, and inferring the fundamental functional organization of the brain, given a set of anatomical lesion patterns with matched functional deficit, treatment, and outcome data.

The dataset consists of 4119 anatomically registered binary lesion maps of real-world ischaemic damage of dimensions $91 * 109 * 91$ voxels, each with a corresponding deficit severity score (0-no deficit, 100 maximal), and--for those with a non-zero deficit--a treatment assignment variable (verum vs placebo) and a treatment outcome severity score (0-no deficit, 100 maximal). The value of each voxel indicates the presence or absence of damage at that anatomical location, in an anatomical space common to the group.

# Tasks
There are three modelling tasks:

## Task 1: Prediction
Build a predictive model of deficit severity, evaluated on held-out data from patients with non-zero severity with RMSE.

## Task 2: Prescription
Build a prescriptive model of individualized treatment response, evaluated on held-out data from patients assigned to treatment (verum vs placebo) with balanced accuracy for identifying responders.

## Task 3: Inference
Infer the topology (i.e. a dense anatomical map) of the deficit-related (zero vs non-zero) neural substrate, and the topology of the treatment response-related (responders vs non-responders) neural substrate. The latter is assumed to be a subset of the former. Both maps are common to the population.

Although the lesions are real, the deficit, assignment, and response variables are synthetic, derived from a hidden anatomical ground truth, under a realistically modelled relationship between the extent of overlap of a given lesion and the substrates critical to the deficit and its response to treatment. This formulation permits (otherwise impossible) evaluation of prescriptive performance and topological inference.

Each candidate solution will be evaluated at the end of the challenge with held-out data provided to each team, to be run within their development framework.

This is a hard challenge, with no established optimal solution. It has been chosen to provide maximum opportunity for the display of algorithmic courage, insight, and flair rather than skill in low-level model optimization. Performance will be equally weighted across the tasks (for 75% of the total mark), with the remainder allocated to the judges overall impression of the felicity of the approach.

### Setup
Check pip:
```
python3 -m pip --version
```
Install `python3 -m ensurepip --default-pip` if missing

Setup venv and install:
```
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```
