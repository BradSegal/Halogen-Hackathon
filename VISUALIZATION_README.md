# Brain Lesion Visualization System

This comprehensive visualization system provides tools for analyzing brain lesion data, treatment effects, and population-level patterns for the Halogen Hackathon brain injury challenge.

## Quick Start

### 1. Launch Interactive Dashboard
```bash
streamlit run src/app.py
```

### 2. Explore with Jupyter Notebooks
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

## System Components

### 🎯 Interactive Streamlit App (`src/app.py`)
- **Dataset Overview**: Summary statistics and sample visualizations
- **Individual Lesion Explorer**: Browse and analyze specific lesions
- **Population Analysis**: Frequency maps and group comparisons
- **Treatment Efficacy**: Statistical analysis of treatment effects  
- **Responsive Regions**: Identify brain areas associated with treatment response
- **Lesion Clustering**: Pattern-based patient grouping
- **Volume Analysis**: Lesion size vs clinical outcomes
- **Statistical Reports**: Comprehensive analysis summaries

### 📊 Core Visualization Library (`src/visualization.py`)
- **BrainLesionVisualizer**: Main visualization class
- **3D Brain Rendering**: Interactive lesion visualization
- **2D Slice Viewers**: Axial, sagittal, coronal views
- **Population Heatmaps**: Aggregate frequency maps
- **Statistical Plots**: Clinical score distributions and correlations

### 🔬 Analysis Dashboard (`src/analysis_dashboard.py`)
- **BrainLesionAnalyzer**: Advanced analysis toolkit
- **Treatment Efficacy Analysis**: Statistical tests and effect sizes
- **Responsive Region Detection**: Identify treatment-predictive areas
- **Pattern Clustering**: Machine learning-based patient grouping
- **Comprehensive Reports**: Automated analysis summaries

### 📓 Jupyter Notebooks
1. **01_data_exploration.ipynb**: Dataset overview and initial visualization
2. **02_lesion_analysis.ipynb**: Advanced pattern analysis and clustering
3. **03_treatment_response.ipynb**: Treatment effect analysis and prediction

## Key Features

### 🧠 Brain Visualization
- Interactive 3D lesion models with rotation and zoom
- Multi-slice 2D viewers (axial, sagittal, coronal)
- Population-level frequency heatmaps
- Anatomical overlay with MNI template

### 📈 Statistical Analysis
- Treatment vs control group comparisons
- Effect size calculations (Cohen's d)
- Response rate analysis with confidence intervals
- ANOVA for group differences
- Correlation analysis

### 🎯 Treatment Response Analysis
- Responder vs non-responder pattern identification
- Brain region responsiveness mapping
- Predictive feature extraction
- Machine learning classification models

### 🔍 Advanced Analytics
- Lesion volume calculations and correlations
- Regional distribution analysis
- Pattern-based patient clustering
- Cross-validation and model evaluation

## Usage Examples

### Basic Visualization

```python
from src.visualization import BrainLesionVisualizer

# Initialize visualizer
visualizer = BrainLesionVisualizer('data/')

# Visualize specific lesion
fig = visualizer.plot_lesion_slices('lesion0000.nii.gz')
fig.show()

# Create 3D visualization
fig_3d = visualizer.plot_3d_lesion('lesion0000.nii.gz')
fig_3d.show()
```

### Treatment Analysis
```python
from src.analysis_dashboard import BrainLesionAnalyzer

# Initialize analyzer
analyzer = BrainLesionAnalyzer(visualizer)

# Analyze treatment efficacy
results = analyzer.analyze_treatment_efficacy()
print(f"p-value: {results['ttest_p_value']:.4f}")

# Visualize efficacy analysis
fig = analyzer.plot_efficacy_analysis()
fig.show()
```

### Population Analysis
```python
# Create population heatmap
fig = visualizer.create_population_heatmap('Treatment')
fig.show()

# Analyze volume correlations
fig = visualizer.plot_volume_vs_clinical_score()
fig.show()
```

## Performance Notes

- **Population analyses** may take several minutes due to large dataset size
- **Interactive Streamlit app** provides optimized performance with progress indicators
- **Jupyter notebooks** include sample-based analysis for faster exploration
- Consider using **data subsets** for initial exploration of new analysis methods

## File Structure
```
src/
├── app.py                     # Streamlit interactive dashboard
├── visualization.py           # Core visualization library
├── analysis_dashboard.py      # Advanced analysis toolkit
└── tasks.py                   # Original challenge implementation

notebooks/
├── 01_data_exploration.ipynb  # Initial data exploration
├── 02_lesion_analysis.ipynb   # Pattern analysis and clustering
└── 03_treatment_response.ipynb # Treatment effect analysis

data/
├── lesions/                   # Brain lesion NIfTI files
└── tasks.csv                  # Clinical scores and treatment data
```

## Dependencies

All visualization dependencies are automatically installed with:
```bash
pip install -e .
```

Key libraries: `streamlit`, `plotly`, `matplotlib`, `nilearn`, `seaborn`, `scikit-learn`

## Tips for Best Results

1. **Start with the Streamlit app** for interactive exploration
2. **Use Jupyter notebooks** for reproducible analysis and customization
3. **Sample data strategically** for computational efficiency during development
4. **Export visualizations** using built-in save functions
5. **Combine multiple views** (2D + 3D) for comprehensive understanding

This visualization system provides a complete toolkit for exploring the brain lesion dataset and understanding treatment effects. Each component is designed to work independently or as part of an integrated analysis workflow.