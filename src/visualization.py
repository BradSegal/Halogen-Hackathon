"""
Core visualization module for brain lesion analysis.

This module provides comprehensive visualization tools for analyzing
brain lesions, including 3D brain rendering, slice viewers, statistical
heatmaps, and population-level analysis.
"""

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from nilearn import plotting, datasets, image
from pathlib import Path
import tempfile
import zipfile
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class BrainLesionVisualizer:
    """
    Comprehensive visualization toolkit for brain lesion analysis.
    """
    
    def __init__(self, data_root: Union[str, Path]):
        """
        Initialize the visualizer.
        
        Parameters:
        -----------
        data_root : str or Path
            Path to the data directory containing lesions and tasks.csv
        """
        self.data_root = Path(data_root)
        self.lesions_path = self.data_root / "lesions"
        self.tasks_path = self.data_root / "tasks.csv"
        
        # Load tasks data
        self.tasks_df = pd.read_csv(self.tasks_path)
        
        # Get MNI template for anatomical reference
        try:
            self.mni_template = datasets.load_mni152_template(resolution=2)
        except:
            print("Warning: Could not load MNI template. 3D plotting may be limited.")
            self.mni_template = None
    
    def load_lesion(self, lesion_id: str) -> np.ndarray:
        """
        Load a specific lesion by ID.
        
        Parameters:
        -----------
        lesion_id : str
            Lesion identifier (e.g., 'lesion0000.nii.gz')
            
        Returns:
        --------
        np.ndarray
            3D lesion array
        """
        lesion_path = self.lesions_path / "lesions" / lesion_id
        if not lesion_path.exists():
            raise FileNotFoundError(f"Lesion {lesion_id} not found at {lesion_path}")
        
        img = nib.load(lesion_path)
        return img.get_fdata()
    
    def get_lesion_info(self, lesion_id: str) -> Dict:
        """
        Get metadata for a specific lesion.
        
        Parameters:
        -----------
        lesion_id : str
            Lesion identifier
            
        Returns:
        --------
        Dict
            Lesion metadata including clinical scores and treatment info
        """
        row = self.tasks_df[self.tasks_df['lesion_id'] == lesion_id]
        if row.empty:
            return {}
        
        return row.iloc[0].to_dict()
    
    def plot_lesion_slices(self, lesion_id: str, figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
        """
        Plot axial, sagittal, and coronal slices of a lesion.
        
        Parameters:
        -----------
        lesion_id : str
            Lesion identifier
        figsize : tuple
            Figure size
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure with lesion slices
        """
        lesion_data = self.load_lesion(lesion_id)
        lesion_info = self.get_lesion_info(lesion_id)
        
        # Find center of mass for slice selection
        coords = np.where(lesion_data > 0)
        if len(coords[0]) > 0:
            center_x = int(np.mean(coords[0]))
            center_y = int(np.mean(coords[1]))
            center_z = int(np.mean(coords[2]))
        else:
            center_x, center_y, center_z = lesion_data.shape[0]//2, lesion_data.shape[1]//2, lesion_data.shape[2]//2
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Axial slice
        axes[0].imshow(lesion_data[:, :, center_z].T, cmap='Reds', origin='lower')
        axes[0].set_title(f'Axial (z={center_z})')
        axes[0].axis('off')
        
        # Sagittal slice
        axes[1].imshow(lesion_data[center_x, :, :].T, cmap='Reds', origin='lower')
        axes[1].set_title(f'Sagittal (x={center_x})')
        axes[1].axis('off')
        
        # Coronal slice
        axes[2].imshow(lesion_data[:, center_y, :].T, cmap='Reds', origin='lower')
        axes[2].set_title(f'Coronal (y={center_y})')
        axes[2].axis('off')
        
        clinical_score = lesion_info.get('Clinical score', 'N/A')
        treatment = lesion_info.get('Treatment assignment', 'N/A')
        outcome = lesion_info.get('Outcome score', 'N/A')
        
        fig.suptitle(f'{lesion_id} | Clinical: {clinical_score} | Treatment: {treatment} | Outcome: {outcome}')
        plt.tight_layout()
        
        return fig
    
    def plot_3d_lesion(self, lesion_id: str) -> go.Figure:
        """
        Create interactive 3D plot of lesion.
        
        Parameters:
        -----------
        lesion_id : str
            Lesion identifier
            
        Returns:
        --------
        go.Figure
            Plotly 3D scatter plot
        """
        lesion_data = self.load_lesion(lesion_id)
        lesion_info = self.get_lesion_info(lesion_id)
        
        # Get lesion coordinates
        coords = np.where(lesion_data > 0)
        
        if len(coords[0]) == 0:
            print(f"No lesion voxels found in {lesion_id}")
            return go.Figure()
        
        x, y, z = coords
        
        fig = go.Figure(data=go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=2,
                color='red',
                opacity=0.6
            ),
            name='Lesion'
        ))
        
        clinical_score = lesion_info.get('Clinical score', 'N/A')
        treatment = lesion_info.get('Treatment assignment', 'N/A')
        outcome = lesion_info.get('Outcome score', 'N/A')
        
        fig.update_layout(
            title=f'3D Lesion: {lesion_id}<br>Clinical: {clinical_score} | Treatment: {treatment} | Outcome: {outcome}',
            scene=dict(
                xaxis_title='X (anterior-posterior)',
                yaxis_title='Y (left-right)', 
                zaxis_title='Z (inferior-superior)'
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def create_population_heatmap(self, group_filter: Optional[str] = None) -> go.Figure:
        """
        Create population-level lesion frequency heatmap.
        
        Parameters:
        -----------
        group_filter : str, optional
            Filter by treatment group ('Treatment', 'Control', or None for all)
            
        Returns:
        --------
        go.Figure
            Plotly heatmap showing lesion frequency across the population
        """
        # Filter data
        if group_filter:
            filtered_df = self.tasks_df[self.tasks_df['Treatment assignment'] == group_filter]
        else:
            filtered_df = self.tasks_df
        
        # Initialize accumulator
        lesion_sum = np.zeros((91, 109, 91))
        count = 0
        
        print(f"Processing {len(filtered_df)} lesions for population heatmap...")
        
        for _, row in filtered_df.iterrows():
            lesion_id = row['lesion_id']
            try:
                lesion_data = self.load_lesion(lesion_id)
                lesion_sum += lesion_data
                count += 1
            except FileNotFoundError:
                continue
            
            if count % 500 == 0:
                print(f"Processed {count} lesions...")
        
        # Calculate frequency map
        frequency_map = lesion_sum / count if count > 0 else lesion_sum
        
        # Create sagittal slice visualization
        center_slice = frequency_map.shape[0] // 2
        sagittal_slice = frequency_map[center_slice, :, :]
        
        fig = go.Figure(data=go.Heatmap(
            z=sagittal_slice.T,
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="Lesion Frequency")
        ))
        
        group_name = group_filter if group_filter else "All patients"
        fig.update_layout(
            title=f'Population Lesion Heatmap - {group_name}<br>Sagittal slice (n={count})',
            xaxis_title='Y (left-right)',
            yaxis_title='Z (inferior-superior)',
            width=600,
            height=500
        )
        
        return fig
    
    def plot_clinical_score_distribution(self) -> go.Figure:
        """
        Plot distribution of clinical scores.
        
        Returns:
        --------
        go.Figure
            Plotly histogram of clinical scores
        """
        fig = go.Figure()
        
        # All patients
        fig.add_trace(go.Histogram(
            x=self.tasks_df['Clinical score'],
            name='All patients',
            opacity=0.7,
            nbinsx=20
        ))
        
        # Treatment group
        treatment_scores = self.tasks_df[
            self.tasks_df['Treatment assignment'] == 'Treatment'
        ]['Clinical score']
        
        fig.add_trace(go.Histogram(
            x=treatment_scores,
            name='Treatment group',
            opacity=0.7,
            nbinsx=20
        ))
        
        # Control group
        control_scores = self.tasks_df[
            self.tasks_df['Treatment assignment'] == 'Control'
        ]['Clinical score']
        
        fig.add_trace(go.Histogram(
            x=control_scores,
            name='Control group',
            opacity=0.7,
            nbinsx=20
        ))
        
        fig.update_layout(
            title='Clinical Score Distribution',
            xaxis_title='Clinical Score',
            yaxis_title='Count',
            barmode='overlay',
            width=800,
            height=400
        )
        
        return fig
    
    def plot_treatment_outcome_analysis(self) -> go.Figure:
        """
        Analyze treatment outcomes.
        
        Returns:
        --------
        go.Figure
            Treatment outcome comparison plot
        """
        # Get treatment data only
        treatment_data = self.tasks_df[
            (self.tasks_df['Treatment assignment'].isin(['Treatment', 'Control'])) &
            (self.tasks_df['Outcome score'].notna())
        ].copy()
        
        # Calculate improvement
        treatment_data['Improvement'] = treatment_data['Clinical score'] - treatment_data['Outcome score']
        
        fig = go.Figure()
        
        # Treatment group
        treatment_group = treatment_data[treatment_data['Treatment assignment'] == 'Treatment']
        fig.add_trace(go.Box(
            y=treatment_group['Improvement'],
            name='Treatment',
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8
        ))
        
        # Control group
        control_group = treatment_data[treatment_data['Treatment assignment'] == 'Control']
        fig.add_trace(go.Box(
            y=control_group['Improvement'],
            name='Control',
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8
        ))
        
        fig.update_layout(
            title='Treatment Outcome Analysis<br>(Positive values indicate improvement)',
            yaxis_title='Improvement Score (Baseline - Outcome)',
            width=600,
            height=500
        )
        
        return fig
    
    def calculate_lesion_volume(self, lesion_id: str) -> float:
        """
        Calculate lesion volume in voxels.
        
        Parameters:
        -----------
        lesion_id : str
            Lesion identifier
            
        Returns:
        --------
        float
            Lesion volume in number of voxels
        """
        lesion_data = self.load_lesion(lesion_id)
        return np.sum(lesion_data > 0)
    
    def get_lesion_volumes_summary(self) -> pd.DataFrame:
        """
        Get summary statistics of lesion volumes across all patients.
        
        Returns:
        --------
        pd.DataFrame
            Summary statistics of lesion volumes
        """
        volumes = []
        lesion_ids = []
        
        print("Calculating lesion volumes...")
        
        for _, row in self.tasks_df.iterrows():
            lesion_id = row['lesion_id']
            try:
                volume = self.calculate_lesion_volume(lesion_id)
                volumes.append(volume)
                lesion_ids.append(lesion_id)
            except FileNotFoundError:
                continue
        
        volume_df = pd.DataFrame({
            'lesion_id': lesion_ids,
            'volume': volumes
        })
        
        # Merge with clinical data
        volume_df = volume_df.merge(self.tasks_df, on='lesion_id', how='left')
        
        return volume_df
    
    def plot_volume_vs_clinical_score(self) -> go.Figure:
        """
        Plot lesion volume vs clinical score.
        
        Returns:
        --------
        go.Figure
            Scatter plot of volume vs clinical score
        """
        volume_df = self.get_lesion_volumes_summary()
        
        fig = px.scatter(
            volume_df,
            x='volume',
            y='Clinical score',
            color='Treatment assignment',
            title='Lesion Volume vs Clinical Score',
            labels={'volume': 'Lesion Volume (voxels)', 'Clinical score': 'Clinical Score'}
        )
        
        fig.update_layout(width=800, height=500)
        
        return fig

# Utility functions for batch visualization
def visualize_random_lesions(visualizer: BrainLesionVisualizer, n_samples: int = 6) -> plt.Figure:
    """
    Visualize random sample of lesions.
    
    Parameters:
    -----------
    visualizer : BrainLesionVisualizer
        Initialized visualizer
    n_samples : int
        Number of lesions to sample
        
    Returns:
    --------
    plt.Figure
        Grid of lesion visualizations
    """
    # Sample lesions with non-zero clinical scores
    lesions_with_deficit = visualizer.tasks_df[visualizer.tasks_df['Clinical score'] > 0]
    sample_lesions = lesions_with_deficit.sample(n_samples)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (_, row) in enumerate(sample_lesions.iterrows()):
        lesion_id = row['lesion_id']
        try:
            lesion_data = visualizer.load_lesion(lesion_id)
            
            # Find center slice
            coords = np.where(lesion_data > 0)
            if len(coords[2]) > 0:
                center_z = int(np.mean(coords[2]))
            else:
                center_z = lesion_data.shape[2] // 2
            
            # Plot axial slice
            axes[i].imshow(lesion_data[:, :, center_z].T, cmap='Reds', origin='lower')
            axes[i].set_title(f'{lesion_id}\nScore: {row["Clinical score"]}')
            axes[i].axis('off')
            
        except FileNotFoundError:
            axes[i].text(0.5, 0.5, 'File not found', ha='center', va='center')
            axes[i].set_title(f'{lesion_id} - Missing')
            axes[i].axis('off')
    
    plt.tight_layout()
    return fig