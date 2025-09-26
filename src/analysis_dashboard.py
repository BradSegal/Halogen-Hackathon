"""
Analysis dashboard for brain lesion data exploration and group comparisons.

This module provides comprehensive analysis tools for comparing treatment groups,
exploring correlations, and identifying patterns in brain lesion data.
"""

from typing import Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from visualization import BrainLesionVisualizer


class BrainLesionAnalyzer:
    """
    Advanced analysis toolkit for brain lesion data.

    This class provides comprehensive statistical and machine learning analysis
    of brain lesion patterns, treatment outcomes, and population-level trends.
    All analyses are designed to support clinical research in brain injury rehabilitation.
    """

    def __init__(self, visualizer: BrainLesionVisualizer):
        """
        Initialize analyzer with visualizer instance.

        Parameters:
        -----------
        visualizer : BrainLesionVisualizer
            Initialized visualizer instance containing lesion data and metadata
        """
        self.visualizer = visualizer
        self.tasks_df = visualizer.tasks_df

        # Cache for loaded lesion data to improve performance
        self._lesion_cache = {}

    @staticmethod
    def get_metric_explanations() -> Dict[str, str]:
        """
        Get detailed explanations of all metrics and calculations used in the analysis.

        Returns:
        --------
        Dict[str, str]
            Dictionary mapping metric names to their detailed explanations
        """
        return {
            "improvement_score": """
            **Improvement Score**: Clinical Score (baseline) - Outcome Score (post-treatment)

            • Positive values indicate improvement (reduced deficit severity)
            • Negative values indicate worsening
            • Zero indicates no change
            • Calculated only for patients with both baseline and outcome measurements
            """,
            "response_rate": """
            **Response Rate**: Percentage of patients showing any improvement (Improvement Score > 0)

            • Binary classification: Responder (improvement > threshold) vs Non-responder
            • Default threshold is 0 (any positive improvement)
            • Can be adjusted to require clinically meaningful improvement (e.g., ≥5 points)
            • Used to compare treatment effectiveness between groups
            """,
            "cohens_d": """
            **Cohen's d (Effect Size)**: Standardized measure of the difference between two group means

            Formula: (Mean₁ - Mean₂) / Pooled Standard Deviation

            Interpretation:
            • d = 0.2: Small effect
            • d = 0.5: Medium effect
            • d = 0.8: Large effect
            • Positive values favor the first group (typically treatment)
            """,
            "statistical_tests": """
            **Statistical Tests Used**:

            1. **Independent t-test**: Compares mean improvement between treatment/control groups
               - Assumes normal distribution and equal variances
               - p < 0.05 indicates statistically significant difference

            2. **Mann-Whitney U test**: Non-parametric alternative to t-test
               - Does not assume normal distribution
               - More robust to outliers and skewed data

            3. **Chi-square test**: Compares response rates between groups
               - Tests if proportions of responders differ significantly
               - Used for categorical outcome analysis
            """,
            "lesion_frequency_maps": """
            **Lesion Frequency Maps**: Voxel-wise probability maps showing lesion occurrence patterns

            Calculation:
            1. For each voxel position (x,y,z), count how many patients have a lesion at that location
            2. Divide by total number of patients in the group
            3. Result: probability (0-1) of lesion occurrence at each brain location

            Uses:
            • Identify common lesion sites within a group
            • Compare lesion patterns between responders/non-responders
            • Discover brain regions associated with specific outcomes
            """,
            "pca_clustering": """
            **PCA and Clustering Methodology**:

            1. **Principal Component Analysis (PCA)**:
               - Reduces high-dimensional lesion data (91×109×91 voxels) to lower dimensions
               - Captures main patterns of variance in lesion locations
               - First few components explain most variance in the data

            2. **K-means Clustering**:
               - Groups patients with similar lesion patterns
               - Applied to PCA-reduced data for computational efficiency
               - Number of clusters can be adjusted based on clinical needs

            3. **Cluster Analysis**:
               - Each cluster represents a distinct lesion pattern subtype
               - Can reveal different injury mechanisms or outcomes
               - Useful for personalized treatment approaches
            """,
            "volume_calculations": """
            **Lesion Volume Calculations**:

            Method:
            1. Count total number of voxels marked as lesioned (value = 1) in the 3D binary mask
            2. Each voxel represents a small cube of brain tissue
            3. Total volume = number of lesioned voxels × voxel size

            Clinical Relevance:
            • Larger lesions often correlate with more severe deficits
            • Volume alone doesn't predict outcome - location matters more
            • Used to control for lesion size when comparing treatments
            """,
            "population_heatmaps": """
            **Population Heatmaps**: Aggregate visualization of lesion patterns across groups

            Generation Process:
            1. Load all lesion masks for the specified group (e.g., treatment, control)
            2. Sum lesion masks voxel-wise across all patients
            3. Normalize by group size to get frequency values (0-1)
            4. Apply color mapping where warmer colors = higher lesion frequency

            Interpretation:
            • "Hot spots" indicate brain regions commonly affected in that group
            • Can reveal anatomical patterns associated with specific deficits
            • Comparison between groups shows differential lesion distributions
            """,
        }

    def _load_lesion_cached(self, lesion_id: str) -> np.ndarray:
        """
        Load lesion with caching to improve performance.

        Parameters:
        -----------
        lesion_id : str
            Lesion identifier

        Returns:
        --------
        np.ndarray
            Lesion data
        """
        if lesion_id not in self._lesion_cache:
            try:
                self._lesion_cache[lesion_id] = self.visualizer.load_lesion(lesion_id)
            except FileNotFoundError:
                self._lesion_cache[lesion_id] = None

        return self._lesion_cache[lesion_id]

    def analyze_treatment_efficacy(self) -> Dict:
        """
        Comprehensive analysis of treatment efficacy comparing treatment vs control groups.

        This method performs multiple statistical analyses to evaluate treatment effectiveness:

        **Calculations Performed:**
        1. **Improvement Score**: Clinical Score (baseline) - Outcome Score (post-treatment)
        2. **Group Comparisons**: Treatment group vs Control group mean improvements
        3. **Response Rate**: Percentage of patients with positive improvement (>0)
        4. **Statistical Tests**: t-test and Mann-Whitney U test for group differences
        5. **Effect Size**: Cohen's d to quantify the magnitude of treatment effect
        6. **Response Rate Comparison**: Chi-square test for proportional differences

        **Key Metrics Explained:**
        - **Cohen's d > 0.8**: Large effect size (clinically significant)
        - **p-value < 0.05**: Statistically significant difference
        - **Response Rate**: Simple binary classification of treatment success

        Returns:
        --------
        Dict
            Comprehensive results including:
            - Mean improvements for both groups
            - Response rates and counts
            - Statistical test results (t-test, Mann-Whitney U, Chi-square)
            - Effect size (Cohen's d)
            - Sample sizes for each group
        """
        # Filter treatment data
        treatment_data = self.tasks_df[
            (self.tasks_df["Treatment assignment"].isin(["Treatment", "Control"]))
            & (self.tasks_df["Outcome score"].notna())
        ].copy()

        # Calculate improvement scores
        treatment_data["Improvement"] = (
            treatment_data["Clinical score"] - treatment_data["Outcome score"]
        )

        # Separate groups
        treatment_group = treatment_data[
            treatment_data["Treatment assignment"] == "Treatment"
        ]["Improvement"]
        control_group = treatment_data[
            treatment_data["Treatment assignment"] == "Control"
        ]["Improvement"]

        # Statistical tests
        t_stat, p_value = stats.ttest_ind(treatment_group, control_group)
        u_stat, u_p_value = stats.mannwhitneyu(
            treatment_group, control_group, alternative="two-sided"
        )

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            (
                (len(treatment_group) - 1) * treatment_group.var()
                + (len(control_group) - 1) * control_group.var()
            )
            / (len(treatment_group) + len(control_group) - 2)
        )
        cohens_d = (treatment_group.mean() - control_group.mean()) / pooled_std

        # Response rates (improvement > 0)
        treatment_responders = (treatment_group > 0).sum()
        control_responders = (control_group > 0).sum()
        treatment_response_rate = treatment_responders / len(treatment_group)
        control_response_rate = control_responders / len(control_group)

        # Chi-square test for response rates
        contingency_table = np.array(
            [
                [treatment_responders, len(treatment_group) - treatment_responders],
                [control_responders, len(control_group) - control_responders],
            ]
        )
        chi2, chi2_p = stats.chi2_contingency(contingency_table)[:2]

        results = {
            "treatment_mean_improvement": treatment_group.mean(),
            "control_mean_improvement": control_group.mean(),
            "treatment_response_rate": treatment_response_rate,
            "control_response_rate": control_response_rate,
            "ttest_statistic": t_stat,
            "ttest_p_value": p_value,
            "mannwhitney_statistic": u_stat,
            "mannwhitney_p_value": u_p_value,
            "cohens_d": cohens_d,
            "chi2_statistic": chi2,
            "chi2_p_value": chi2_p,
            "n_treatment": len(treatment_group),
            "n_control": len(control_group),
        }

        return results

    def plot_efficacy_analysis(self) -> go.Figure:
        """
        Create comprehensive efficacy analysis visualization.

        Returns:
        --------
        go.Figure
            Multi-panel efficacy analysis plot
        """
        results = self.analyze_treatment_efficacy()

        # Get treatment data
        treatment_data = self.tasks_df[
            (self.tasks_df["Treatment assignment"].isin(["Treatment", "Control"]))
            & (self.tasks_df["Outcome score"].notna())
        ].copy()
        treatment_data["Improvement"] = (
            treatment_data["Clinical score"] - treatment_data["Outcome score"]
        )

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Improvement Distribution",
                "Response Rates",
                "Before vs After Treatment",
                "Improvement by Baseline Severity",
            ),
            specs=[
                [{"secondary_y": False}, {"type": "bar"}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
        )

        # 1. Improvement distribution
        treatment_group = treatment_data[
            treatment_data["Treatment assignment"] == "Treatment"
        ]["Improvement"]
        control_group = treatment_data[
            treatment_data["Treatment assignment"] == "Control"
        ]["Improvement"]

        fig.add_trace(
            go.Histogram(
                x=treatment_group,
                name="Treatment",
                opacity=0.7,
                nbinsx=15,
                legendgroup="treatment",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Histogram(
                x=control_group,
                name="Control",
                opacity=0.7,
                nbinsx=15,
                legendgroup="control",
            ),
            row=1,
            col=1,
        )

        # 2. Response rates
        fig.add_trace(
            go.Bar(
                x=["Treatment", "Control"],
                y=[
                    results["treatment_response_rate"],
                    results["control_response_rate"],
                ],
                name="Response Rate",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # 3. Before vs After
        for group, color in [("Treatment", "blue"), ("Control", "red")]:
            group_data = treatment_data[treatment_data["Treatment assignment"] == group]

            fig.add_trace(
                go.Scatter(
                    x=group_data["Clinical score"],
                    y=group_data["Outcome score"],
                    mode="markers",
                    name=f"{group} (Before vs After)",
                    marker=dict(color=color, opacity=0.6),
                    legendgroup=group.lower(),
                ),
                row=2,
                col=1,
            )

        # Add diagonal line
        max_score = max(
            treatment_data["Clinical score"].max(),
            treatment_data["Outcome score"].max(),
        )
        fig.add_trace(
            go.Scatter(
                x=[0, max_score],
                y=[0, max_score],
                mode="lines",
                line=dict(dash="dash", color="gray"),
                name="No change",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # 4. Improvement by baseline severity
        fig.add_trace(
            go.Scatter(
                x=treatment_data[treatment_data["Treatment assignment"] == "Treatment"][
                    "Clinical score"
                ],
                y=treatment_data[treatment_data["Treatment assignment"] == "Treatment"][
                    "Improvement"
                ],
                mode="markers",
                name="Treatment (Baseline vs Improvement)",
                marker=dict(color="blue", opacity=0.6),
                legendgroup="treatment",
            ),
            row=2,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=treatment_data[treatment_data["Treatment assignment"] == "Control"][
                    "Clinical score"
                ],
                y=treatment_data[treatment_data["Treatment assignment"] == "Control"][
                    "Improvement"
                ],
                mode="markers",
                name="Control (Baseline vs Improvement)",
                marker=dict(color="red", opacity=0.6),
                legendgroup="control",
            ),
            row=2,
            col=2,
        )

        # Update layout
        fig.update_xaxes(title_text="Improvement Score", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Response Rate", row=1, col=2)
        fig.update_xaxes(title_text="Baseline Score", row=2, col=1)
        fig.update_yaxes(title_text="Outcome Score", row=2, col=1)
        fig.update_xaxes(title_text="Baseline Score", row=2, col=2)
        fig.update_yaxes(title_text="Improvement", row=2, col=2)

        fig.update_layout(
            height=800,
            title_text=f"Treatment Efficacy Analysis<br>p-value: {results['ttest_p_value']:.4f}, Cohen's d: {results['cohens_d']:.3f}",
            showlegend=True,
        )

        return fig

    def identify_responsive_regions(self, min_improvement: float = 5.0) -> Dict:
        """
        Identify brain regions associated with treatment response using lesion frequency mapping.

        **Methodology:**
        This analysis compares lesion patterns between treatment responders and non-responders
        to identify brain regions where lesions are associated with better/worse outcomes.

        **Processing Steps:**
        1. **Patient Classification**: Divide treatment patients into responders (improvement ≥ threshold)
           and non-responders (improvement < threshold)
        2. **Lesion Frequency Calculation**: For each voxel, calculate the proportion of patients
           in each group who have a lesion at that location
        3. **Difference Mapping**: Subtract non-responder frequency from responder frequency
        4. **Interpretation**: Positive values indicate regions where lesions are more common
           in responders; negative values indicate regions more common in non-responders

        **Clinical Significance:**
        - Regions with higher lesion frequency in responders may indicate areas where
          damage doesn't prevent treatment benefit
        - Regions with higher frequency in non-responders may represent critical areas
          where lesions impede recovery

        Parameters:
        -----------
        min_improvement : float, default=5.0
            Minimum improvement score to classify a patient as a treatment responder.
            Higher thresholds require more substantial clinical improvement.

        Returns:
        --------
        Dict
            Analysis results containing:
            - responder_frequency_map: 3D array of lesion frequencies in responders
            - non_responder_frequency_map: 3D array of lesion frequencies in non-responders
            - difference_map: 3D array showing responder - non_responder differences
            - n_responders, n_non_responders: Sample sizes for each group
            - response_rate: Overall proportion of patients who responded to treatment
        """
        # Get treatment data
        treatment_data = self.tasks_df[
            (self.tasks_df["Treatment assignment"] == "Treatment")
            & (self.tasks_df["Outcome score"].notna())
        ].copy()

        treatment_data["Improvement"] = (
            treatment_data["Clinical score"] - treatment_data["Outcome score"]
        )
        treatment_data["Responder"] = treatment_data["Improvement"] >= min_improvement

        # Separate responders and non-responders
        responders = treatment_data[treatment_data["Responder"]]
        non_responders = treatment_data[~treatment_data["Responder"]]

        print(
            f"Found {len(responders)} responders and {len(non_responders)} non-responders"
        )

        # Calculate lesion frequency maps
        responder_map = np.zeros((91, 109, 91))
        non_responder_map = np.zeros((91, 109, 91))

        responder_count = 0
        for _, row in responders.iterrows():
            lesion_data = self._load_lesion_cached(row["lesion_id"])
            if lesion_data is not None:
                responder_map += lesion_data
                responder_count += 1

        non_responder_count = 0
        for _, row in non_responders.iterrows():
            lesion_data = self._load_lesion_cached(row["lesion_id"])
            if lesion_data is not None:
                non_responder_map += lesion_data
                non_responder_count += 1

        # Normalize to frequencies
        responder_freq = (
            responder_map / responder_count if responder_count > 0 else responder_map
        )
        non_responder_freq = (
            non_responder_map / non_responder_count
            if non_responder_count > 0
            else non_responder_map
        )

        # Calculate difference map (responder - non_responder)
        difference_map = responder_freq - non_responder_freq

        return {
            "responder_frequency_map": responder_freq,
            "non_responder_frequency_map": non_responder_freq,
            "difference_map": difference_map,
            "n_responders": responder_count,
            "n_non_responders": non_responder_count,
            "response_rate": responder_count / (responder_count + non_responder_count),
        }

    def plot_responsive_regions(self, min_improvement: float = 5.0) -> go.Figure:
        """
        Visualize brain regions associated with treatment response.

        Parameters:
        -----------
        min_improvement : float
            Minimum improvement score to be considered a responder

        Returns:
        --------
        go.Figure
            Heatmap showing regions associated with treatment response
        """
        results = self.identify_responsive_regions(min_improvement)

        # Use sagittal slice for visualization
        center_x = results["difference_map"].shape[0] // 2
        difference_slice = results["difference_map"][center_x, :, :]
        responder_slice = results["responder_frequency_map"][center_x, :, :]
        non_responder_slice = results["non_responder_frequency_map"][center_x, :, :]

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                f'Responders (n={results["n_responders"]})',
                f'Non-responders (n={results["n_non_responders"]})',
                "Difference (Responder - Non-responder)",
                "Response Rate Analysis",
            ),
            specs=[
                [{"type": "heatmap"}, {"type": "heatmap"}],
                [{"type": "heatmap"}, {"type": "bar"}],
            ],
        )

        # Responder frequency map
        fig.add_trace(
            go.Heatmap(
                z=responder_slice.T,
                colorscale="Reds",
                showscale=True,
                name="Responders",
            ),
            row=1,
            col=1,
        )

        # Non-responder frequency map
        fig.add_trace(
            go.Heatmap(
                z=non_responder_slice.T,
                colorscale="Blues",
                showscale=True,
                name="Non-responders",
            ),
            row=1,
            col=2,
        )

        # Difference map
        fig.add_trace(
            go.Heatmap(
                z=difference_slice.T,
                colorscale="RdBu_r",
                showscale=True,
                zmid=0,
                name="Difference",
            ),
            row=2,
            col=1,
        )

        # Response rate summary
        fig.add_trace(
            go.Bar(
                x=["Overall Response Rate"],
                y=[results["response_rate"]],
                name="Response Rate",
                showlegend=False,
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            height=800,
            title_text=f"Treatment Response Analysis (Improvement ≥ {min_improvement})",
            showlegend=False,
        )

        return fig

    def cluster_lesion_patterns(self, n_clusters: int = 5) -> Dict:
        """
        Cluster patients based on lesion patterns using dimensionality reduction and k-means clustering.

        **Methodology:**
        This unsupervised machine learning approach identifies distinct lesion pattern subtypes
        within the patient population, which can reveal different injury mechanisms or prognoses.

        **Processing Pipeline:**
        1. **Data Preparation**: Load and flatten 3D lesion masks into 1D feature vectors
           (91×109×91 = 902,629 features per patient)
        2. **Dimensionality Reduction**: Apply PCA to reduce to ~50 components while
           preserving most variance in lesion patterns
        3. **K-means Clustering**: Group patients with similar lesion patterns into clusters
        4. **Cluster Characterization**: Analyze clinical characteristics of each cluster
        5. **Treatment Response Analysis**: Compare response rates across clusters

        **Clinical Applications:**
        - **Personalized Medicine**: Different lesion patterns may respond differently to treatments
        - **Prognostic Modeling**: Clusters may predict different recovery trajectories
        - **Treatment Selection**: Identify which patients are likely to benefit from specific interventions
        - **Research Stratification**: Improve clinical trial design by accounting for lesion heterogeneity

        **Interpretation Guidelines:**
        - Clusters with high treatment response rates may represent "treatable" lesion patterns
        - Clusters with similar baseline scores but different responses suggest lesion location matters
        - Large, well-separated clusters indicate distinct lesion subtypes in the population

        Parameters:
        -----------
        n_clusters : int, default=5
            Number of distinct lesion pattern clusters to identify.
            Should be chosen based on clinical needs and data characteristics.

        Returns:
        --------
        Dict
            Comprehensive clustering analysis including:
            - cluster_labels: Assigned cluster for each patient
            - clustered_data: DataFrame with cluster assignments and clinical data
            - cluster_statistics: Summary statistics for each cluster
            - pca_components: Low-dimensional representation of lesion patterns
            - pca_explained_variance: Proportion of variance captured by each component
            - n_clusters: Number of clusters used
        """
        print("Loading lesion data for clustering...")

        # Collect lesion data
        lesion_vectors = []
        valid_indices = []

        for idx, row in self.tasks_df.iterrows():
            lesion_data = self._load_lesion_cached(row["lesion_id"])
            if lesion_data is not None:
                # Flatten lesion to 1D vector
                lesion_vector = lesion_data.flatten()
                lesion_vectors.append(lesion_vector)
                valid_indices.append(idx)

        if len(lesion_vectors) == 0:
            return {"error": "No valid lesion data found"}

        lesion_matrix = np.array(lesion_vectors)
        valid_df = self.tasks_df.iloc[valid_indices].copy()

        print(f"Clustering {len(lesion_vectors)} lesions...")

        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=min(50, len(lesion_vectors) - 1))
        lesion_pca = pca.fit_transform(lesion_matrix)

        # Apply clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(lesion_pca)

        # Add cluster labels to dataframe
        valid_df["Cluster"] = cluster_labels

        # Analyze cluster characteristics
        cluster_stats = []
        for cluster_id in range(n_clusters):
            cluster_data = valid_df[valid_df["Cluster"] == cluster_id]

            stats_dict = {
                "cluster_id": cluster_id,
                "n_patients": len(cluster_data),
                "mean_clinical_score": cluster_data["Clinical score"].mean(),
                "std_clinical_score": cluster_data["Clinical score"].std(),
                "treatment_patients": len(
                    cluster_data[cluster_data["Treatment assignment"] == "Treatment"]
                ),
                "control_patients": len(
                    cluster_data[cluster_data["Treatment assignment"] == "Control"]
                ),
            }

            # Calculate mean improvement for treatment patients
            treatment_cluster = cluster_data[
                (cluster_data["Treatment assignment"] == "Treatment")
                & (cluster_data["Outcome score"].notna())
            ]

            if len(treatment_cluster) > 0:
                improvements = (
                    treatment_cluster["Clinical score"]
                    - treatment_cluster["Outcome score"]
                )
                stats_dict["mean_improvement"] = improvements.mean()
                stats_dict["response_rate"] = (improvements > 0).mean()
            else:
                stats_dict["mean_improvement"] = np.nan
                stats_dict["response_rate"] = np.nan

            cluster_stats.append(stats_dict)

        return {
            "cluster_labels": cluster_labels,
            "clustered_data": valid_df,
            "cluster_statistics": pd.DataFrame(cluster_stats),
            "pca_components": lesion_pca,
            "pca_explained_variance": pca.explained_variance_ratio_,
            "n_clusters": n_clusters,
        }

    def plot_cluster_analysis(self, n_clusters: int = 5) -> go.Figure:
        """
        Visualize lesion pattern clustering results.

        Parameters:
        -----------
        n_clusters : int
            Number of clusters

        Returns:
        --------
        go.Figure
            Cluster analysis visualization
        """
        results = self.cluster_lesion_patterns(n_clusters)

        if "error" in results:
            fig = go.Figure()
            fig.add_annotation(
                text=results["error"],
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return fig

        clustered_data = results["clustered_data"]
        cluster_stats = results["cluster_statistics"]
        pca_components = results["pca_components"]

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "PCA Visualization (First 2 Components)",
                "Clinical Score by Cluster",
                "Treatment Response by Cluster",
                "Cluster Sizes",
            ),
        )

        # 1. PCA scatter plot
        colors = px.colors.qualitative.Set1[:n_clusters]
        for cluster_id in range(n_clusters):
            cluster_mask = clustered_data["Cluster"] == cluster_id
            cluster_pca = pca_components[cluster_mask]

            fig.add_trace(
                go.Scatter(
                    x=cluster_pca[:, 0],
                    y=cluster_pca[:, 1],
                    mode="markers",
                    name=f"Cluster {cluster_id}",
                    marker=dict(color=colors[cluster_id]),
                    legendgroup=f"cluster_{cluster_id}",
                ),
                row=1,
                col=1,
            )

        # 2. Clinical score by cluster
        fig.add_trace(
            go.Bar(
                x=cluster_stats["cluster_id"],
                y=cluster_stats["mean_clinical_score"],
                error_y=dict(type="data", array=cluster_stats["std_clinical_score"]),
                name="Mean Clinical Score",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # 3. Treatment response by cluster
        valid_response = cluster_stats.dropna(subset=["response_rate"])
        if len(valid_response) > 0:
            fig.add_trace(
                go.Bar(
                    x=valid_response["cluster_id"],
                    y=valid_response["response_rate"],
                    name="Response Rate",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

        # 4. Cluster sizes
        fig.add_trace(
            go.Bar(
                x=cluster_stats["cluster_id"],
                y=cluster_stats["n_patients"],
                name="Cluster Size",
                showlegend=False,
            ),
            row=2,
            col=2,
        )

        # Update layout
        fig.update_xaxes(title_text="PC1", row=1, col=1)
        fig.update_yaxes(title_text="PC2", row=1, col=1)
        fig.update_xaxes(title_text="Cluster", row=1, col=2)
        fig.update_yaxes(title_text="Clinical Score", row=1, col=2)
        fig.update_xaxes(title_text="Cluster", row=2, col=1)
        fig.update_yaxes(title_text="Response Rate", row=2, col=1)
        fig.update_xaxes(title_text="Cluster", row=2, col=2)
        fig.update_yaxes(title_text="Number of Patients", row=2, col=2)

        fig.update_layout(
            height=800,
            title_text=f"Lesion Pattern Clustering Analysis ({n_clusters} clusters)",
            showlegend=True,
        )

        return fig

    def generate_analysis_report(self) -> Dict:
        """
        Generate comprehensive analysis report.

        Returns:
        --------
        Dict
            Comprehensive analysis report
        """
        report = {
            "dataset_summary": {
                "total_patients": len(self.tasks_df),
                "patients_with_deficits": len(
                    self.tasks_df[self.tasks_df["Clinical score"] > 0]
                ),
                "treatment_patients": len(
                    self.tasks_df[self.tasks_df["Treatment assignment"] == "Treatment"]
                ),
                "control_patients": len(
                    self.tasks_df[self.tasks_df["Treatment assignment"] == "Control"]
                ),
                "mean_clinical_score": self.tasks_df["Clinical score"].mean(),
                "std_clinical_score": self.tasks_df["Clinical score"].std(),
            }
        }

        # Treatment efficacy analysis
        try:
            report["treatment_efficacy"] = self.analyze_treatment_efficacy()
        except Exception as e:
            report["treatment_efficacy"] = {"error": str(e)}

        # Response analysis
        try:
            response_analysis = self.identify_responsive_regions()
            report["response_analysis"] = {
                "response_rate": response_analysis["response_rate"],
                "n_responders": response_analysis["n_responders"],
                "n_non_responders": response_analysis["n_non_responders"],
            }
        except Exception as e:
            report["response_analysis"] = {"error": str(e)}

        # Volume analysis
        try:
            volume_df = self.visualizer.get_lesion_volumes_summary()
            report["volume_analysis"] = {
                "mean_volume": volume_df["volume"].mean(),
                "std_volume": volume_df["volume"].std(),
                "volume_clinical_correlation": volume_df["volume"].corr(
                    volume_df["Clinical score"]
                ),
            }
        except Exception as e:
            report["volume_analysis"] = {"error": str(e)}

        return report

    def get_analysis_methodology_guide(self) -> Dict[str, str]:
        """
        Get comprehensive methodology guide for all dashboard analyses.

        Returns:
        --------
        Dict[str, str]
            Detailed explanations of analysis methodologies and interpretations
        """
        return {
            "data_preprocessing": """
            ## Data Preprocessing and Quality Control

            **Lesion Data Processing:**
            1. **NIfTI File Loading**: Brain lesion masks loaded as 3D binary arrays (91×109×91 voxels)
            2. **Quality Checks**: Verification of file integrity and expected dimensions
            3. **Coordinate System**: Standard MNI space alignment for cross-patient comparison
            4. **Missing Data Handling**: Patients with corrupted/missing lesion files are excluded from spatial analyses

            **Clinical Data Processing:**
            1. **Score Validation**: Clinical and outcome scores checked for valid ranges
            2. **Treatment Group Assignment**: Verified assignment to Treatment/Control/N/A groups
            3. **Improvement Calculation**: Baseline score - Outcome score (higher = better recovery)
            4. **Outlier Detection**: Extreme values flagged but retained unless clearly erroneous
            """,
            "statistical_methodology": """
            ## Statistical Analysis Framework

            **Parametric vs Non-parametric Testing:**
            - **T-tests**: Used when data approximates normal distribution
            - **Mann-Whitney U**: Used for non-normal or ordinal data (more robust)
            - **Chi-square**: Used for categorical outcomes (responder/non-responder)

            **Multiple Comparisons:**
            - Results should be interpreted cautiously when multiple brain regions are analyzed
            - Consider Bonferroni or FDR correction for exploratory analyses
            - Focus on effect sizes, not just p-values

            **Effect Size Interpretation:**
            - **Cohen's d**: Standardized measure of group difference
            - **Clinical Significance**: Effect size ≥ 0.5 generally considered meaningful
            - **Statistical vs Clinical**: Significant p-value doesn't guarantee clinical relevance
            """,
            "brain_mapping_methods": """
            ## Brain Mapping and Spatial Analysis

            **Lesion Frequency Mapping:**
            1. **Voxel-wise Analysis**: Each brain location analyzed independently
            2. **Group Aggregation**: Lesion masks summed across patients in each group
            3. **Normalization**: Divided by group size to get proportion (0-1)
            4. **Statistical Mapping**: Can be extended with permutation testing

            **Population Heatmaps:**
            - **Color Interpretation**: Warmer colors = higher lesion frequency
            - **Anatomical Context**: Results should be interpreted with neuroanatomical knowledge
            - **Resolution Limits**: Analysis limited by original scan resolution
            - **Smoothing**: May be applied to reduce noise while preserving main patterns

            **Difference Maps:**
            - **Responder - Non-responder**: Positive values favor responders
            - **Treatment - Control**: Shows treatment-specific lesion patterns
            - **Threshold Selection**: Consider clinical significance when interpreting differences
            """,
            "machine_learning_approaches": """
            ## Machine Learning Methods

            **Principal Component Analysis (PCA):**
            - **Purpose**: Reduce 900K+ voxel features to manageable dimensions
            - **Variance Explained**: First 50 components typically capture 80-90% of variance
            - **Interpretation**: Components represent main patterns of lesion co-occurrence
            - **Limitations**: Components may not correspond to anatomically meaningful regions

            **K-means Clustering:**
            - **Algorithm**: Iteratively assigns patients to nearest cluster centroid
            - **Initialization**: Random seed fixed for reproducible results
            - **Cluster Count**: Should be chosen based on clinical interpretability
            - **Validation**: Consider silhouette analysis or clinical outcome differences

            **Cluster Interpretation:**
            - **Clinical Characteristics**: Compare baseline scores, treatment response across clusters
            - **Anatomical Patterns**: Visualize average lesion patterns for each cluster
            - **Outcome Prediction**: Clusters may predict treatment response or recovery trajectory
            """,
            "visualization_principles": """
            ## Visualization and Interpretation Guidelines

            **2D Brain Slice Visualization:**
            - **Slice Selection**: Typically sagittal (side view) for overview
            - **Anatomical Orientation**: Following radiological conventions (left/right)
            - **Color Schemes**: Consistent mapping across all visualizations
            - **Overlay Transparency**: Balance between lesion visibility and brain structure

            **Statistical Plot Interpretation:**
            - **Box Plots**: Show median, quartiles, and outliers for group comparisons
            - **Scatter Plots**: Reveal relationships and identify potential subgroups
            - **Histograms**: Show distribution shape and identify skewness
            - **Error Bars**: Represent standard error or confidence intervals

            **Interactive Features:**
            - **Parameter Adjustment**: Real-time updates for threshold changes
            - **Filtering**: Dynamic subsetting based on clinical criteria
            - **Zoom/Pan**: Detailed examination of brain regions
            - **Export Options**: High-resolution figures for publication
            """,
        }
