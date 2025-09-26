"""
Interactive Streamlit app for brain lesion visualization and analysis.

This app provides an interactive interface for exploring brain lesion data,
analyzing treatment effects, and visualizing population-level patterns.

Usage:
    streamlit run src/app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from visualization import BrainLesionVisualizer, visualize_random_lesions
from analysis_dashboard import BrainLesionAnalyzer

# Page config
st.set_page_config(
    page_title="Brain Lesion Analysis Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_data():
    """Load and cache the brain lesion data."""
    data_root = Path(__file__).parent.parent / "data"
    if not data_root.exists():
        st.error(f"Data directory not found at {data_root}")
        st.stop()

    visualizer = BrainLesionVisualizer(data_root)
    analyzer = BrainLesionAnalyzer(visualizer)

    return visualizer, analyzer


def main():
    """Main Streamlit application."""

    st.title("🧠 Brain Lesion Analysis Dashboard")
    st.markdown("Interactive visualization and analysis of focal brain injury data")

    # Load data
    try:
        visualizer, analyzer = load_data()
        tasks_df = visualizer.tasks_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Analysis Mode",
        [
            "Dataset Overview",
            "Individual Lesion Explorer",
            "Population Analysis",
            "Treatment Efficacy",
            "Responsive Regions",
            "Lesion Clustering",
            "Volume Analysis",
            "Statistical Report",
            "Model Performance",
            "CNN Substrate Maps",
            "Map Statistics",
        ],
    )

    if app_mode == "Dataset Overview":
        show_dataset_overview(visualizer, tasks_df)

    elif app_mode == "Individual Lesion Explorer":
        show_individual_lesion_explorer(visualizer, tasks_df)

    elif app_mode == "Population Analysis":
        show_population_analysis(visualizer, tasks_df)

    elif app_mode == "Treatment Efficacy":
        show_treatment_efficacy(analyzer)

    elif app_mode == "Responsive Regions":
        show_responsive_regions(analyzer)

    elif app_mode == "Lesion Clustering":
        show_lesion_clustering(analyzer)

    elif app_mode == "Volume Analysis":
        show_volume_analysis(visualizer)

    elif app_mode == "Statistical Report":
        show_statistical_report(analyzer)

    elif app_mode == "Model Performance":
        show_model_performance(analyzer)

    elif app_mode == "CNN Substrate Maps":
        show_cnn_substrate_maps(analyzer)

    elif app_mode == "Map Statistics":
        show_map_statistics(analyzer)


def show_dataset_overview(visualizer, tasks_df):
    """Show dataset overview and summary statistics."""

    st.header("Dataset Overview")

    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Patients", len(tasks_df))

    with col2:
        patients_with_deficits = len(tasks_df[tasks_df["clinical_score"] > 0])
        st.metric("Patients with Deficits", patients_with_deficits)

    with col3:
        treatment_patients = len(
            tasks_df[tasks_df["treatment_assignment"] == "Treatment"]
        )
        st.metric("Treatment Group", treatment_patients)

    with col4:
        control_patients = len(tasks_df[tasks_df["treatment_assignment"] == "Control"])
        st.metric("Control Group", control_patients)

    # Clinical score distribution
    st.subheader("Clinical Score Distribution")
    fig = visualizer.plot_clinical_score_distribution()
    st.plotly_chart(fig, use_container_width=True)

    # Sample lesions
    st.subheader("Sample Lesion Visualizations")

    if st.button("Generate New Random Sample"):
        st.rerun()

    try:
        # Use matplotlib for sample lesions
        import matplotlib.pyplot as plt

        fig_sample = visualize_random_lesions(visualizer, n_samples=6)
        st.pyplot(fig_sample)
        plt.close()
    except Exception as e:
        st.warning(f"Could not generate sample lesions: {e}")

    # Data table
    st.subheader("Data Sample")
    st.dataframe(tasks_df.head(20), use_container_width=True)


def show_individual_lesion_explorer(visualizer, tasks_df):
    """Individual lesion exploration interface."""

    st.header("Individual Lesion Explorer")

    # Lesion selection
    lesion_ids = tasks_df["lesion_id"].tolist()

    # Filter options
    col1, col2 = st.columns(2)

    with col1:
        filter_by_score = st.checkbox("Filter by Clinical Score")
        if filter_by_score:
            min_score = st.slider("Minimum Clinical Score", 0, 100, 10)
            filtered_df = tasks_df[tasks_df["clinical_score"] >= min_score]
            lesion_ids = filtered_df["lesion_id"].tolist()

    with col2:
        filter_by_treatment = st.checkbox("Filter by Treatment")
        if filter_by_treatment:
            treatment_type = st.selectbox(
                "Treatment Type", ["Treatment", "Control", "N/A"]
            )
            filtered_df = tasks_df[tasks_df["treatment_assignment"] == treatment_type]
            lesion_ids = filtered_df["lesion_id"].tolist()

    if not lesion_ids:
        st.warning("No lesions match the selected criteria.")
        return

    # Select lesion
    selected_lesion = st.selectbox("Select Lesion", lesion_ids, index=0)

    if selected_lesion:
        # Display lesion info
        lesion_info = visualizer.get_lesion_info(selected_lesion)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Clinical Score", lesion_info.get("clinical_score", "N/A"))

        with col2:
            st.metric("Treatment", lesion_info.get("treatment_assignment", "N/A"))

        with col3:
            st.metric("Outcome Score", lesion_info.get("outcome_score", "N/A"))

        with col4:
            if pd.notna(lesion_info.get("clinical_score", np.nan)) and pd.notna(
                lesion_info.get("outcome_score", np.nan)
            ):
                improvement = (
                    lesion_info["clinical_score"] - lesion_info["outcome_score"]
                )
                st.metric("Improvement", f"{improvement:.1f}")

        # Visualization tabs
        tab1, tab2, tab3 = st.tabs(["2D Slices", "3D Visualization", "Volume Info"])

        with tab1:
            try:
                import matplotlib.pyplot as plt

                fig_slices = visualizer.plot_lesion_slices(selected_lesion)
                st.pyplot(fig_slices)
                plt.close()
            except Exception as e:
                st.error(f"Error creating slice visualization: {e}")

        with tab2:
            try:
                fig_3d = visualizer.plot_3d_lesion(selected_lesion)
                st.plotly_chart(fig_3d, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating 3D visualization: {e}")

        with tab3:
            try:
                volume = visualizer.calculate_lesion_volume(selected_lesion)
                st.metric("Lesion Volume (voxels)", int(volume))

                # Volume context
                volume_df = visualizer.get_lesion_volumes_summary()
                percentile = (volume_df["volume"] <= volume).mean() * 100
                st.info(f"This lesion is larger than {percentile:.1f}% of all lesions")

            except Exception as e:
                st.error(f"Error calculating volume: {e}")


def show_population_analysis(visualizer, tasks_df):
    """Population-level analysis visualization."""

    st.header("Population Analysis")

    # Add methodology explanation
    with st.expander("🌍 Understanding Population Analysis"):
        st.markdown("### Population Heatmap Methodology")
        st.markdown(
            """
        **How Population Heatmaps are Generated:**

        1. **Data Aggregation**: Combine lesion masks from all patients in the selected group
        2. **Frequency Calculation**: For each brain voxel, calculate what percentage of patients have a lesion
        3. **Normalization**: Convert counts to frequencies (0-1 or 0-100%)
        4. **Visualization**: Apply color mapping where intensity reflects lesion frequency

        **Color Interpretation:**
        - 🔴 **Red/Hot colors**: Brain regions frequently lesioned in this population
        - 🔵 **Blue/Cool colors**: Brain regions rarely lesioned
        - ⚫ **Black/Dark areas**: Brain regions never lesioned in this sample
        """
        )

        st.markdown("### 📊 Clinical Applications")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
            **Research Applications:**
            - Identify common lesion sites for specific conditions
            - Compare lesion patterns between treatment groups
            - Design targeted interventions for frequently affected regions
            - Understand population-level brain vulnerability patterns
            """
            )

        with col2:
            st.markdown(
                """
            **Clinical Insights:**
            - "Hot spots" may represent anatomically vulnerable regions
            - Group differences suggest differential selection criteria
            - Sparse regions may indicate areas less critical for the studied deficit
            - Pattern consistency across patients suggests common pathophysiology
            """
            )

        st.markdown("### ⚠️ Interpretation Considerations")
        st.warning(
            """
        **Important Factors:**

        📊 **Sample Size**: Larger groups provide more reliable frequency estimates

        🎯 **Selection Bias**: Treatment group assignment may influence lesion patterns

        🧠 **Anatomical Constraints**: Some brain regions are naturally more vulnerable to certain injury types

        📍 **Spatial Resolution**: Analysis is limited by original image resolution

        ⚖️ **Statistical Significance**: High frequency doesn't necessarily mean causal relationship
        """
        )

    # Group selection
    group_filter = st.selectbox(
        "Select Group",
        ["All patients", "Treatment", "Control"],
        index=0,
        help="Compare lesion patterns between different patient groups",
    )

    group_param = None if group_filter == "All patients" else group_filter

    st.subheader("Population Lesion Heatmap")
    if group_filter != "All patients":
        st.info(
            f"🎯 Showing lesion frequency patterns for **{group_filter}** group patients only"
        )

    with st.spinner("Generating population heatmap... This may take a few minutes."):
        try:
            fig_heatmap = visualizer.create_population_heatmap(group_param)
            st.plotly_chart(fig_heatmap, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating population heatmap: {e}")

    # Volume distribution
    st.subheader("Lesion Volume Distribution")

    try:
        volume_df = visualizer.get_lesion_volumes_summary()

        fig_volume = px.histogram(
            volume_df,
            x="volume",
            color="treatment_assignment",
            title="Distribution of Lesion Volumes",
            labels={"volume": "Lesion Volume (voxels)"},
            nbins=30,
        )

        st.plotly_chart(fig_volume, use_container_width=True)

        # Volume statistics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Mean Volume", f"{volume_df['volume'].mean():.0f} voxels")

        with col2:
            st.metric("Median Volume", f"{volume_df['volume'].median():.0f} voxels")

        with col3:
            st.metric("Max Volume", f"{volume_df['volume'].max():.0f} voxels")

    except Exception as e:
        st.error(f"Error in volume analysis: {e}")


def show_treatment_efficacy(analyzer):
    """Treatment efficacy analysis."""

    st.header("Treatment Efficacy Analysis")

    # Add methodology explanation
    with st.expander("📖 Understanding Treatment Efficacy Analysis"):
        explanations = analyzer.get_metric_explanations()

        st.markdown("### Key Metrics Explained")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(explanations["improvement_score"])
            st.markdown(explanations["response_rate"])

        with col2:
            st.markdown(explanations["cohens_d"])
            st.markdown(explanations["statistical_tests"])

        st.markdown("### 📊 Interpretation Guide")
        st.info(
            """
        **How to interpret the results:**

        🎯 **Primary Endpoint**: Mean improvement score difference between treatment and control groups

        📈 **Statistical Significance**: p < 0.05 indicates treatment effect is unlikely due to chance

        📏 **Clinical Significance**: Cohen's d ≥ 0.5 suggests meaningful clinical benefit

        ✅ **Response Rate**: Percentage showing any improvement - useful for patient counseling

        ⚖️ **Multiple Tests**: Both t-test (parametric) and Mann-Whitney U (non-parametric) provided for robustness
        """
        )

    # Statistical summary
    try:
        efficacy_results = analyzer.analyze_treatment_efficacy()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Treatment Mean Improvement",
                f"{efficacy_results['treatment_mean_improvement']:.2f}",
            )

        with col2:
            st.metric(
                "Control Mean Improvement",
                f"{efficacy_results['control_mean_improvement']:.2f}",
            )

        with col3:
            st.metric(
                "Treatment Response Rate",
                f"{efficacy_results['treatment_response_rate']:.2%}",
            )

        with col4:
            st.metric(
                "Control Response Rate",
                f"{efficacy_results['control_response_rate']:.2%}",
            )

        # Statistical significance
        st.subheader("Statistical Tests")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**T-test Results**")
            st.write(f"t-statistic: {efficacy_results['ttest_statistic']:.3f}")
            st.write(f"p-value: {efficacy_results['ttest_p_value']:.4f}")

            if efficacy_results["ttest_p_value"] < 0.05:
                st.success("Statistically significant difference (p < 0.05)")
            else:
                st.warning("No statistically significant difference (p ≥ 0.05)")

        with col2:
            st.write("**Effect Size**")
            st.write(f"Cohen's d: {efficacy_results['cohens_d']:.3f}")

            if abs(efficacy_results["cohens_d"]) > 0.8:
                effect_size = "Large"
            elif abs(efficacy_results["cohens_d"]) > 0.5:
                effect_size = "Medium"
            elif abs(efficacy_results["cohens_d"]) > 0.2:
                effect_size = "Small"
            else:
                effect_size = "Negligible"

            st.write(f"Effect size: {effect_size}")

        # Comprehensive visualization
        st.subheader("Detailed Analysis")

        fig_efficacy = analyzer.plot_efficacy_analysis()
        st.plotly_chart(fig_efficacy, use_container_width=True)

    except Exception as e:
        st.error(f"Error in treatment efficacy analysis: {e}")


def show_responsive_regions(analyzer):
    """Treatment responsive regions analysis."""

    st.header("Treatment Responsive Regions")

    # Add methodology explanation
    with st.expander("🧠 Understanding Responsive Regions Analysis"):
        explanations = analyzer.get_metric_explanations()

        st.markdown("### Analysis Methodology")
        st.markdown(explanations["lesion_frequency_maps"])

        st.markdown("### 🎨 Color Map Interpretation")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
            **Responder vs Non-responder Maps:**
            - 🔴 **Red/Warm colors**: Higher lesion frequency
            - 🔵 **Blue/Cool colors**: Lower lesion frequency
            - ⚫ **Dark areas**: No lesions in this region
            """
            )

        with col2:
            st.markdown(
                """
            **Difference Map:**
            - 🔴 **Red regions**: More lesions in responders
            - 🔵 **Blue regions**: More lesions in non-responders
            - ⚪ **White/neutral**: Similar frequencies
            """
            )

        st.markdown("### 🔬 Clinical Interpretation")
        st.warning(
            """
        **Important Considerations:**

        🧩 **Paradoxical Findings**: Regions with more lesions in responders may indicate:
        - Areas where damage doesn't prevent treatment benefit
        - Compensatory mechanisms or alternate pathways
        - Selection bias in treatment assignment

        🎯 **Critical Regions**: Areas with more lesions in non-responders may represent:
        - Brain regions essential for treatment response
        - Areas where damage predicts poor prognosis
        - Targets for future therapeutic development

        📊 **Sample Size**: Results are more reliable with larger, balanced groups
        """
        )

    # Parameter selection
    min_improvement = st.slider(
        "Minimum Improvement Score (to be considered responder)",
        min_value=0.0,
        max_value=20.0,
        value=5.0,
        step=0.5,
        help="Higher thresholds require more substantial clinical improvement to classify as 'responder'",
    )

    st.subheader("Brain Regions Associated with Treatment Response")

    with st.spinner("Analyzing responsive regions... This may take a few minutes."):
        try:
            fig_responsive = analyzer.plot_responsive_regions(min_improvement)
            st.plotly_chart(fig_responsive, use_container_width=True)

            # Get additional statistics
            results = analyzer.identify_responsive_regions(min_improvement)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Overall Response Rate", f"{results['response_rate']:.2%}")

            with col2:
                st.metric("Number of Responders", results["n_responders"])

            with col3:
                st.metric("Number of Non-responders", results["n_non_responders"])

            st.info(
                """
            **Interpretation Guide:**
            - **Red regions** in the difference map indicate areas where lesions are more common in responders
            - **Blue regions** indicate areas where lesions are more common in non-responders
            - Darker colors indicate stronger associations
            """
            )

        except Exception as e:
            st.error(f"Error in responsive regions analysis: {e}")


def show_lesion_clustering(analyzer):
    """Lesion pattern clustering analysis."""

    st.header("Lesion Pattern Clustering")

    # Add methodology explanation
    with st.expander("🤖 Understanding Lesion Pattern Clustering"):
        explanations = analyzer.get_metric_explanations()

        st.markdown("### Machine Learning Approach")
        st.markdown(explanations["pca_clustering"])

        st.markdown("### 📈 Visualization Components")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
            **PCA Scatter Plot:**
            - Each dot represents one patient
            - Colors show cluster assignments
            - Distance between dots reflects similarity
            - Well-separated clusters indicate distinct patterns
            """
            )

        with col2:
            st.markdown(
                """
            **Cluster Statistics:**
            - Clinical scores by cluster
            - Treatment response rates
            - Patient counts per cluster
            - Baseline characteristics comparison
            """
            )

        st.markdown("### 🎯 Clinical Applications")
        st.success(
            """
        **Potential Uses:**

        🏥 **Personalized Treatment**: Identify which lesion patterns respond best to specific treatments

        📊 **Prognostic Modeling**: Predict recovery based on lesion pattern subtype

        🔬 **Research Stratification**: Improve clinical trial design by accounting for lesion heterogeneity

        🧠 **Mechanism Discovery**: Understand different injury types and recovery pathways

        ⚡ **Treatment Selection**: Guide clinical decisions about intervention choices
        """
        )

        st.markdown("### ⚠️ Important Considerations")
        st.warning(
            """
        - **Cluster Count**: Start with 3-5 clusters for interpretability
        - **Sample Size**: Each cluster should have meaningful sample sizes (>20 patients)
        - **Clinical Validation**: Clusters should make anatomical/clinical sense
        - **Reproducibility**: Results may vary with different random initializations
        """
        )

    # Parameter selection
    n_clusters = st.slider(
        "Number of Clusters",
        min_value=2,
        max_value=10,
        value=5,
        step=1,
        help="More clusters provide finer distinctions but may be harder to interpret clinically",
    )

    st.subheader("Clustering Analysis")
    st.info(
        "🔬 This analysis uses unsupervised machine learning to identify distinct lesion pattern subtypes in your patient population."
    )

    if st.button("Run Clustering Analysis"):
        with st.spinner(
            "Performing clustering analysis... This may take several minutes."
        ):
            try:
                fig_clustering = analyzer.plot_cluster_analysis(n_clusters)
                st.plotly_chart(fig_clustering, use_container_width=True)

                # Get clustering results for summary
                results = analyzer.cluster_lesion_patterns(n_clusters)

                if "error" not in results:
                    cluster_stats = results["cluster_statistics"]

                    st.subheader("Cluster Summary")
                    st.dataframe(cluster_stats, use_container_width=True)

                    # Interpretation
                    st.subheader("Interpretation")
                    st.write(
                        f"""
                    The analysis identified {n_clusters} distinct lesion patterns:
                    - **PCA Visualization**: Shows how different the lesion patterns are
                    - **Clinical Score by Cluster**: Different patterns may be associated with different severities
                    - **Treatment Response**: Some patterns may respond better to treatment
                    - **Cluster Sizes**: Distribution of patients across patterns
                    """
                    )

            except Exception as e:
                st.error(f"Error in clustering analysis: {e}")


def show_volume_analysis(visualizer):
    """Lesion volume analysis."""

    st.header("Lesion Volume Analysis")

    # Add methodology explanation
    with st.expander("📏 Understanding Volume Analysis"):
        st.markdown("### Volume Calculation Method")
        st.markdown(
            """
        **How Lesion Volumes are Calculated:**

        1. **Binary Mask Processing**: Each lesion is represented as a 3D binary mask where:
           - 1 = lesioned brain tissue
           - 0 = healthy brain tissue

        2. **Voxel Counting**: Volume = total number of voxels marked as lesioned
           - Each voxel represents a small cube of brain tissue
           - Standard resolution provides clinically meaningful measurements

        3. **Units**: Reported in voxels (can be converted to mm³ if voxel size is known)
        """
        )

        st.markdown("### 📊 Clinical Significance")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
            **Volume-Deficit Relationship:**
            - Generally, larger lesions → more severe deficits
            - But location matters more than size alone
            - Small lesions in critical areas can be devastating
            - Large lesions in "silent" areas may have minimal impact
            """
            )

        with col2:
            st.markdown(
                """
            **Volume-Recovery Relationship:**
            - Very large lesions may have limited recovery potential
            - Medium-sized lesions often show variable outcomes
            - Small lesions may recover more completely
            - Individual variation is substantial
            """
            )

        st.markdown("### 🎯 Analysis Components")
        st.info(
            """
        **What the visualizations show:**

        📈 **Volume vs Clinical Score**: Relationship between lesion size and deficit severity

        🔄 **Volume vs Treatment Response**: Whether lesion size predicts treatment benefit

        📊 **Distribution Plots**: Comparison of volume patterns between treatment groups

        📈 **Correlation Analysis**: Statistical relationship between volume and outcomes
        """
        )

        st.markdown("### ⚠️ Interpretation Guidelines")
        st.warning(
            """
        **Important Considerations:**

        🎯 **Location Trumps Size**: A small lesion in a critical area (e.g., motor cortex) may cause more severe deficits than a large lesion in a less critical region

        📊 **Non-linear Relationships**: The relationship between volume and outcome is rarely linear

        🧠 **Compensatory Mechanisms**: The brain can sometimes compensate for even large lesions

        ⚖️ **Treatment Implications**: Volume may predict treatment response, but individual factors matter more
        """
        )

    with st.spinner("Calculating lesion volumes..."):
        try:
            # Volume vs clinical score
            fig_volume = visualizer.plot_volume_vs_clinical_score()
            st.plotly_chart(fig_volume, use_container_width=True)

            # Volume distribution by treatment outcome
            volume_df = visualizer.get_lesion_volumes_summary()

            # Add improvement calculation
            treatment_volume_df = volume_df[
                (volume_df["treatment_assignment"].isin(["Treatment", "Control"]))
                & (volume_df["outcome_score"].notna())
            ].copy()

            if len(treatment_volume_df) > 0:
                treatment_volume_df["Improvement"] = (
                    treatment_volume_df["clinical_score"]
                    - treatment_volume_df["outcome_score"]
                )

                st.subheader("Volume vs Treatment Response")

                fig_volume_response = px.scatter(
                    treatment_volume_df,
                    x="volume",
                    y="Improvement",
                    color="treatment_assignment",
                    title="Lesion Volume vs Treatment Improvement",
                    labels={
                        "volume": "Lesion Volume (voxels)",
                        "Improvement": "Improvement Score",
                    },
                )

                st.plotly_chart(fig_volume_response, use_container_width=True)

            # Volume statistics
            st.subheader("Volume Statistics")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Mean Volume", f"{volume_df['volume'].mean():.0f}")

            with col2:
                st.metric("Median Volume", f"{volume_df['volume'].median():.0f}")

            with col3:
                st.metric("Standard Deviation", f"{volume_df['volume'].std():.0f}")

            with col4:
                correlation = volume_df["volume"].corr(volume_df["clinical_score"])
                st.metric("Volume-Clinical Correlation", f"{correlation:.3f}")

            # Volume distribution
            st.subheader("Volume Distribution")

            fig_dist = px.box(
                volume_df,
                y="volume",
                color="treatment_assignment",
                title="Lesion Volume Distribution by Treatment Group",
            )

            st.plotly_chart(fig_dist, use_container_width=True)

        except Exception as e:
            st.error(f"Error in volume analysis: {e}")


def show_statistical_report(analyzer):
    """Generate and display comprehensive statistical report."""

    st.header("Statistical Report")

    # Add methodology explanation
    with st.expander("📊 Understanding the Statistical Report"):
        methodology = analyzer.get_analysis_methodology_guide()

        st.markdown("### Report Components")
        st.markdown(
            """
        This comprehensive report synthesizes all analyses performed on your brain lesion dataset:

        🎯 **Dataset Summary**: Basic descriptive statistics and sample characteristics

        📈 **Treatment Efficacy**: Statistical comparison of treatment vs control outcomes

        🧠 **Response Analysis**: Identification of treatment responders and their characteristics

        📏 **Volume Analysis**: Relationship between lesion size and clinical outcomes
        """
        )

        st.markdown("### Statistical Framework")
        st.markdown(methodology["statistical_methodology"])

        st.markdown("### 📋 Report Interpretation Guide")
        st.success(
            """
        **How to Use This Report:**

        📖 **Clinical Context**: Always interpret statistical results within clinical context

        🔬 **Multiple Comparisons**: Consider correction for multiple testing in exploratory analyses

        📊 **Effect Sizes**: Focus on clinical significance (effect sizes) not just p-values

        🎯 **Sample Sizes**: Ensure adequate power for detected effects

        ⚖️ **Limitations**: Consider selection bias, missing data, and confounding variables
        """
        )

        st.markdown("### Data Quality Considerations")
        st.info(
            """
        **Before interpreting results:**

        ✅ Check sample sizes for each analysis
        ✅ Verify data completeness and quality
        ✅ Consider potential confounding variables
        ✅ Assess clinical significance alongside statistical significance
        """
        )

    if st.button("Generate Comprehensive Report"):
        with st.spinner("Generating comprehensive analysis report..."):
            try:
                report = analyzer.generate_analysis_report()

                # Dataset summary
                st.subheader("Dataset Summary")
                summary = report["dataset_summary"]

                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**Total Patients:** {summary['total_patients']}")
                    st.write(
                        f"**Patients with Deficits:** {summary['patients_with_deficits']}"
                    )
                    st.write(f"**Treatment Patients:** {summary['treatment_patients']}")

                with col2:
                    st.write(f"**Control Patients:** {summary['control_patients']}")
                    st.write(
                        f"**Mean Clinical Score:** {summary['mean_clinical_score']:.2f} ± {summary['std_clinical_score']:.2f}"
                    )

                # Treatment efficacy
                if "error" not in report["treatment_efficacy"]:
                    st.subheader("Treatment Efficacy Results")
                    efficacy = report["treatment_efficacy"]

                    st.write(
                        f"**Treatment Group:** {efficacy['treatment_mean_improvement']:.2f} mean improvement"
                    )
                    st.write(
                        f"**Control Group:** {efficacy['control_mean_improvement']:.2f} mean improvement"
                    )
                    st.write(
                        f"**Statistical Significance:** p = {efficacy['ttest_p_value']:.4f}"
                    )
                    st.write(f"**Effect Size (Cohen's d):** {efficacy['cohens_d']:.3f}")

                # Response analysis
                if "error" not in report["response_analysis"]:
                    st.subheader("Response Analysis")
                    response = report["response_analysis"]

                    st.write(
                        f"**Overall Response Rate:** {response['response_rate']:.2%}"
                    )
                    st.write(f"**Number of Responders:** {response['n_responders']}")
                    st.write(
                        f"**Number of Non-responders:** {response['n_non_responders']}"
                    )

                # Volume analysis
                if "error" not in report["volume_analysis"]:
                    st.subheader("Volume Analysis")
                    volume = report["volume_analysis"]

                    st.write(
                        f"**Mean Lesion Volume:** {volume['mean_volume']:.0f} ± {volume['std_volume']:.0f} voxels"
                    )
                    st.write(
                        f"**Volume-Clinical Correlation:** {volume['volume_clinical_correlation']:.3f}"
                    )

                # Download report
                report_text = f"""
Brain Lesion Analysis Report
============================

Dataset Summary:
- Total Patients: {summary['total_patients']}
- Patients with Deficits: {summary['patients_with_deficits']}
- Treatment Patients: {summary['treatment_patients']}
- Control Patients: {summary['control_patients']}
- Mean Clinical Score: {summary['mean_clinical_score']:.2f} ± {summary['std_clinical_score']:.2f}

Treatment Efficacy:
- Treatment Mean Improvement: {efficacy.get('treatment_mean_improvement', 'N/A')}
- Control Mean Improvement: {efficacy.get('control_mean_improvement', 'N/A')}
- p-value: {efficacy.get('ttest_p_value', 'N/A')}
- Effect Size: {efficacy.get('cohens_d', 'N/A')}

Response Analysis:
- Response Rate: {response.get('response_rate', 'N/A')}
- Responders: {response.get('n_responders', 'N/A')}
- Non-responders: {response.get('n_non_responders', 'N/A')}

Volume Analysis:
- Mean Volume: {volume.get('mean_volume', 'N/A')} voxels
- Volume-Clinical Correlation: {volume.get('volume_clinical_correlation', 'N/A')}
                """

                st.download_button(
                    label="Download Report",
                    data=report_text,
                    file_name="brain_lesion_analysis_report.txt",
                    mime="text/plain",
                )

            except Exception as e:
                st.error(f"Error generating report: {e}")


def show_model_performance(analyzer):
    """Display model performance comparison."""

    st.header("🎯 Model Performance Comparison")

    with st.expander("ℹ️ About Model Performance Metrics", expanded=False):
        st.markdown("""
        This section compares the performance of different models trained for the brain lesion analysis tasks:

        **Task 1 - Severity Prediction (Regression)**:
        - **RMSE (Root Mean Squared Error)**: Lower values indicate better performance
        - Measures how accurately the model predicts clinical severity scores

        **Task 2 - Treatment Response Classification**:
        - **BACC (Balanced Accuracy)**: Higher values indicate better performance
        - Measures how well the model predicts treatment response (responder vs non-responder)
        - Balanced accuracy accounts for class imbalance

        **Models Evaluated**:
        - **Baseline**: Simple linear models using downsampled voxel features
        - **Atlas**: Models using atlas-based ROI features
        - **CNN**: Convolutional Neural Network models learning directly from 3D images
        """)

    # Display performance comparison
    fig = analyzer.plot_model_performance()
    st.plotly_chart(fig, use_container_width=True)

    # Display evaluation results table if available
    eval_df = analyzer.load_evaluation_results()
    if eval_df is not None:
        st.subheader("📊 Detailed Results")

        # Format the dataframe for better display
        eval_df_display = eval_df.copy()
        eval_df_display['Score'] = eval_df_display['Score'].round(4)
        eval_df_display.columns = ['Model/Metric', 'Score']

        st.dataframe(
            eval_df_display,
            use_container_width=True,
            hide_index=True
        )

        # Add download button for results
        csv = eval_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results CSV",
            data=csv,
            file_name="model_evaluation_results.csv",
            mime="text/csv"
        )


def show_cnn_substrate_maps(analyzer):
    """Display CNN substrate maps with interactive controls."""

    st.header("🧠 CNN Substrate Maps")

    with st.expander("ℹ️ About Substrate Maps", expanded=False):
        st.markdown("""
        Substrate maps identify brain regions associated with specific deficits or treatment responses:

        **Deficit Map**:
        - Shows brain regions where lesions correlate with higher clinical deficit scores
        - Generated using saliency/attention mechanisms from trained CNN models
        - Warmer colors indicate stronger association with deficits

        **Treatment Map**:
        - Shows brain regions where lesions predict treatment response
        - Identifies areas important for determining who will benefit from treatment
        - Can guide personalized treatment decisions

        **Map Comparison**:
        - **Baseline**: Generated using traditional lesion-symptom mapping
        - **CNN**: Generated using deep learning saliency/explainability methods
        - **Difference**: Shows where CNN and baseline maps diverge
        """)

    # View mode selection
    view_mode = st.radio(
        "View Mode",
        ["Compare Models (Side-by-side)", "All Maps (Grid View)", "Single Map Type"],
        horizontal=True
    )

    # Controls
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        slice_axis = st.selectbox(
            "Select View",
            ["sagittal", "coronal", "axial"],
            format_func=lambda x: x.capitalize()
        )

    # Get the maps to determine slice range
    maps = analyzer.load_result_maps()

    if maps:
        # Get the first available map to determine dimensions
        sample_map = next(iter(maps.values()))
        axis_idx = {'sagittal': 0, 'coronal': 1, 'axial': 2}[slice_axis]
        max_slice = sample_map.shape[axis_idx] - 1

        with col2:
            slice_idx = st.slider(
                "Slice Index",
                min_value=0,
                max_value=max_slice,
                value=max_slice // 2,
                help=f"Navigate through {slice_axis} slices"
            )

        if view_mode == "Single Map Type":
            with col3:
                map_type = st.selectbox(
                    "Map Type",
                    ["deficit", "treatment"],
                    format_func=lambda x: x.capitalize()
                )
            show_all = False
        elif view_mode == "All Maps (Grid View)":
            map_type = "deficit"  # Default, not used when show_all=True
            show_all = True
        else:  # Compare Models
            with col3:
                map_type = st.selectbox(
                    "Map Type to Compare",
                    ["deficit", "treatment"],
                    format_func=lambda x: x.capitalize()
                )
            show_all = False

        # Display the maps
        fig = analyzer.plot_result_maps(map_type, slice_axis, slice_idx, show_all=show_all)
        st.plotly_chart(fig, use_container_width=True)

        # Show available maps summary
        st.subheader("📂 Available Maps")
        available_maps = analyzer.load_result_maps()
        if available_maps:
            cols = st.columns(len(available_maps))
            for idx, (map_name, map_data) in enumerate(available_maps.items()):
                with cols[idx]:
                    non_zero = np.sum(map_data != 0)
                    st.metric(
                        label=map_name.replace('_', ' ').title(),
                        value=f"{non_zero:,} voxels",
                        delta=f"{(non_zero/map_data.size)*100:.1f}% active"
                    )

        # Add export options
        st.subheader("📥 Export Options")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📊 Export Current View as Image"):
                st.info("To save the image, right-click on the plot and select 'Save as image'")

        with col2:
            if st.button("ℹ️ Map Information"):
                with st.expander("Map Details", expanded=True):
                    for map_name in available_maps.keys():
                        st.write(f"**{map_name.replace('_', ' ').title()}**")
                        if 'baseline' in map_name:
                            st.write("- Method: Traditional lesion-symptom mapping")
                        elif 'cnn' in map_name:
                            st.write("- Method: CNN saliency/gradient analysis")
                        if 'deficit' in map_name:
                            st.write("- Purpose: Identifies regions associated with clinical deficits")
                        elif 'treatment' in map_name:
                            st.write("- Purpose: Identifies regions predictive of treatment response")
    else:
        st.warning("No result maps found. Please run the model training and map generation scripts first.")

        with st.expander("🔧 How to Generate Maps", expanded=True):
            st.code("""
# 1. Train models
python scripts/train_baseline_models.py
python scripts/train_cnn_models.py

# 2. Generate substrate maps
python scripts/generate_cnn_maps.py
            """, language='bash')


def show_map_statistics(analyzer):
    """Display statistical analysis of result maps."""

    st.header("📈 Map Statistics & Analysis")

    with st.expander("ℹ️ About Map Statistics", expanded=False):
        st.markdown("""
        Statistical analysis of the generated substrate maps provides insights into:

        **Mean Values**: Average activation/importance across the brain
        **Standard Deviation**: Variability in map values
        **Non-zero Voxels**: Number of brain regions identified as relevant
        **Sparsity**: Proportion of brain that is not implicated (higher = more focused)

        These statistics help:
        - Compare different mapping approaches (baseline vs CNN)
        - Assess map reliability and focus
        - Identify potential issues (e.g., overly sparse or dense maps)
        """)

    # Display statistics
    maps = analyzer.load_result_maps()

    if maps:
        # Show statistical comparison
        fig = analyzer.plot_map_statistics()
        st.plotly_chart(fig, use_container_width=True)

        # Create detailed statistics table
        st.subheader("📊 Detailed Statistics")

        stats_data = []
        for map_name, map_data in maps.items():
            non_zero = np.sum(map_data != 0)
            total = map_data.size

            stats_data.append({
                'Map': map_name.replace('_', ' ').title(),
                'Mean': f"{np.mean(map_data):.6f}",
                'Std': f"{np.std(map_data):.6f}",
                'Min': f"{np.min(map_data):.6f}",
                'Max': f"{np.max(map_data):.6f}",
                'Non-zero Voxels': f"{non_zero:,}",
                'Total Voxels': f"{total:,}",
                'Sparsity': f"{(1 - non_zero/total):.2%}"
            })

        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

        # Add interpretation guide
        st.subheader("🔍 Interpretation Guide")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Deficit Maps**:
            - Higher mean values → Stronger overall deficit associations
            - Lower sparsity → More widespread deficit patterns
            - Compare baseline vs CNN to see methodological differences
            """)

        with col2:
            st.markdown("""
            **Treatment Maps**:
            - Higher std deviation → More heterogeneous response patterns
            - Higher sparsity → More focal treatment targets
            - Non-zero voxels indicate potential treatment targets
            """)

        # Download statistics
        csv = pd.DataFrame(stats_data).to_csv(index=False)
        st.download_button(
            label="📥 Download Statistics CSV",
            data=csv,
            file_name="map_statistics.csv",
            mime="text/csv"
        )
    else:
        st.warning("No result maps found. Please generate maps first using the training scripts.")


if __name__ == "__main__":
    main()
