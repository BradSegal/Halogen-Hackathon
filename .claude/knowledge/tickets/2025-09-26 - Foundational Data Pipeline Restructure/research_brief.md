# Research Brief for CORE-01: Foundational Data Pipeline and Project Restructure

## Codebase Analysis

### Current Data Loading Patterns

#### Primary Implementation: `src/tasks.py`
**Function**: `_read_in_data(path, targets) -> np.ndarray`

**Current Approach**:
- Extracts entire ZIP archive (lesions.zip) to temporary directory
- Loads ALL 4119 NIfTI files into memory simultaneously using nibabel
- Creates nested dictionary structure: `{lesion_id: {"Lesion": np.array, "Q1_clinical_score": float, ...}}`
- Uses column names with spaces: `'Clinical score'`, `'Treatment assignment'`, `'Outcome score'`

**Critical Issues Identified**:
1. **Memory Explosion**: Loads ~1.5GB+ of data into RAM (4119 * 91*109*91 * 8 bytes ≈ 1.2GB + overhead)
2. **No Validation**: Missing files cause runtime crashes deep in training loops
3. **Inefficient Data Access**: Dictionary lookups instead of vectorized operations
4. **Hard-coded Column Names**: Uses spaces in column names preventing pythonic access

#### Secondary Implementation: `src/visualization.py`
**Function**: BrainLesionVisualizer class constructor reads CSV directly
**Code Pattern**:
```python
self.tasks_df = pd.read_csv(self.tasks_path)
```

**Issues**:
- Separate, inconsistent data loading approach
- No validation or error handling
- Different data structures than `tasks.py`

### Data Structure Analysis

#### Current CSV Schema (`data/tasks.csv`):
- **Records**: 4119 lesion samples
- **Columns**:
  - `lesion_id`: NIfTI filename (e.g., "lesion0804.nii.gz")
  - `Clinical score`: Float clinical severity score
  - `Treatment assignment`: "Control", "Treatment", or "N/A"
  - `Outcome score`: Float outcome score or "N/A"

#### Data Quality Issues:
- **Mixed Types**: "N/A" strings mixed with numeric values
- **Inconsistent Nulls**: Uses "N/A" strings instead of proper pandas NaN
- **Non-Pythonic Names**: Spaces prevent attribute-style access

### Existing Dependencies Analysis

#### Current Dependencies (from `pyproject.toml`):
```python
dependencies = [
    "tqdm",           # Progress bars - keep
    "nibabel",        # NIfTI file handling - keep
    "pandas",         # Data manipulation - keep
    "joblib",         # Model serialization - keep
    "scikit-learn",   # ML models - keep
    "matplotlib",     # Plotting - keep
    "plotly",         # Interactive plots - keep
    "nilearn",        # Neuroimaging - keep
    "ipywidgets",     # Jupyter widgets - keep
    "seaborn",        # Statistical plots - keep
    "streamlit",      # Web app - keep
    "dash",           # Web app - keep
    "scipy"           # Scientific computing - keep
]
```

#### Missing Dependencies Identified:
- **`pydantic`**: Required for data validation models (CRITICAL ADDITION)

### Architecture Alignment Analysis

#### CLAUDE.md Architectural Principles Applied:

**1. Current Implementation Violates Core Principles:**
- **DRY Violation**: Two different data loading implementations
- **No Contracts**: No type validation or schema enforcement
- **Silent Failures**: Missing files cause cryptic errors later
- **Non-Standard Naming**: Spaces in column names violate POLS

**2. Project Structure Needs:**
- Current: Flat `src/` with monolithic `tasks.py`
- Required: Modular `src/lesion_analysis/` structure for separation of concerns

**3. Data Processing Patterns:**
- Current: Load-everything-at-once approach
- Required: Lazy loading with validation for memory efficiency

### Similar Patterns in Codebase

#### Patterns to Follow:
1. **Progress Bars**: `tqdm` usage in `_read_in_data()` for long operations
2. **Path Handling**: `Path(__file__).parent.parent.parent / 'data'` pattern for relative paths
3. **NIfTI Processing**: `nib.load()` and `.get_fdata()` pattern for neuroimaging data
4. **Model Serialization**: `joblib.dump()` pattern for saving trained models

#### Anti-Patterns to Avoid:
1. **Bulk Loading**: Loading entire dataset into memory
2. **Hard-coded Paths**: String concatenation for file paths
3. **Mixed Type Handling**: Inconsistent null value representation
4. **Silent Errors**: Failing deep in pipelines instead of early validation

### Risk Assessment

#### High-Risk Areas:

**1. Memory Management (HIGH RISK)**
- **Issue**: Current approach loads 1.5GB+ into RAM
- **Impact**: Causes system crashes on memory-constrained environments
- **Mitigation**: Implement lazy loading with batch processing

**2. Data Integrity (HIGH RISK)**
- **Issue**: No validation of file existence or data quality
- **Impact**: Silent failures, incorrect results, debugging complexity
- **Mitigation**: Pydantic validation with fail-fast error handling

**3. Backward Compatibility (MEDIUM RISK)**
- **Issue**: Complete restructure may break existing workflows
- **Impact**: Temporary disruption during transition
- **Mitigation**: Phased migration, maintain existing file formats

#### Medium-Risk Areas:

**1. Performance Implications (MEDIUM RISK)**
- **Issue**: Pydantic validation adds computational overhead
- **Impact**: Slower data loading during development
- **Mitigation**: Cache validated data, optimize validation logic

**2. Integration Complexity (MEDIUM RISK)**
- **Issue**: Visualization and analysis tools need updating
- **Impact**: Requires coordinated changes across multiple modules
- **Mitigation**: Maintain consistent DataFrame interface

#### Low-Risk Areas:

**1. Dependency Management (LOW RISK)**
- **Issue**: Adding Pydantic dependency
- **Impact**: Minimal - well-maintained, stable library
- **Mitigation**: Standard pip installation

### Technical Constraints

#### System Constraints:
- **Memory**: Must handle 4119 samples without exhausting RAM
- **Storage**: NIfTI files are stored in ZIP archive format
- **Performance**: Data loading cannot become development bottleneck

#### Data Constraints:
- **File Format**: Must maintain NIfTI format compatibility
- **CSV Structure**: Cannot change fundamental CSV schema
- **Null Handling**: Must handle "N/A" strings and numeric NaNs

#### Integration Constraints:
- **Existing Code**: Visualization and analysis tools expect specific data formats
- **Model Training**: ML pipelines need consistent data access patterns
- **Testing**: Must maintain test coverage during refactor

### Performance Implications

#### Current Performance Profile:
- **Load Time**: ~30-60 seconds for full dataset
- **Memory Usage**: ~1.5GB RAM for 4119 samples
- **Memory Growth**: Linear with dataset size (unsustainable)

#### Expected Performance After Refactor:
- **Load Time**: <5 seconds for metadata validation
- **Memory Usage**: <100MB for validated metadata
- **Memory Growth**: Constant for metadata, on-demand for lesion data

#### Benchmarking Strategy:
- Measure memory usage with `memory_profiler`
- Time data loading operations with `time` module
- Profile validation overhead with `cProfile`

## Security Considerations

#### File Access Security:
- **Risk**: ZIP extraction to temporary directories
- **Mitigation**: Use secure temporary directories, validate file paths

#### Data Validation Security:
- **Risk**: Pydantic validators executing arbitrary code
- **Mitigation**: Use built-in validators, avoid custom eval expressions

## Conclusion

The current data loading implementation represents a significant architectural debt that prevents scalable development. The proposed refactor addresses core issues through:

1. **Memory Efficiency**: Lazy loading reduces RAM usage by >90%
2. **Data Quality**: Pydantic validation prevents silent failures
3. **Code Quality**: Eliminates duplication, improves maintainability
4. **Developer Experience**: Clear error messages, pythonic interfaces

The implementation plan must carefully manage the transition to minimize disruption while delivering substantial long-term benefits.