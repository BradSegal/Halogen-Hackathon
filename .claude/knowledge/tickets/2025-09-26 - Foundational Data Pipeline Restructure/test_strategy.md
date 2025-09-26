# Test Strategy for CORE-01: Foundational Data Pipeline and Project Restructure

## Testing Overview

This test strategy ensures comprehensive validation of the new data pipeline architecture, focusing on data integrity, validation logic, error handling, and integration stability. The approach emphasizes fail-fast validation, clear error messaging, and maintainable test patterns.

## Testing Principles

1. **Test Data Quality**: Validate that all data transformations preserve integrity
2. **Test Error Paths**: Ensure validation failures produce clear, actionable error messages
3. **Test Edge Cases**: Handle boundary conditions and malformed data gracefully
4. **Test Integration Points**: Validate end-to-end workflows work correctly
5. **Test Performance**: Ensure memory efficiency and acceptable load times

## Unit Test Plan

### Test Module: `tests/data/test_loader.py`

#### Test Class: `TestLesionRecord`
Tests for the Pydantic data model validation.

**Test Case: `test_lesion_record_valid_data()`**
- **Purpose**: Verify valid data passes validation
- **Setup**: Create LesionRecord with valid fields
- **Assertions**:
  - Model instantiates successfully
  - All fields contain expected values
  - Field types match specification

**Test Case: `test_lesion_record_na_treatment_conversion()`**
- **Purpose**: Verify "N/A" strings convert to None for treatment_assignment
- **Setup**: Create LesionRecord with treatment_assignment="N/A"
- **Assertions**:
  - `treatment_assignment` field becomes None
  - Other fields remain unchanged

**Test Case: `test_lesion_record_nan_outcome_conversion()`**
- **Purpose**: Verify NaN values convert to None for outcome_score
- **Setup**: Create LesionRecord with outcome_score=np.nan
- **Assertions**:
  - `outcome_score` field becomes None
  - Other fields remain unchanged

**Test Case: `test_lesion_record_missing_file_validation()`**
- **Purpose**: Verify file path validation catches missing files
- **Setup**: Create LesionRecord with non-existent lesion_filepath
- **Assertions**:
  - Raises ValueError
  - Error message contains specific file path
  - Error message is actionable

**Test Case: `test_lesion_record_invalid_types()`**
- **Purpose**: Verify type validation catches invalid data
- **Setup**: Create LesionRecord with invalid field types (e.g., string for clinical_score)
- **Assertions**:
  - Raises ValidationError
  - Error specifies invalid field and value
  - No partial validation occurs

#### Test Class: `TestLoadAndPrepareData`
Tests for the core data loading function.

**Test Case: `test_load_and_prepare_data_happy_path()`**
- **Purpose**: Verify successful data loading with valid inputs
- **Setup**:
  - Create temporary CSV with 5 valid records
  - Create temporary lesion files matching CSV lesion_ids
  - Call load_and_prepare_data() with temp paths
- **Assertions**:
  - Returns pandas DataFrame with correct shape (5 rows)
  - Contains all required columns: lesion_id, clinical_score, treatment_assignment, outcome_score, lesion_filepath, is_responder
  - `is_responder` calculated correctly: `outcome_score < clinical_score`
  - `treatment_assignment` "N/A" values converted to pd.NA
  - All lesion_filepath values are valid Path objects

**Test Case: `test_load_and_prepare_data_missing_csv()`**
- **Purpose**: Verify error handling for missing CSV file
- **Setup**: Call load_and_prepare_data() with non-existent CSV path
- **Assertions**:
  - Raises FileNotFoundError
  - Error message contains specific CSV path
  - Error occurs before processing any data

**Test Case: `test_load_and_prepare_data_missing_lesions_dir()`**
- **Purpose**: Verify error handling for missing lesions directory
- **Setup**: Call load_and_prepare_data() with non-existent lesions directory
- **Assertions**:
  - Raises FileNotFoundError
  - Error message contains specific directory path
  - Error occurs before processing any data

**Test Case: `test_load_and_prepare_data_missing_lesion_file()`**
- **Purpose**: Verify validation catches missing individual lesion files
- **Setup**:
  - Create CSV referencing lesion file that doesn't exist
  - Create lesions directory without the referenced file
- **Assertions**:
  - Raises ValueError (from Pydantic validator)
  - Error message contains specific missing file path
  - No partial DataFrame is returned

**Test Case: `test_load_and_prepare_data_malformed_csv()`**
- **Purpose**: Verify handling of malformed CSV data
- **Setup**:
  - Create CSV with invalid clinical_score (non-numeric)
  - Create corresponding lesion files
- **Assertions**:
  - Raises ValidationError
  - Error specifies the problematic field and value
  - No data processing occurs past validation point

**Test Case: `test_load_and_prepare_data_is_responder_calculation()`**
- **Purpose**: Verify is_responder feature engineering logic
- **Setup**:
  - Create CSV with known clinical_score and outcome_score pairs
  - Test cases: outcome < clinical (True), outcome >= clinical (False), outcome is NaN (NaN)
- **Assertions**:
  - `is_responder` column created correctly
  - Boolean dtype maintained
  - NaN values handled appropriately

#### Test Class: `TestDataTypeHandling`
Tests for consistent data type handling across the pipeline.

**Test Case: `test_na_string_handling()`**
- **Purpose**: Verify consistent "N/A" string processing
- **Setup**: Create data with various "N/A" representations
- **Assertions**:
  - All "N/A" strings converted to pd.NA in DataFrame
  - Converted to None in Pydantic models
  - No inconsistent null representations remain

**Test Case: `test_numeric_nan_handling()`**
- **Purpose**: Verify consistent numeric NaN processing
- **Setup**: Create data with np.nan and pd.NA values
- **Assertions**:
  - All numeric NaN values handled consistently
  - Boolean operations work correctly with NaN values
  - No numeric conversion errors occur

## Integration Test Plan

### Test Module: `tests/integration/test_data_pipeline.py`

#### Test Class: `TestDataPreparationPipeline`
Tests for end-to-end data preparation workflows.

**Test Case: `test_end_to_end_data_preparation()`**
- **Purpose**: Verify complete data preparation pipeline
- **Setup**:
  - Use subset of actual data/tasks.csv (first 100 records)
  - Create corresponding subset of lesion files
  - Execute scripts/prepare_data.py programmatically
- **Assertions**:
  - Script executes without errors
  - Generates data/processed/train.csv and data/processed/test.csv
  - Split sizes are correct (80/20 proportion)
  - Both files contain all required columns
  - No data leakage between train/test splits
  - Stratification maintains treatment_assignment distribution

**Test Case: `test_reproducible_data_splits()`**
- **Purpose**: Verify train/test splits are reproducible
- **Setup**:
  - Run data preparation script twice with same inputs
  - Compare generated train.csv and test.csv files
- **Assertions**:
  - Identical train/test splits across runs
  - Same random_state produces same results
  - File contents are byte-for-byte identical

**Test Case: `test_stratified_sampling()`**
- **Purpose**: Verify treatment assignment distribution is preserved
- **Setup**: Run data preparation with full dataset
- **Assertions**:
  - Train and test sets have similar treatment_assignment proportions
  - Chi-square test shows no significant distribution difference
  - All treatment categories represented in both splits

## Edge Case Test Plan

### Test Module: `tests/edge_cases/test_edge_cases.py`

#### Boundary Condition Tests

**Test Case: `test_empty_csv()`**
- **Purpose**: Handle empty CSV files gracefully
- **Setup**: Create empty CSV file
- **Expected**: Clear error message, no crashes

**Test Case: `test_single_record_csv()`**
- **Purpose**: Handle minimum viable dataset
- **Setup**: CSV with only 1 record
- **Expected**: Processes successfully, appropriate warnings for train/test split

**Test Case: `test_large_clinical_scores()`**
- **Purpose**: Handle extreme numeric values
- **Setup**: Very large positive/negative clinical scores
- **Expected**: Values preserved accurately, no overflow errors

**Test Case: `test_unicode_lesion_ids()`**
- **Purpose**: Handle non-ASCII characters in lesion IDs
- **Setup**: Lesion IDs with unicode characters
- **Expected**: Processed correctly or clear validation error

#### Malformed Data Tests

**Test Case: `test_mixed_data_types_in_columns()`**
- **Purpose**: Handle inconsistent data types within columns
- **Setup**: Mix of numeric and string values in clinical_score column
- **Expected**: Clear validation errors identifying problematic records

**Test Case: `test_extra_csv_columns()`**
- **Purpose**: Handle CSV files with unexpected additional columns
- **Setup**: CSV with extra columns beyond required schema
- **Expected**: Extra columns ignored, required columns processed

**Test Case: `test_missing_required_columns()`**
- **Purpose**: Handle CSV files missing required columns
- **Setup**: CSV missing clinical_score column
- **Expected**: Clear error identifying missing required columns

## Performance Test Plan

### Test Module: `tests/performance/test_performance.py`

#### Memory Efficiency Tests

**Test Case: `test_memory_usage_large_dataset()`**
- **Purpose**: Verify memory efficiency with large datasets
- **Setup**: Process full 4119-record dataset with memory monitoring
- **Assertions**:
  - Peak memory usage <200MB (significant improvement from 1.5GB+)
  - Memory usage grows sublinearly with dataset size
  - No memory leaks during processing

**Test Case: `test_load_time_performance()`**
- **Purpose**: Verify acceptable data loading performance
- **Setup**: Time data loading operations with various dataset sizes
- **Assertions**:
  - Full dataset validation completes in <30 seconds
  - Load time grows linearly with dataset size
  - Performance acceptable for development workflows

#### Scalability Tests

**Test Case: `test_scalability_stress_test()`**
- **Purpose**: Verify system handles dataset growth
- **Setup**: Test with hypothetical larger datasets (10K, 20K records)
- **Expected**: Graceful scaling, predictable resource usage

## Test Data Management

### Test Fixtures Strategy

**Fixture: `mock_csv_data()`**
- **Purpose**: Generate consistent test CSV data
- **Content**: 5-10 records with known values for predictable testing
- **Usage**: Shared across multiple test functions

**Fixture: `temp_lesion_files()`**
- **Purpose**: Create temporary NIfTI files for testing
- **Content**: Minimal valid NIfTI files (small arrays)
- **Cleanup**: Automatic cleanup after test completion

**Fixture: `invalid_data_scenarios()`**
- **Purpose**: Parameterized test data for validation testing
- **Content**: Various invalid data combinations
- **Usage**: Test error handling comprehensively

### Test Environment Setup

**Requirements**:
- pytest for test execution
- pytest-mock for mocking dependencies
- memory_profiler for memory usage testing
- temporary file handling utilities

**Configuration**:
- Separate test configuration to avoid affecting production data
- Isolated temporary directories for each test
- Automatic cleanup of test artifacts

## Continuous Integration Integration

### Test Execution Strategy
- **Unit Tests**: Run on every commit, fast execution (<30 seconds)
- **Integration Tests**: Run on pull requests, moderate execution (<5 minutes)
- **Performance Tests**: Run nightly, comprehensive execution (<30 minutes)

### Success Criteria
- **Unit Test Coverage**: >95% line coverage for all data loading code
- **Integration Test Pass Rate**: 100% for all core workflows
- **Performance Benchmarks**: Memory usage <200MB, load time <30s
- **Error Message Quality**: All validation errors include actionable information

This comprehensive test strategy ensures the refactored data pipeline is robust, maintainable, and provides clear feedback when issues occur. The multi-layered approach validates both individual components and complete workflows, with special attention to the error conditions that could cause silent failures in the previous implementation.