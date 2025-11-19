# Phase 1: Schema-Driven Data Loader

Production-ready, schema-enforced Phase 1 loader for BrightStar v2.0 pipeline.

## Overview

The Phase 1 loader provides a canonical, auditable data ingestion system that:
- ✅ Validates data against YAML schema contracts
- ✅ Normalizes IDs (ASIN, vendor_id, week, category) to canonical formats
- ✅ Performs comprehensive data quality checks
- ✅ Generates audit-ready validation reports
- ✅ Outputs clean Parquet files ready for downstream phases

## Quick Start

```python
from phase1 import load_phase1

# Load and validate data
df = load_phase1(
    path="data/raw/export.csv",
    schema_path="config/schema_phase1.yaml",
    output_dir="output/"
)
```

## Architecture

```
src/phase1/
├── __init__.py              # Public API
├── loader.py                # Main orchestration and file loading
├── schema_validator.py      # YAML schema validation
├── id_normalizer.py         # ID canonicalization
├── dq_checks.py             # Data quality checks
└── validation_report.py     # Validation report generation
```

## Modules

### loader.py
Main entry point providing `load_phase1()` function that:
1. Auto-detects file format (CSV/Parquet)
2. Normalizes column names
3. Validates against schema
4. Casts to schema types
5. Normalizes IDs
6. Performs DQ checks
7. Generates reports
8. Outputs validated Parquet

### schema_validator.py
Schema validation using YAML configuration:
- Required column checks
- Data type validation
- Categorical value validation
- Custom validation rules

### id_normalizer.py
ID canonicalization functions:
- `normalize_asin()` - Uppercase alphanumeric, 10 chars
- `normalize_vendor_id()` - Uppercase, no special chars except underscores
- `normalize_week()` - YYYYWW integer format (validated range)
- `normalize_category()` - Title case normalization

### dq_checks.py
Data quality framework:
- Missing required columns detection
- Invalid ID format detection
- Invalid numeric value detection
- Out-of-range value detection
- Null percentage calculation
- Numeric outlier detection

### validation_report.py
Audit trail generation:
- JSON validation reports with metrics
- CSV exports of invalid rows
- Summary statistics
- Categorical summaries

## Schema Configuration

Define canonical schema in `config/schema_phase1.yaml`:

```yaml
required_columns:
  - vendor_id
  - asin
  - week
  - gms
  # ... more columns

dtypes:
  vendor_id: string
  asin: string
  week: int
  gms: float

categorical_fields:
  category:
    - Frozen Food
    - Baking Supplies
    # ... more values

validation:
  non_negative_fields:
    - gms
    - ordered_units
  
  week_format:
    pattern: "^\\d{6}$"
    min_value: 202001
    max_value: 209952
```

## Usage Examples

### Basic Usage

```python
from phase1 import load_phase1

df = load_phase1(
    path="data/raw/export.csv",
    schema_path="config/schema_phase1.yaml",
    output_dir="output/",
    log_level="INFO"
)
```

### Programmatic Validation

```python
from phase1.schema_validator import SchemaValidator

validator = SchemaValidator("config/schema_phase1.yaml")
validation_result = validator.validate_schema(df)

if validation_result["valid"]:
    print("✅ Schema validation passed")
else:
    print("❌ Validation errors:", validation_result)
```

### ID Normalization

```python
from phase1.id_normalizer import (
    normalize_asin,
    normalize_vendor_id,
    normalize_week
)

asin = normalize_asin("b07-xyz-1234")  # Returns: "B07XYZ1234"
vendor = normalize_vendor_id("v_001")  # Returns: "V_001"
week = normalize_week(202501)          # Returns: 202501
```

### Custom DQ Checks

```python
from phase1.dq_checks import (
    add_data_quality_flags,
    get_invalid_rows
)

df_with_flags = add_data_quality_flags(
    df,
    required_columns=["vendor_id", "asin"],
    numeric_columns=["gms", "asp"],
    non_negative_fields=["gms"]
)

invalid_rows = get_invalid_rows(df_with_flags)
print(f"Found {len(invalid_rows)} invalid rows")
```

## Output Files

The loader generates the following outputs:

1. **phase1_validated.parquet** - Clean, validated data ready for Phase 2
2. **phase1_validation_report.json** - Comprehensive validation metrics
3. **phase1_invalid_rows.csv** - Rows that failed validation (for debugging)

### Validation Report Structure

```json
{
  "timestamp": "2025-11-18T23:45:00",
  "row_count": 1000,
  "valid_row_count": 950,
  "invalid_row_count": 50,
  "invalid_id_rows": 10,
  "null_percentage_by_column": {
    "vendor_id": 0.0,
    "gms": 2.5
  },
  "numeric_outlier_counts": {
    "gms": 15
  },
  "data_quality_summary": {
    "dq_missing_required_columns": 5,
    "dq_invalid_ids": 10,
    "dq_invalid_numeric": 20,
    "dq_out_of_range": 15
  }
}
```

## Data Quality Flags

The loader adds the following boolean columns to the DataFrame:

- `dq_missing_required_columns` - Missing values in required fields
- `dq_invalid_ids` - Invalid ID formats (ASIN, vendor_id, week)
- `dq_invalid_numeric` - Invalid numeric values (NaN, inf)
- `dq_out_of_range` - Out-of-range values (e.g., negative quantities)
- `is_valid_row` - Aggregate flag (False if any DQ issue)

## Testing

Run the comprehensive test suite:

```bash
pytest tests/unit/test_phase1_loader.py -v
```

Test coverage includes:
- ID normalization (11 tests)
- Schema validation (4 tests)
- Data quality checks (7 tests)
- File loading and type casting (4 tests)
- End-to-end integration (2 tests)

## Error Handling

The loader raises structured exceptions:

```python
from phase1.loader import IDFormatError
from phase1.schema_validator import SchemaMismatchError

try:
    df = load_phase1(path="data.csv")
except FileNotFoundError:
    print("Input file not found")
except SchemaMismatchError as e:
    print(f"Schema validation failed: {e}")
except IDFormatError as e:
    print(f"ID normalization failed: {e}")
except ValueError as e:
    print(f"Data validation failed: {e}")
```

## Best Practices

1. **Always use the schema YAML** - Don't hardcode column expectations
2. **Review validation reports** - Check for data quality issues
3. **Handle invalid rows** - Decide whether to fix or reject
4. **Use logging** - Set `log_level="DEBUG"` for detailed diagnostics
5. **Version your schema** - Track schema changes with your data

## Performance Considerations

- CSV loading is slower than Parquet (use Parquet when possible)
- ID normalization is applied row-by-row (consider vectorization for huge datasets)
- DQ checks use pandas operations (efficient for most datasets)
- Outlier detection uses z-score method (configurable threshold)

## Integration with BrightStar Pipeline

The Phase 1 loader integrates seamlessly with downstream phases:

```python
# Phase 1: Load and validate
from phase1 import load_phase1
df_validated = load_phase1(path="raw/export.csv")

# Phase 2: Score using validated data
from brightstar.phase2_scoring import score_vendors
scores = score_vendors(df_validated)

# Phase 3+: Continue pipeline...
```

## Requirements

- Python 3.10+
- pandas >= 1.5.0
- pyarrow >= 12.0
- pyyaml >= 6.0
- pydantic >= 2.0 (optional, for enhanced validation)

## License

Part of the BrightStar analytics pipeline.
