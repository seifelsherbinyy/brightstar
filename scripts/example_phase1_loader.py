"""
Example usage of Phase 1 schema-driven loader.

This script demonstrates how to use the load_phase1 function to validate
and normalize raw data according to a YAML schema.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phase1 import load_phase1


def main():
    """Run Phase 1 loader example."""
    # Example 1: Load CSV file
    print("=" * 60)
    print("Phase 1 Loader Example")
    print("=" * 60)

    # Define paths
    input_file = "data/raw/sample_phase1.csv"  # Replace with your file
    schema_file = "config/schema_phase1.yaml"
    output_dir = "output/"

    print(f"\nConfiguration:")
    print(f"  Input file: {input_file}")
    print(f"  Schema: {schema_file}")
    print(f"  Output directory: {output_dir}")

    # Check if input file exists
    if not Path(input_file).exists():
        print(f"\n⚠️  Input file not found: {input_file}")
        print("Please create a sample CSV file with the following structure:")
        print()
        print("Vendor_ID,ASIN,Week,Category,Subcategory,GMS,Ordered_Units,...")
        print("V_001,B07XYZ1234,202501,Frozen Food,Ice Cream,1000.0,100,...")
        print()
        return

    try:
        # Load and validate data
        print("\n📊 Loading and validating data...")
        df = load_phase1(
            path=input_file,
            schema_path=schema_file,
            output_dir=output_dir,
            log_level="INFO",
        )

        print(f"\n✅ Success! Processed {len(df)} rows")
        print(f"\nOutput files:")
        print(f"  - {output_dir}phase1_validated.parquet")
        print(f"  - {output_dir}phase1_validation_report.json")
        print(f"  - {output_dir}phase1_invalid_rows.csv (if any)")

        # Display sample of validated data
        print(f"\nSample of validated data (first 5 rows):")
        print(df.head().to_string())

        # Display data quality summary
        if "is_valid_row" in df.columns:
            valid_count = df["is_valid_row"].sum()
            total_count = len(df)
            print(f"\nData Quality Summary:")
            print(f"  Valid rows: {valid_count}/{total_count} ({100*valid_count/total_count:.1f}%)")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
