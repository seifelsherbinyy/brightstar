"""Unit tests for ingestion utilities and value coercion."""
import sys
from pathlib import Path
from io import BytesIO

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.phase1_ingestion import _coerce_value_series
from src.ingestion_utils import detect_identifier_column, load_master_lookup


def test_coerce_value_series():
    """Test value coercion handles currency, percentages, and commas."""
    # Create a series with various formats
    s = pd.Series(['100', '1,234', '$2,500.50', '43.92%', 'invalid', ''])
    
    result = _coerce_value_series(s)
    
    # Check specific values
    assert result.iloc[0] == 100.0
    assert result.iloc[1] == 1234.0
    assert result.iloc[2] == 2500.5
    assert abs(result.iloc[3] - 0.4392) < 0.0001  # 43.92% -> 0.4392
    assert pd.isna(result.iloc[4])  # 'invalid' -> NaN
    assert pd.isna(result.iloc[5])  # '' -> NaN


def test_coerce_value_series_with_negatives():
    """Test value coercion handles negative values."""
    s = pd.Series(['-100', '-$1,234.56', '-5.5%'])
    
    result = _coerce_value_series(s)
    
    assert result.iloc[0] == -100.0
    assert result.iloc[1] == -1234.56
    assert abs(result.iloc[2] - (-0.055)) < 0.0001


def test_detect_identifier_column_finds_asin():
    """Test that detect_identifier_column properly finds ASIN column."""
    # Create a DataFrame with various header styles
    df = pd.DataFrame({
        'Vendor Code': ['V1', 'V2', 'V3'],
        'ASIN': ['A100', 'A200', 'A300'],
        'GMS($)': [1000, 2000, 3000]
    })
    
    asin_aliases = ['asin', 'ASIN', 'product_id']
    result_df, asin_col = detect_identifier_column(df, asin_aliases)
    
    assert asin_col == 'ASIN'
    assert list(result_df[asin_col]) == ['A100', 'A200', 'A300']
    # Ensure we didn't duplicate a single value
    assert len(result_df[asin_col].unique()) == 3


def test_detect_identifier_column_case_insensitive():
    """Test case-insensitive matching for identifier columns."""
    df = pd.DataFrame({
        'vendor code': ['V1', 'V2'],
        'asin': ['A1', 'A2'],
        'value': [100, 200]
    })
    
    vendor_aliases = ['Vendor', 'Vendor Code', 'vendor_code']
    result_df, vendor_col = detect_identifier_column(df, vendor_aliases)
    
    assert vendor_col == 'vendor code'
    assert list(result_df[vendor_col]) == ['V1', 'V2']


def test_detect_identifier_column_with_spaces():
    """Test that columns with single space are properly detected via normalization."""
    df = pd.DataFrame({
        'Vendor Code': ['V1', 'V2'],  # Single space
        'Product ID': ['A1', 'A2'],
        'Sales': [100, 200]
    })
    
    vendor_aliases = ['vendor code', 'Vendor Code', 'vendor_code']
    result_df, vendor_col = detect_identifier_column(df, vendor_aliases)
    
    # Should find the match through normalization
    assert vendor_col == 'Vendor Code'
    # Ensure values are preserved
    assert len(result_df[vendor_col].unique()) == 2


def test_load_master_lookup_csv(tmp_path):
    """Test loading master lookup from CSV with case-insensitive columns."""
    csv_path = tmp_path / "master.csv"
    csv_content = """ASIN,Vendor Code,Vendor Name
A1,V100,Vendor One
A2,V200,Vendor Two
A1,V101,Vendor One Updated"""
    csv_path.write_text(csv_content)
    
    result = load_master_lookup(csv_path)
    
    # Check columns are normalized
    assert set(result.columns) == {'asin', 'vendor_code', 'vendor_name'}
    
    # Check deduplication (last entry for A1 should be kept)
    assert len(result) == 2
    a1_row = result[result['asin'] == 'A1'].iloc[0]
    # Use string comparison to avoid NA ambiguity
    assert str(a1_row['vendor_code']) == 'V101'
    assert str(a1_row['vendor_name']) == 'Vendor One Updated'


def test_load_master_lookup_xlsx(tmp_path):
    """Test loading master lookup from Excel with mixed case columns."""
    xlsx_path = tmp_path / "master.xlsx"
    
    # Create DataFrame with mixed case columns
    df = pd.DataFrame({
        'asin': ['A1', 'A2', 'A3'],
        'Vendor Code': ['V1', 'V2', 'V3'],
        'VENDOR NAME': ['Vendor One', 'Vendor Two', 'Vendor Three']
    })
    df.to_excel(xlsx_path, index=False, sheet_name='Info')
    
    result = load_master_lookup(xlsx_path)
    
    # Check columns are normalized
    assert set(result.columns) == {'asin', 'vendor_code', 'vendor_name'}
    
    # Check all rows are present
    assert len(result) == 3
    assert list(result['asin']) == ['A1', 'A2', 'A3']


def test_load_master_lookup_xlsx_multi_sheet(tmp_path):
    """Test loading master lookup from Excel with multiple sheets."""
    xlsx_path = tmp_path / "master.xlsx"
    
    # Create Excel with multiple sheets
    with pd.ExcelWriter(xlsx_path) as writer:
        df1 = pd.DataFrame({
            'ASIN': ['A1'],
            'Vendor Code': ['V1'],
            'Vendor Name': ['Wrong Sheet']
        })
        df2 = pd.DataFrame({
            'ASIN': ['A2', 'A3'],
            'Vendor Code': ['V2', 'V3'],
            'Vendor Name': ['Vendor Two', 'Vendor Three']
        })
        df1.to_excel(writer, sheet_name='Summary', index=False)
        df2.to_excel(writer, sheet_name='Info', index=False)
    
    # Should prefer 'Info' sheet
    result = load_master_lookup(xlsx_path)
    
    assert len(result) == 2
    assert list(result['asin']) == ['A2', 'A3']


def test_load_master_lookup_drops_missing_asin(tmp_path):
    """Test that rows with missing ASIN are dropped."""
    csv_path = tmp_path / "master.csv"
    csv_content = """ASIN,Vendor Code,Vendor Name
A1,V1,Vendor One
,V2,Vendor Two
A3,V3,Vendor Three"""
    csv_path.write_text(csv_content)
    
    result = load_master_lookup(csv_path)
    
    # Only 2 rows should remain (missing ASIN dropped)
    assert len(result) == 2
    assert 'A1' in result['asin'].values
    assert 'A3' in result['asin'].values


def test_parquet_write_roundtrip(tmp_path):
    """Test that coerced values can be written to and read from Parquet."""
    # Create a DataFrame with percent values
    df = pd.DataFrame({
        'asin': ['A1', 'A2', 'A3'],
        'vendor_code': ['V1', 'V1', 'V2'],
        'metric': ['Fill Rate (%)', 'Fill Rate (%)', 'Fill Rate (%)'],
        'week_label': ['2025W01', '2025W02', '2025W01'],
        'week_raw': ['W1', 'W2', 'W1'],
        'week_start': ['2025-01-01', '2025-01-08', '2025-01-01'],
        'week_end': ['2025-01-07', '2025-01-14', '2025-01-07'],
        'value': ['43.92%', '85.5%', '99%']
    })
    
    # Apply coercion
    df['value'] = _coerce_value_series(df['value'])
    df['asin'] = df['asin'].astype('string')
    df['vendor_code'] = df['vendor_code'].astype('string')
    
    # Write to Parquet
    parquet_path = tmp_path / "test.parquet"
    df.to_parquet(parquet_path, index=False)
    
    # Read back
    result = pd.read_parquet(parquet_path)
    
    # Verify values are floats and correctly converted
    assert result['value'].dtype == 'float64'
    assert abs(result['value'].iloc[0] - 0.4392) < 0.0001
    assert abs(result['value'].iloc[1] - 0.855) < 0.0001
    assert abs(result['value'].iloc[2] - 0.99) < 0.0001
    
    # Verify string types are preserved
    assert result['asin'].dtype == 'string'
    assert result['vendor_code'].dtype == 'string'


def test_parquet_write_with_currency(tmp_path):
    """Test that currency values can be written to Parquet."""
    df = pd.DataFrame({
        'asin': ['A1', 'A2'],
        'vendor_code': ['V1', 'V2'],
        'metric': ['GMS ($)', 'GMS ($)'],
        'week_label': ['2025W01', '2025W01'],
        'week_raw': ['W1', 'W1'],
        'week_start': ['2025-01-01', '2025-01-01'],
        'week_end': ['2025-01-07', '2025-01-07'],
        'value': ['$1,234.56', '$2,500.00']
    })
    
    # Apply coercion
    df['value'] = _coerce_value_series(df['value'])
    df['asin'] = df['asin'].astype('string')
    df['vendor_code'] = df['vendor_code'].astype('string')
    
    # Write to Parquet
    parquet_path = tmp_path / "currency.parquet"
    df.to_parquet(parquet_path, index=False)
    
    # Read back
    result = pd.read_parquet(parquet_path)
    
    # Verify values
    assert result['value'].iloc[0] == 1234.56
    assert result['value'].iloc[1] == 2500.00
