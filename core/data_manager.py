"""
Data Manager - Handles data loading, storage, and querying using Polars and DuckDB
"""

import polars as pl
import duckdb
import io
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import pandas as pd


class DataManager:
    """
    Manages data loading, storage, and querying using Polars for processing
    and DuckDB for analytical queries
    """
    
    def __init__(self):
        self.tables: Dict[str, pl.DataFrame] = {}
        self.db = duckdb.connect(':memory:')  # In-memory DuckDB instance
        self.file_metadata: Dict[str, Dict] = {}
    
    def load_file(self, file) -> pl.DataFrame:
        """
        Load file into Polars DataFrame and auto-generate summaries
        
        Args:
            file: Streamlit UploadedFile object or file path
            
        Returns:
            Polars DataFrame
        """
        file_name = file.name if hasattr(file, 'name') else str(file)
        file_ext = Path(file_name).suffix.lower()
        
        try:
            if file_ext == '.csv':
                # First pass - detect problematic columns
                try:
                    df = pl.read_csv(
                        file,
                        infer_schema_length=10000,
                        ignore_errors=False
                    )
                except Exception as e:
                    # If parsing fails, read with all columns as strings first
                    df = pl.read_csv(
                        file,
                        infer_schema_length=0,  # Treat all as strings
                        ignore_errors=True
                    )
                    
                    # Try to convert columns to appropriate types
                    for col in df.columns:
                        try:
                            # Try to convert to numeric if possible
                            df = df.with_columns(
                                pl.col(col).cast(pl.Float64, strict=False).alias(col)
                            )
                        except:
                            # Keep as string if conversion fails
                            pass
            elif file_ext == '.parquet':
                df = pl.read_parquet(file)
            elif file_ext in ['.xlsx', '.xls']:
                # Use pandas to read Excel, then convert to Polars
                pandas_df = pd.read_excel(file)
                df = pl.from_pandas(pandas_df)
            elif file_ext == '.json':
                df = pl.read_json(file)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            # Detect and mask PII columns (GDPR compliance)
            df = self._mask_pii_columns(df)
            
            # Store in tables
            table_name = Path(file_name).stem
            self.tables[table_name] = df
            
            # Register with DuckDB for SQL queries
            self.db.register(table_name, df.to_pandas())
            
            # Auto-generate comprehensive summary using DuckDB
            summary = self._generate_auto_summary(table_name, df)
            
            # Store metadata with summary
            self.file_metadata[table_name] = {
                'filename': file_name,
                'format': file_ext,
                'shape': (df.height, df.width),
                'columns': df.columns,
                'dtypes': {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
                'auto_summary': summary,
                'upload_time': datetime.now().isoformat()
            }
            
            return df
            
        except Exception as e:
            raise Exception(f"Error loading file {file_name}: {str(e)}")
    
    def _mask_pii_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Detect and mask potential PII columns for GDPR compliance
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with PII columns masked
        """
        pii_keywords = ['email', 'phone', 'ssn', 'passport', 'license', 'credit_card', 
                       'address', 'zipcode', 'postal', 'ip_address', 'mac_address']
        
        masked_df = df.clone()
        
        for col in df.columns:
            col_lower = col.lower()
            # Check if column name contains PII keywords
            if any(keyword in col_lower for keyword in pii_keywords):
                # Mask the column
                if df[col].dtype == pl.Utf8:
                    masked_df = masked_df.with_columns(
                        pl.lit("***MASKED***").alias(col)
                    )
                print(f"⚠️ Masked PII column: {col}")
        
        return masked_df
    
    def _generate_auto_summary(self, table_name: str, df: pl.DataFrame) -> Dict[str, Any]:
        """
        Auto-generate comprehensive statistical summary using DuckDB
        This runs immediately on upload for efficient LLM context
        
        Args:
            table_name: Name of the table
            df: DataFrame to summarize
            
        Returns:
            Dictionary with comprehensive summary statistics
        """
        summary = {
            'row_count': df.height,
            'column_count': df.width,
            'memory_mb': df.estimated_size() / (1024 * 1024),
            'columns': {}
        }
        
        # Generate column-level summaries using DuckDB for efficiency
        for col in df.columns:
            dtype = df[col].dtype
            
            col_summary = {
                'dtype': str(dtype),
                'null_count': int(df[col].null_count()),
                'null_percentage': float(df[col].null_count() / df.height * 100) if df.height > 0 else 0
            }
            
            # Numeric columns - use DuckDB for fast aggregation
            if dtype in [pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.Float64, pl.Float32]:
                try:
                    stats_query = f"""
                    SELECT 
                        MIN("{col}") as min_val,
                        MAX("{col}") as max_val,
                        AVG("{col}") as mean_val,
                        MEDIAN("{col}") as median_val,
                        STDDEV("{col}") as std_val,
                        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY "{col}") as q25,
                        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY "{col}") as q75,
                        COUNT(DISTINCT "{col}") as unique_count
                    FROM {table_name}
                    """
                    stats_result = self.db.execute(stats_query).fetchone()
                    
                    col_summary.update({
                        'min': float(stats_result[0]) if stats_result[0] is not None else None,
                        'max': float(stats_result[1]) if stats_result[1] is not None else None,
                        'mean': float(stats_result[2]) if stats_result[2] is not None else None,
                        'median': float(stats_result[3]) if stats_result[3] is not None else None,
                        'std': float(stats_result[4]) if stats_result[4] is not None else None,
                        'q25': float(stats_result[5]) if stats_result[5] is not None else None,
                        'q75': float(stats_result[6]) if stats_result[6] is not None else None,
                        'unique': int(stats_result[7]) if stats_result[7] is not None else 0
                    })
                except Exception as e:
                    print(f"Error calculating stats for {col}: {e}")
            
            # String columns - get top values and unique count
            elif dtype == pl.Utf8:
                try:
                    unique_query = f"""
                    SELECT COUNT(DISTINCT "{col}") as unique_count
                    FROM {table_name}
                    """
                    unique_count = self.db.execute(unique_query).fetchone()[0]
                    
                    # Get top 5 values
                    top_values_query = f"""
                    SELECT "{col}", COUNT(*) as count
                    FROM {table_name}
                    WHERE "{col}" IS NOT NULL
                    GROUP BY "{col}"
                    ORDER BY count DESC
                    LIMIT 5
                    """
                    top_values = self.db.execute(top_values_query).fetchall()
                    
                    col_summary['unique'] = int(unique_count)
                    col_summary['top_values'] = [
                        {'value': str(val[0]), 'count': int(val[1])} 
                        for val in top_values
                    ]
                except Exception as e:
                    print(f"Error calculating stats for {col}: {e}")
            
            # Date/Datetime columns
            elif dtype in [pl.Date, pl.Datetime]:
                try:
                    date_query = f"""
                    SELECT 
                        MIN("{col}") as min_date,
                        MAX("{col}") as max_date
                    FROM {table_name}
                    """
                    date_result = self.db.execute(date_query).fetchone()
                    col_summary.update({
                        'min': str(date_result[0]) if date_result[0] else None,
                        'max': str(date_result[1]) if date_result[1] else None
                    })
                except Exception as e:
                    print(f"Error calculating stats for {col}: {e}")
            
            summary['columns'][col] = col_summary
        
        # Generate correlations for numeric columns using DuckDB
        numeric_cols = [col for col, dtype in zip(df.columns, df.dtypes)
                       if dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]]
        
        if len(numeric_cols) >= 2:
            try:
                summary['correlations'] = self._calculate_correlations_duckdb(table_name, numeric_cols)
            except Exception as e:
                print(f"Error calculating correlations: {e}")
        
        return summary
    
    def _calculate_correlations_duckdb(self, table_name: str, numeric_cols: List[str]) -> Dict[str, float]:
        """Calculate correlations between numeric columns using DuckDB"""
        correlations = {}
        
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                try:
                    corr_query = f"""
                    SELECT CORR("{col1}", "{col2}") as correlation
                    FROM {table_name}
                    WHERE "{col1}" IS NOT NULL AND "{col2}" IS NOT NULL
                    """
                    corr = self.db.execute(corr_query).fetchone()[0]
                    if corr is not None:
                        correlations[f"{col1}_vs_{col2}"] = float(corr)
                except:
                    pass
        
        return correlations
    
    def get_table(self, table_name: str) -> Optional[pl.DataFrame]:
        """Get table by name"""
        return self.tables.get(table_name)
    
    def list_tables(self) -> List[str]:
        """List all available tables"""
        return list(self.tables.keys())
    
    def execute_sql(self, query: str) -> pl.DataFrame:
        """
        Execute SQL query using DuckDB and return Polars DataFrame
        
        Args:
            query: SQL query string
            
        Returns:
            Query result as Polars DataFrame
        """
        try:
            result = self.db.execute(query).fetchdf()
            return pl.from_pandas(result)
        except Exception as e:
            raise Exception(f"Error executing SQL query: {str(e)}\nQuery: {query}")
    
    def get_data_profile(self, table_name: str) -> Dict[str, Any]:
        """
        Get comprehensive data profile for a table
        
        Args:
            table_name: Name of the table to profile
            
        Returns:
            Dictionary containing data profile statistics
        """
        df = self.tables.get(table_name)
        if df is None:
            raise ValueError(f"Table '{table_name}' not found")
        
        profile = {
            'table_name': table_name,
            'shape': (df.height, df.width),
            'memory_usage': df.estimated_size(),
            'columns': {}
        }
        
        for col in df.columns:
            col_profile = self._profile_column(df, col)
            profile['columns'][col] = col_profile
        
        return profile
    
    def _profile_column(self, df: pl.DataFrame, col: str) -> Dict[str, Any]:
        """Profile a single column"""
        col_data = df[col]
        dtype = col_data.dtype
        
        base_stats = {
            'dtype': str(dtype),
            'null_count': col_data.null_count(),
            'null_percentage': (col_data.null_count() / df.height * 100) if df.height > 0 else 0,
            'unique': col_data.n_unique()
        }
        
        # Numeric columns
        if dtype in [pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.Float64, pl.Float32]:
            try:
                base_stats.update({
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'mean': float(col_data.mean()),
                    'median': float(col_data.median()),
                    'std': float(col_data.std()),
                    'q25': float(col_data.quantile(0.25)),
                    'q75': float(col_data.quantile(0.75))
                })
            except:
                pass
        
        # String columns
        elif dtype == pl.Utf8:
            try:
                # Get top values
                top_values = (
                    col_data
                    .value_counts()
                    .sort('counts', descending=True)
                    .head(5)
                )
                base_stats['top_values'] = top_values.to_dicts()
            except:
                pass
        
        # Date/Datetime columns
        elif dtype in [pl.Date, pl.Datetime]:
            try:
                base_stats.update({
                    'min': str(col_data.min()),
                    'max': str(col_data.max())
                })
            except:
                pass
        
        return base_stats
    
    def get_sample_data(self, table_name: str, n: int = 5) -> pl.DataFrame:
        """Get sample rows from table"""
        df = self.tables.get(table_name)
        if df is None:
            raise ValueError(f"Table '{table_name}' not found")
        return df.head(n)
    
    def get_column_stats(self, table_name: str, column: str) -> Dict[str, Any]:
        """Get detailed statistics for a specific column"""
        df = self.tables.get(table_name)
        if df is None:
            raise ValueError(f"Table '{table_name}' not found")
        
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in table '{table_name}'")
        
        return self._profile_column(df, column)
    
    def filter_data(
        self,
        table_name: str,
        conditions: List[Dict[str, Any]]
    ) -> pl.DataFrame:
        """
        Filter data based on conditions
        
        Args:
            table_name: Name of the table
            conditions: List of condition dicts with 'column', 'operator', 'value'
            
        Returns:
            Filtered DataFrame
        """
        df = self.tables.get(table_name)
        if df is None:
            raise ValueError(f"Table '{table_name}' not found")
        
        for condition in conditions:
            col = condition['column']
            op = condition['operator']
            val = condition['value']
            
            if op == 'eq':
                df = df.filter(pl.col(col) == val)
            elif op == 'ne':
                df = df.filter(pl.col(col) != val)
            elif op == 'gt':
                df = df.filter(pl.col(col) > val)
            elif op == 'lt':
                df = df.filter(pl.col(col) < val)
            elif op == 'gte':
                df = df.filter(pl.col(col) >= val)
            elif op == 'lte':
                df = df.filter(pl.col(col) <= val)
            elif op == 'contains':
                df = df.filter(pl.col(col).str.contains(val))
            elif op == 'isin':
                df = df.filter(pl.col(col).is_in(val))
        
        return df
    
    def aggregate_data(
        self,
        table_name: str,
        group_by: List[str],
        aggregations: Dict[str, List[str]]
    ) -> pl.DataFrame:
        """
        Aggregate data
        
        Args:
            table_name: Name of the table
            group_by: Columns to group by
            aggregations: Dict mapping columns to aggregation functions
            
        Returns:
            Aggregated DataFrame
        """
        df = self.tables.get(table_name)
        if df is None:
            raise ValueError(f"Table '{table_name}' not found")
        
        agg_exprs = []
        
        for col, funcs in aggregations.items():
            for func in funcs:
                if func == 'sum':
                    agg_exprs.append(pl.col(col).sum().alias(f"{col}_sum"))
                elif func == 'mean':
                    agg_exprs.append(pl.col(col).mean().alias(f"{col}_mean"))
                elif func == 'median':
                    agg_exprs.append(pl.col(col).median().alias(f"{col}_median"))
                elif func == 'min':
                    agg_exprs.append(pl.col(col).min().alias(f"{col}_min"))
                elif func == 'max':
                    agg_exprs.append(pl.col(col).max().alias(f"{col}_max"))
                elif func == 'count':
                    agg_exprs.append(pl.col(col).count().alias(f"{col}_count"))
                elif func == 'std':
                    agg_exprs.append(pl.col(col).std().alias(f"{col}_std"))
        
        if group_by:
            return df.group_by(group_by).agg(agg_exprs)
        else:
            return df.select(agg_exprs)
    
    def join_tables(
        self,
        left_table: str,
        right_table: str,
        on: Union[str, List[str]],
        how: str = 'inner'
    ) -> pl.DataFrame:
        """
        Join two tables
        
        Args:
            left_table: Name of left table
            right_table: Name of right table
            on: Column(s) to join on
            how: Join type ('inner', 'left', 'right', 'outer')
            
        Returns:
            Joined DataFrame
        """
        left_df = self.tables.get(left_table)
        right_df = self.tables.get(right_table)
        
        if left_df is None:
            raise ValueError(f"Table '{left_table}' not found")
        if right_df is None:
            raise ValueError(f"Table '{right_table}' not found")
        
        return left_df.join(right_df, on=on, how=how)
    
    def export_to_parquet(self, table_name: str, path: str):
        """Export table to Parquet format"""
        df = self.tables.get(table_name)
        if df is None:
            raise ValueError(f"Table '{table_name}' not found")
        
        df.write_parquet(path)
    
    def export_to_csv(self, table_name: str) -> str:
        """Export table to CSV string"""
        df = self.tables.get(table_name)
        if df is None:
            raise ValueError(f"Table '{table_name}' not found")
        
        return df.write_csv()
    
    def get_correlation_matrix(self, table_name: str) -> pl.DataFrame:
        """Calculate correlation matrix for numeric columns"""
        df = self.tables.get(table_name)
        if df is None:
            raise ValueError(f"Table '{table_name}' not found")
        
        # Get numeric columns only
        numeric_cols = [col for col, dtype in zip(df.columns, df.dtypes)
                       if dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]]
        
        if not numeric_cols:
            raise ValueError(f"No numeric columns found in table '{table_name}'")
        
        # Use pandas for correlation calculation (easier)
        pandas_df = df.select(numeric_cols).to_pandas()
        corr_matrix = pandas_df.corr()
        
        return pl.from_pandas(corr_matrix.reset_index())
