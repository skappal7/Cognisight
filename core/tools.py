"""
Tool Registry - Defines all available tools for the data analyst agent
"""

from typing import Dict, List, Any, Callable
import polars as pl
from core.data_manager import DataManager
from utils.visualizer import Visualizer


class ToolRegistry:
    """
    Registry of all tools available to the agent
    Each tool is a function that can be called by the agent
    """
    
    def __init__(self, data_manager: DataManager, visualizer: Visualizer):
        self.data_manager = data_manager
        self.visualizer = visualizer
        self.tools = self._register_tools()
    
    def _register_tools(self) -> Dict[str, Dict[str, Any]]:
        """Register all available tools"""
        return {
            'get_data_summary': {
                'function': self.get_data_summary,
                'description': 'Get comprehensive summary statistics for a table',
                'parameters': {
                    'table_name': 'Name of the table to summarize'
                }
            },
            'execute_sql': {
                'function': self.execute_sql,
                'description': 'Execute SQL query on the data using DuckDB',
                'parameters': {
                    'query': 'SQL query string to execute'
                }
            },
            'filter_data': {
                'function': self.filter_data,
                'description': 'Filter data based on conditions',
                'parameters': {
                    'table_name': 'Name of the table',
                    'conditions': 'List of filter conditions (column, operator, value)'
                }
            },
            'aggregate_data': {
                'function': self.aggregate_data,
                'description': 'Aggregate data with grouping',
                'parameters': {
                    'table_name': 'Name of the table',
                    'group_by': 'List of columns to group by',
                    'aggregations': 'Dict of column: [functions] to aggregate'
                }
            },
            'get_column_stats': {
                'function': self.get_column_stats,
                'description': 'Get detailed statistics for a specific column',
                'parameters': {
                    'table_name': 'Name of the table',
                    'column': 'Column name to analyze'
                }
            },
            'get_correlation': {
                'function': self.get_correlation,
                'description': 'Calculate correlation matrix for numeric columns',
                'parameters': {
                    'table_name': 'Name of the table'
                }
            },
            'create_line_chart': {
                'function': self.create_line_chart,
                'description': 'Create a line chart visualization',
                'parameters': {
                    'table_name': 'Name of the table',
                    'x_column': 'Column for x-axis',
                    'y_columns': 'Column(s) for y-axis (can be list)',
                    'title': 'Chart title'
                }
            },
            'create_bar_chart': {
                'function': self.create_bar_chart,
                'description': 'Create a bar chart visualization',
                'parameters': {
                    'table_name': 'Name of the table',
                    'x_column': 'Column for x-axis (categories)',
                    'y_column': 'Column for y-axis (values)',
                    'title': 'Chart title'
                }
            },
            'create_scatter_plot': {
                'function': self.create_scatter_plot,
                'description': 'Create a scatter plot visualization',
                'parameters': {
                    'table_name': 'Name of the table',
                    'x_column': 'Column for x-axis',
                    'y_column': 'Column for y-axis',
                    'color_column': 'Optional column for color coding',
                    'title': 'Chart title'
                }
            },
            'create_histogram': {
                'function': self.create_histogram,
                'description': 'Create a histogram visualization',
                'parameters': {
                    'table_name': 'Name of the table',
                    'column': 'Column to plot',
                    'bins': 'Number of bins (optional)',
                    'title': 'Chart title'
                }
            },
            'create_box_plot': {
                'function': self.create_box_plot,
                'description': 'Create a box plot visualization',
                'parameters': {
                    'table_name': 'Name of the table',
                    'column': 'Numeric column to plot',
                    'group_by': 'Optional column to group by',
                    'title': 'Chart title'
                }
            },
            'create_heatmap': {
                'function': self.create_heatmap,
                'description': 'Create a correlation heatmap',
                'parameters': {
                    'table_name': 'Name of the table',
                    'title': 'Chart title'
                }
            },
            'create_pie_chart': {
                'function': self.create_pie_chart,
                'description': 'Create a pie chart visualization',
                'parameters': {
                    'table_name': 'Name of the table',
                    'column': 'Column for categories',
                    'value_column': 'Optional column for values',
                    'title': 'Chart title'
                }
            },
            'detect_outliers': {
                'function': self.detect_outliers,
                'description': 'Detect outliers in numeric columns using IQR method',
                'parameters': {
                    'table_name': 'Name of the table',
                    'column': 'Numeric column to check for outliers'
                }
            },
            'time_series_decomposition': {
                'function': self.time_series_decomposition,
                'description': 'Decompose time series into trend, seasonality, and residuals',
                'parameters': {
                    'table_name': 'Name of the table',
                    'date_column': 'Date/time column',
                    'value_column': 'Value column to decompose'
                }
            }
        }
    
    def get_tool_descriptions(self) -> str:
        """Get formatted descriptions of all tools"""
        descriptions = []
        for tool_name, tool_info in self.tools.items():
            params = ', '.join([f"{k}: {v}" for k, v in tool_info['parameters'].items()])
            descriptions.append(
                f"**{tool_name}**\n"
                f"  Description: {tool_info['description']}\n"
                f"  Parameters: {params}"
            )
        return '\n\n'.join(descriptions)
    
    def execute(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool with given parameters"""
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        tool = self.tools[tool_name]
        try:
            result = tool['function'](**parameters)
            return result
        except Exception as e:
            return {
                'error': True,
                'message': f"Error executing {tool_name}: {str(e)}"
            }
    
    # Tool implementations
    
    def get_data_summary(self, table_name: str) -> Dict[str, Any]:
        """Get comprehensive data summary"""
        profile = self.data_manager.get_data_profile(table_name)
        
        # Format for readable output
        summary = {
            'type': 'data',
            'description': f"Summary of {table_name}",
            'result': profile
        }
        
        return summary
    
    def execute_sql(self, query: str) -> Dict[str, Any]:
        """Execute SQL query"""
        result_df = self.data_manager.execute_sql(query)
        
        return {
            'type': 'data',
            'description': 'SQL query result',
            'dataframe': result_df,
            'result': f"Query returned {result_df.height} rows and {result_df.width} columns"
        }
    
    def filter_data(self, table_name: str, conditions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Filter data"""
        filtered_df = self.data_manager.filter_data(table_name, conditions)
        
        return {
            'type': 'data',
            'description': f"Filtered data from {table_name}",
            'dataframe': filtered_df,
            'result': f"Filter returned {filtered_df.height} rows"
        }
    
    def aggregate_data(
        self,
        table_name: str,
        group_by: List[str],
        aggregations: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Aggregate data"""
        agg_df = self.data_manager.aggregate_data(table_name, group_by, aggregations)
        
        return {
            'type': 'data',
            'description': f"Aggregated data from {table_name}",
            'dataframe': agg_df,
            'result': f"Aggregation returned {agg_df.height} groups"
        }
    
    def get_column_stats(self, table_name: str, column: str) -> Dict[str, Any]:
        """Get column statistics"""
        stats = self.data_manager.get_column_stats(table_name, column)
        
        return {
            'type': 'data',
            'description': f"Statistics for {column} in {table_name}",
            'result': stats
        }
    
    def get_correlation(self, table_name: str) -> Dict[str, Any]:
        """Get correlation matrix"""
        corr_df = self.data_manager.get_correlation_matrix(table_name)
        
        return {
            'type': 'data',
            'description': f"Correlation matrix for {table_name}",
            'dataframe': corr_df,
            'result': "Correlation matrix calculated"
        }
    
    def create_line_chart(
        self,
        table_name: str,
        x_column: str,
        y_columns: Any,
        title: str
    ) -> Dict[str, Any]:
        """Create line chart"""
        df = self.data_manager.get_table(table_name)
        
        # Ensure y_columns is a list
        if not isinstance(y_columns, list):
            y_columns = [y_columns]
        
        fig = self.visualizer.create_line_chart(
            df=df,
            x_column=x_column,
            y_columns=y_columns,
            title=title
        )
        
        return {
            'type': 'visualization',
            'title': title,
            'figure': fig,
            'chart_type': 'line'
        }
    
    def create_bar_chart(
        self,
        table_name: str,
        x_column: str,
        y_column: str,
        title: str
    ) -> Dict[str, Any]:
        """Create bar chart"""
        df = self.data_manager.get_table(table_name)
        
        fig = self.visualizer.create_bar_chart(
            df=df,
            x_column=x_column,
            y_column=y_column,
            title=title
        )
        
        return {
            'type': 'visualization',
            'title': title,
            'figure': fig,
            'chart_type': 'bar'
        }
    
    def create_scatter_plot(
        self,
        table_name: str,
        x_column: str,
        y_column: str,
        color_column: str = None,
        title: str = "Scatter Plot"
    ) -> Dict[str, Any]:
        """Create scatter plot"""
        df = self.data_manager.get_table(table_name)
        
        fig = self.visualizer.create_scatter_plot(
            df=df,
            x_column=x_column,
            y_column=y_column,
            color_column=color_column,
            title=title
        )
        
        return {
            'type': 'visualization',
            'title': title,
            'figure': fig,
            'chart_type': 'scatter'
        }
    
    def create_histogram(
        self,
        table_name: str,
        column: str,
        bins: int = 30,
        title: str = "Histogram"
    ) -> Dict[str, Any]:
        """Create histogram"""
        df = self.data_manager.get_table(table_name)
        
        fig = self.visualizer.create_histogram(
            df=df,
            column=column,
            bins=bins,
            title=title
        )
        
        return {
            'type': 'visualization',
            'title': title,
            'figure': fig,
            'chart_type': 'histogram'
        }
    
    def create_box_plot(
        self,
        table_name: str,
        column: str,
        group_by: str = None,
        title: str = "Box Plot"
    ) -> Dict[str, Any]:
        """Create box plot"""
        df = self.data_manager.get_table(table_name)
        
        fig = self.visualizer.create_box_plot(
            df=df,
            column=column,
            group_by=group_by,
            title=title
        )
        
        return {
            'type': 'visualization',
            'title': title,
            'figure': fig,
            'chart_type': 'box'
        }
    
    def create_heatmap(self, table_name: str, title: str = "Correlation Heatmap") -> Dict[str, Any]:
        """Create correlation heatmap"""
        corr_df = self.data_manager.get_correlation_matrix(table_name)
        
        fig = self.visualizer.create_heatmap(
            corr_matrix=corr_df,
            title=title
        )
        
        return {
            'type': 'visualization',
            'title': title,
            'figure': fig,
            'chart_type': 'heatmap'
        }
    
    def create_pie_chart(
        self,
        table_name: str,
        column: str,
        value_column: str = None,
        title: str = "Pie Chart"
    ) -> Dict[str, Any]:
        """Create pie chart"""
        df = self.data_manager.get_table(table_name)
        
        fig = self.visualizer.create_pie_chart(
            df=df,
            column=column,
            value_column=value_column,
            title=title
        )
        
        return {
            'type': 'visualization',
            'title': title,
            'figure': fig,
            'chart_type': 'pie'
        }
    
    def detect_outliers(self, table_name: str, column: str) -> Dict[str, Any]:
        """Detect outliers using IQR method"""
        df = self.data_manager.get_table(table_name)
        
        col_data = df[column]
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = df.filter(
            (pl.col(column) < lower_bound) | (pl.col(column) > upper_bound)
        )
        
        return {
            'type': 'data',
            'description': f"Outliers in {column}",
            'dataframe': outliers,
            'result': {
                'outlier_count': outliers.height,
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'percentage': (outliers.height / df.height * 100) if df.height > 0 else 0
            }
        }
    
    def time_series_decomposition(
        self,
        table_name: str,
        date_column: str,
        value_column: str
    ) -> Dict[str, Any]:
        """Time series decomposition (basic implementation)"""
        df = self.data_manager.get_table(table_name)
        
        # Sort by date
        df = df.sort(date_column)
        
        # Calculate rolling average for trend
        window_size = min(30, df.height // 4)
        trend = df.select([
            pl.col(date_column),
            pl.col(value_column).rolling_mean(window_size).alias('trend')
        ])
        
        return {
            'type': 'data',
            'description': f"Time series decomposition of {value_column}",
            'dataframe': trend,
            'result': "Trend component calculated"
        }
