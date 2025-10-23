"""
Visualizer - Creates professional, publication-ready visualizations using Plotly
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl
from typing import List, Optional, Union


class Visualizer:
    """
    Creates various types of visualizations using Plotly
    All charts are interactive and professionally styled
    """
    
    def __init__(self, template: str = "plotly_white"):
        self.template = template
        self.color_palette = px.colors.qualitative.Set2
    
    def create_line_chart(
        self,
        df: pl.DataFrame,
        x_column: str,
        y_columns: Union[str, List[str]],
        title: str = "Line Chart",
        x_label: Optional[str] = None,
        y_label: Optional[str] = None
    ) -> go.Figure:
        """Create an interactive line chart"""
        
        if isinstance(y_columns, str):
            y_columns = [y_columns]
        
        pandas_df = df.to_pandas()
        
        fig = go.Figure()
        
        for i, y_col in enumerate(y_columns):
            fig.add_trace(go.Scatter(
                x=pandas_df[x_column],
                y=pandas_df[y_col],
                mode='lines+markers',
                name=y_col,
                line=dict(width=2),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'family': 'Arial, sans-serif'}
            },
            xaxis_title=x_label or x_column,
            yaxis_title=y_label or ', '.join(y_columns),
            template=self.template,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=500
        )
        
        return fig
    
    def create_bar_chart(
        self,
        df: pl.DataFrame,
        x_column: str,
        y_column: str,
        title: str = "Bar Chart",
        horizontal: bool = False,
        color_column: Optional[str] = None
    ) -> go.Figure:
        """Create an interactive bar chart"""
        
        pandas_df = df.to_pandas()
        
        if horizontal:
            fig = px.bar(
                pandas_df,
                y=x_column,
                x=y_column,
                color=color_column,
                title=title,
                orientation='h',
                template=self.template
            )
        else:
            fig = px.bar(
                pandas_df,
                x=x_column,
                y=y_column,
                color=color_column,
                title=title,
                template=self.template
            )
        
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'family': 'Arial, sans-serif'}
            },
            height=500,
            showlegend=color_column is not None
        )
        
        return fig
    
    def create_scatter_plot(
        self,
        df: pl.DataFrame,
        x_column: str,
        y_column: str,
        color_column: Optional[str] = None,
        size_column: Optional[str] = None,
        title: str = "Scatter Plot"
    ) -> go.Figure:
        """Create an interactive scatter plot"""
        
        pandas_df = df.to_pandas()
        
        fig = px.scatter(
            pandas_df,
            x=x_column,
            y=y_column,
            color=color_column,
            size=size_column,
            title=title,
            template=self.template,
            trendline="ols" if color_column is None else None
        )
        
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'family': 'Arial, sans-serif'}
            },
            height=500
        )
        
        fig.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')))
        
        return fig
    
    def create_histogram(
        self,
        df: pl.DataFrame,
        column: str,
        bins: int = 30,
        title: str = "Histogram",
        show_normal: bool = True
    ) -> go.Figure:
        """Create a histogram with optional normal distribution overlay"""
        
        pandas_df = df.to_pandas()
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=pandas_df[column],
            nbinsx=bins,
            name='Distribution',
            marker_color='lightblue',
            marker_line_color='darkblue',
            marker_line_width=1
        ))
        
        if show_normal:
            # Add normal distribution curve
            import numpy as np
            from scipy import stats
            
            data = pandas_df[column].dropna()
            mu, sigma = data.mean(), data.std()
            x_range = np.linspace(data.min(), data.max(), 100)
            
            # Scale to match histogram
            hist, bin_edges = np.histogram(data, bins=bins)
            bin_width = bin_edges[1] - bin_edges[0]
            scale = len(data) * bin_width
            
            y_normal = stats.norm.pdf(x_range, mu, sigma) * scale
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_normal,
                mode='lines',
                name='Normal Distribution',
                line=dict(color='red', width=2, dash='dash')
            ))
        
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'family': 'Arial, sans-serif'}
            },
            xaxis_title=column,
            yaxis_title='Frequency',
            template=self.template,
            height=500,
            bargap=0.1
        )
        
        return fig
    
    def create_box_plot(
        self,
        df: pl.DataFrame,
        column: str,
        group_by: Optional[str] = None,
        title: str = "Box Plot"
    ) -> go.Figure:
        """Create an interactive box plot"""
        
        pandas_df = df.to_pandas()
        
        if group_by:
            fig = px.box(
                pandas_df,
                x=group_by,
                y=column,
                title=title,
                template=self.template,
                points='outliers'
            )
        else:
            fig = px.box(
                pandas_df,
                y=column,
                title=title,
                template=self.template,
                points='outliers'
            )
        
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'family': 'Arial, sans-serif'}
            },
            height=500
        )
        
        return fig
    
    def create_heatmap(
        self,
        corr_matrix: pl.DataFrame,
        title: str = "Correlation Heatmap",
        color_scale: str = "RdBu_r"
    ) -> go.Figure:
        """Create a correlation heatmap"""
        
        pandas_df = corr_matrix.to_pandas()
        
        # Set index if not already set
        if 'index' in pandas_df.columns:
            pandas_df = pandas_df.set_index('index')
        
        fig = go.Figure(data=go.Heatmap(
            z=pandas_df.values,
            x=pandas_df.columns,
            y=pandas_df.index,
            colorscale=color_scale,
            zmid=0,
            text=pandas_df.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'family': 'Arial, sans-serif'}
            },
            template=self.template,
            height=600,
            width=700
        )
        
        return fig
    
    def create_pie_chart(
        self,
        df: pl.DataFrame,
        column: str,
        value_column: Optional[str] = None,
        title: str = "Pie Chart",
        top_n: int = 10
    ) -> go.Figure:
        """Create an interactive pie chart"""
        
        pandas_df = df.to_pandas()
        
        if value_column:
            # Aggregate by category
            data = pandas_df.groupby(column)[value_column].sum().reset_index()
            data = data.nlargest(top_n, value_column)
            
            fig = px.pie(
                data,
                values=value_column,
                names=column,
                title=title,
                template=self.template
            )
        else:
            # Count occurrences
            data = pandas_df[column].value_counts().head(top_n).reset_index()
            data.columns = [column, 'count']
            
            fig = px.pie(
                data,
                values='count',
                names=column,
                title=title,
                template=self.template
            )
        
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'family': 'Arial, sans-serif'}
            },
            height=500
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        return fig
    
    def create_time_series(
        self,
        df: pl.DataFrame,
        date_column: str,
        value_columns: Union[str, List[str]],
        title: str = "Time Series",
        show_range_slider: bool = True
    ) -> go.Figure:
        """Create a time series visualization with range slider"""
        
        if isinstance(value_columns, str):
            value_columns = [value_columns]
        
        pandas_df = df.to_pandas()
        
        fig = go.Figure()
        
        for col in value_columns:
            fig.add_trace(go.Scatter(
                x=pandas_df[date_column],
                y=pandas_df[col],
                mode='lines',
                name=col,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'family': 'Arial, sans-serif'}
            },
            xaxis_title=date_column,
            yaxis_title='Value',
            template=self.template,
            hovermode='x unified',
            height=500
        )
        
        if show_range_slider:
            fig.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )
        
        return fig
    
    def create_multi_chart_dashboard(
        self,
        charts: List[go.Figure],
        rows: int,
        cols: int,
        title: str = "Dashboard"
    ) -> go.Figure:
        """Combine multiple charts into a dashboard"""
        
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[chart.layout.title.text for chart in charts]
        )
        
        for i, chart in enumerate(charts):
            row = i // cols + 1
            col = i % cols + 1
            
            for trace in chart.data:
                fig.add_trace(trace, row=row, col=col)
        
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24, 'family': 'Arial, sans-serif'}
            },
            template=self.template,
            height=400 * rows,
            showlegend=False
        )
        
        return fig
