"""
Visualization Utilities for Gait and Balance Analysis

This module provides functions to visualize gait and balance data,
model performance, and feature importance.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn')
sns.set_palette('colorblind')

class GaitVisualizer:
    """Class for visualizing gait and balance data and model results."""
    
    def __init__(self, output_dir: str = 'reports/figures'):
        """Initialize the visualizer.
        
        Args:
            output_dir: Directory to save output figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_gait_cycle(self, joint_angles: Dict[str, np.ndarray], 
                        sample_rate: float = 100.0,
                        title: str = 'Gait Cycle Analysis',
                        output_file: Optional[str] = None) -> go.Figure:
        """Plot joint angles over a gait cycle.
        
        Args:
            joint_angles: Dictionary of joint angle time series
            sample_rate: Sample rate in Hz
            title: Plot title
            output_file: Path to save the figure (optional)
            
        Returns:
            Plotly figure object
        """
        time = np.arange(len(next(iter(joint_angles.values())))) / sample_rate
        
        fig = go.Figure()
        
        for joint, angles in joint_angles.items():
            fig.add_trace(go.Scatter(
                x=time,
                y=angles,
                mode='lines',
                name=joint,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Time (s)',
            yaxis_title='Joint Angle (degrees)',
            legend_title='Joint',
            template='plotly_white',
            hovermode='x unified'
        )
        
        if output_file:
            output_path = self.output_dir / output_file
            fig.write_html(str(output_path.with_suffix('.html')))
            fig.write_image(str(output_path.with_suffix('.png')))
        
        return fig
    
    def plot_feature_importance(self, feature_importance: Dict[str, float], 
                              top_n: int = 20,
                              title: str = 'Feature Importance',
                              output_file: Optional[str] = None) -> go.Figure:
        """Plot feature importance scores.
        
        Args:
            feature_importance: Dictionary of feature names and importance scores
            top_n: Number of top features to display
            title: Plot title
            output_file: Path to save the figure (optional)
            
        Returns:
            Plotly figure object
        """
        # Sort features by importance
        features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        
        if top_n > 0:
            features = features[:top_n]
        
        feature_names, importance_scores = zip(*features)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=importance_scores,
            y=feature_names,
            orientation='h',
            marker_color='skyblue'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Importance Score',
            yaxis_title='Feature',
            template='plotly_white',
            height=600,
            yaxis=dict(autorange="reversed")
        )
        
        if output_file:
            output_path = self.output_dir / output_file
            fig.write_html(str(output_path.with_suffix('.html')))
            fig.write_image(str(output_path.with_suffix('.png')))
        
        return fig
    
    def plot_confusion_matrix(self, cm: np.ndarray, 
                            class_names: List[str],
                            title: str = 'Confusion Matrix',
                            normalize: bool = True,
                            output_file: Optional[str] = None) -> go.Figure:
        """Plot a confusion matrix.
        
        Args:
            cm: Confusion matrix (n_classes, n_classes)
            class_names: List of class names
            title: Plot title
            normalize: Whether to normalize the confusion matrix
            output_file: Path to save the figure (optional)
            
        Returns:
            Plotly figure object
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            z_text = [[f"{val:.2f}" for val in row] for row in cm]
        else:
            z_text = [[str(val) for val in row] for row in cm]
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            text=z_text,
            texttemplate="%{text}",
            colorscale='Blues',
            colorbar=dict(title='Proportion' if normalize else 'Count')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Predicted Label',
            yaxis_title='True Label',
            template='plotly_white',
            width=800,
            height=700
        )
        
        if output_file:
            output_path = self.output_dir / output_file
            fig.write_html(str(output_path.with_suffix('.html')))
            fig.write_image(str(output_path.with_suffix('.png')))
        
        return fig
    
    def plot_roc_curve(self, fpr: np.ndarray, tpr: np.ndarray, 
                      roc_auc: float,
                      title: str = 'ROC Curve',
                      output_file: Optional[str] = None) -> go.Figure:
        """Plot a ROC curve.
        
        Args:
            fpr: False positive rates
            tpr: True positive rates
            roc_auc: Area under the ROC curve
            title: Plot title
            output_file: Path to save the figure (optional)
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # ROC curve
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'ROC curve (AUC = {roc_auc:.2f})',
            line=dict(color='darkorange', width=2)
        ))
        
        # Random chance line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random chance',
            line=dict(color='navy', width=1, dash='dash')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            template='plotly_white',
            width=700,
            height=600,
            showlegend=True
        )
        
        if output_file:
            output_path = self.output_dir / output_file
            fig.write_html(str(output_path.with_suffix('.html')))
            fig.write_image(str(output_path.with_suffix('.png')))
        
        return fig
    
    def plot_learning_curve(self, train_scores: np.ndarray, 
                           val_scores: np.ndarray,
                           train_sizes: np.ndarray,
                           title: str = 'Learning Curve',
                           output_file: Optional[str] = None) -> go.Figure:
        """Plot a learning curve.
        
        Args:
            train_scores: Training scores for different training set sizes
            val_scores: Validation scores for different training set sizes
            train_sizes: Training set sizes
            title: Plot title
            output_file: Path to save the figure (optional)
            
        Returns:
            Plotly figure object
        """
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        fig = go.Figure()
        
        # Training scores
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=train_mean,
            mode='lines+markers',
            name='Training score',
            line=dict(color='blue', width=2),
            error_y=dict(
                type='data',
                array=train_std,
                visible=True,
                thickness=1.5,
                width=3
            )
        ))
        
        # Validation scores
        fig.add_trace(go.Scatter(
            x=train_sizes,
            y=val_mean,
            mode='lines+markers',
            name='Cross-validation score',
            line=dict(color='green', width=2),
            error_y=dict(
                type='data',
                array=val_std,
                visible=True,
                thickness=1.5,
                width=3
            )
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Training Examples',
            yaxis_title='Score',
            template='plotly_white',
            showlegend=True,
            width=800,
            height=600
        )
        
        if output_file:
            output_path = self.output_dir / output_file
            fig.write_html(str(output_path.with_suffix('.html')))
            fig.write_image(str(output_path.with_suffix('.png')))
        
        return fig
    
    def plot_time_series(self, time_series: Dict[str, np.ndarray],
                        sample_rate: float = 100.0,
                        title: str = 'Time Series Data',
                        output_file: Optional[str] = None) -> go.Figure:
        """Plot multiple time series.
        
        Args:
            time_series: Dictionary of time series data
            sample_rate: Sample rate in Hz
            title: Plot title
            output_file: Path to save the figure (optional)
            
        Returns:
            Plotly figure object
        """
        time = np.arange(len(next(iter(time_series.values())))) / sample_rate
        
        fig = go.Figure()
        
        for name, data in time_series.items():
            fig.add_trace(go.Scatter(
                x=time,
                y=data,
                mode='lines',
                name=name,
                line=dict(width=1.5)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Time (s)',
            yaxis_title='Value',
            template='plotly_white',
            hovermode='x unified',
            height=500
        )
        
        if output_file:
            output_path = self.output_dir / output_file
            fig.write_html(str(output_path.with_suffix('.html')))
            fig.write_image(str(output_path.with_suffix('.png')))
        
        return fig
    
    def plot_correlation_heatmap(self, data: pd.DataFrame,
                               title: str = 'Feature Correlation Heatmap',
                               output_file: Optional[str] = None) -> go.Figure:
        """Plot a correlation heatmap of features.
        
        Args:
            data: DataFrame containing features
            title: Plot title
            output_file: Path to save the figure (optional)
            
        Returns:
            Plotly figure object
        """
        corr = data.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale='RdBu_r',
            zmin=-1,
            zmax=1,
            colorbar=dict(title='Correlation')
        ))
        
        fig.update_layout(
            title=title,
            template='plotly_white',
            width=900,
            height=800,
            xaxis=dict(tickangle=-45)
        )
        
        if output_file:
            output_path = self.output_dir / output_file
            fig.write_html(str(output_path.with_suffix('.html')))
            fig.write_image(str(output_path.with_suffix('.png')))
        
        return fig
    
    def plot_3d_trajectory(self, trajectory: np.ndarray,
                          title: str = '3D Trajectory',
                          labels: Optional[List[str]] = None,
                          output_file: Optional[str] = None) -> go.Figure:
        """Plot a 3D trajectory.
        
        Args:
            trajectory: 3D trajectory data (n_samples, 3)
            title: Plot title
            labels: Labels for the trajectory points (optional)
            output_file: Path to save the figure (optional)
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Create scatter plot
        scatter = go.Scatter3d(
            x=trajectory[:, 0],
            y=trajectory[:, 1],
            z=trajectory[:, 2],
            mode='lines+markers',
            marker=dict(
                size=4,
                color=np.arange(len(trajectory)),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Time')
            ),
            line=dict(
                color='darkblue',
                width=2
            )
        )
        
        if labels is not None:
            scatter.text = labels
        
        fig.add_trace(scatter)
        
        # Add start and end markers
        fig.add_trace(go.Scatter3d(
            x=[trajectory[0, 0]],
            y=[trajectory[0, 1]],
            z=[trajectory[0, 2]],
            mode='markers',
            marker=dict(
                size=8,
                color='green',
                symbol='circle'
            ),
            name='Start'
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[trajectory[-1, 0]],
            y=[trajectory[-1, 1]],
            z=[trajectory[-1, 2]],
            mode='markers',
            marker=dict(
                size=8,
                color='red',
                symbol='x'
            ),
            name='End'
        ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            width=900,
            height=700,
            showlegend=True
        )
        
        if output_file:
            output_path = self.output_dir / output_file
            fig.write_html(str(output_path.with_suffix('.html')))
            fig.write_image(str(output_path.with_suffix('.png')))
        
        return fig
