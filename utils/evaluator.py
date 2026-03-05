import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import streamlit as st
import tensorflow as tf

class ModelEvaluator:
    def __init__(self, model, X_test, y_test, class_names):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.class_names = class_names
        self.predictions = None
        self.y_pred_classes = None
        self.y_true_classes = None
        
    def evaluate(self):
        """Evaluate the model"""
        # Get predictions
        self.predictions = self.model.predict(self.X_test, verbose=0)
        self.y_pred_classes = np.argmax(self.predictions, axis=1)
        self.y_true_classes = np.argmax(self.y_test, axis=1)
        
        # Calculate accuracy
        accuracy = accuracy_score(self.y_true_classes, self.y_pred_classes)
        
        return accuracy
    
    def get_classification_report(self):
        """Generate classification report"""
        report = classification_report(
            self.y_true_classes, 
            self.y_pred_classes,
            target_names=self.class_names,
            output_dict=True
        )
        return pd.DataFrame(report).transpose()
    
    def plot_confusion_matrix(self, plot_type='both'):
        """Plot confusion matrix"""
        cm = confusion_matrix(self.y_true_classes, self.y_pred_classes)
        
        if plot_type == 'seaborn' or plot_type == 'both':
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=self.class_names, 
                       yticklabels=self.class_names,
                       ax=ax)
            ax.set_title('Confusion Matrix (Seaborn)')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        if plot_type == 'plotly' or plot_type == 'both':
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=self.class_names,
                y=self.class_names,
                colorscale='Blues',
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 16},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title='Confusion Matrix (Plotly)',
                xaxis_title='Predicted Label',
                yaxis_title='True Label',
                template='plotly_dark',
                height=500,
                width=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training Accuracy', 'Validation Accuracy',
                          'Training Loss', 'Validation Loss')
        )
        
        epochs = list(range(1, len(history['accuracy']) + 1))
        
        # Training Accuracy
        fig.add_trace(
            go.Scatter(x=epochs, y=history['accuracy'], 
                      mode='lines', name='Train Acc',
                      line=dict(color='#667eea', width=2)),
            row=1, col=1
        )
        
        # Validation Accuracy
        fig.add_trace(
            go.Scatter(x=epochs, y=history['val_accuracy'], 
                      mode='lines', name='Val Acc',
                      line=dict(color='#764ba2', width=2)),
            row=1, col=2
        )
        
        # Training Loss
        fig.add_trace(
            go.Scatter(x=epochs, y=history['loss'], 
                      mode='lines', name='Train Loss',
                      line=dict(color='#667eea', width=2)),
            row=2, col=1
        )
        
        # Validation Loss
        fig.add_trace(
            go.Scatter(x=epochs, y=history['val_loss'], 
                      mode='lines', name='Val Loss',
                      line=dict(color='#764ba2', width=2)),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=600,
            showlegend=False,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        fig.update_xaxes(title_text="Epochs", row=2, col=1)
        fig.update_xaxes(title_text="Epochs", row=2, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_class_distribution(self):
        """Plot class distribution"""
        unique, counts = np.unique(self.y_true_classes, return_counts=True)
        
        fig = go.Figure(data=[
            go.Bar(
                x=self.class_names,
                y=counts,
                marker_color=['#667eea', '#764ba2', '#9f7aea', '#b794f4'],
                text=counts,
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title='Class Distribution in Test Set',
            xaxis_title='Classes',
            yaxis_title='Number of Samples',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_per_class_metrics(self):
        """Plot per-class precision, recall, f1-score"""
        report = self.get_classification_report()
        
        # Filter only classes
        class_report = report.loc[self.class_names]
        
        fig = go.Figure()
        
        metrics = ['precision', 'recall', 'f1-score']
        colors = ['#667eea', '#764ba2', '#9f7aea']
        
        for i, metric in enumerate(metrics):
            fig.add_trace(go.Bar(
                name=metric.capitalize(),
                x=self.class_names,
                y=class_report[metric],
                marker_color=colors[i],
                text=class_report[metric].round(3),
                textposition='auto',
            ))
        
        fig.update_layout(
            title='Per-Class Performance Metrics',
            xaxis_title='Classes',
            yaxis_title='Score',
            barmode='group',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)