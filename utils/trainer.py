import tensorflow as tf
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import os

class ModelTrainer:
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = y_test
        self.y_test = y_test
        self.history = None
        
    def train(self, epochs=50, batch_size=32):
        """Train the model with real-time updates"""
        
        # Create progress tracking
        progress_bar = st.progress(0)
        epoch_text = st.empty()
        metrics_text = st.empty()
        
        # Create placeholder for live plots
        plot_placeholder = st.empty()
        
        # Initialize history
        history_dict = {
            'accuracy': [], 'val_accuracy': [],
            'loss': [], 'val_loss': []
        }
        
        # Data augmentation
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Custom training loop for live updates
        for epoch in range(epochs):
            # Train one epoch
            history_epoch = self.model.fit(
                datagen.flow(self.X_train, self.y_train, batch_size=batch_size),
                validation_data=(self.X_test, self.y_test),
                epochs=1,
                verbose=0,
                steps_per_epoch=len(self.X_train) // batch_size
            )
            
            # Update history
            for key in history_dict.keys():
                if key in history_epoch.history:
                    history_dict[key].extend(history_epoch.history[key])
            
            # Update progress
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            
            # Update epoch text
            epoch_text.text(f"Epoch {epoch + 1}/{epochs}")
            
            # Update metrics text
            metrics_text.text(
                f"Train Loss: {history_dict['loss'][-1]:.4f} | "
                f"Train Acc: {history_dict['accuracy'][-1]:.4f} | "
                f"Val Loss: {history_dict['val_loss'][-1]:.4f} | "
                f"Val Acc: {history_dict['val_accuracy'][-1]:.4f}"
            )
            
            # Update plots every 5 epochs or at the end
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                fig = self.create_live_plots(history_dict)
                plot_placeholder.plotly_chart(fig, use_container_width=True)
        
        # Store history
        self.history = history_dict
        
        # Clear progress indicators
        progress_bar.empty()
        epoch_text.empty()
        metrics_text.empty()
        
        return history_dict
    
    def create_live_plots(self, history_dict):
        """Create live training plots"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Model Accuracy', 'Model Loss')
        )
        
        epochs = list(range(1, len(history_dict['accuracy']) + 1))
        
        # Accuracy plot
        fig.add_trace(
            go.Scatter(x=epochs, y=history_dict['accuracy'], 
                      mode='lines', name='Train Accuracy',
                      line=dict(color='#667eea', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=history_dict['val_accuracy'], 
                      mode='lines', name='Validation Accuracy',
                      line=dict(color='#764ba2', width=2)),
            row=1, col=1
        )
        
        # Loss plot
        fig.add_trace(
            go.Scatter(x=epochs, y=history_dict['loss'], 
                      mode='lines', name='Train Loss',
                      line=dict(color='#667eea', width=2)),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=history_dict['val_loss'], 
                      mode='lines', name='Validation Loss',
                      line=dict(color='#764ba2', width=2)),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=400,
            showlegend=True,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        fig.update_xaxes(title_text="Epochs", row=1, col=1)
        fig.update_xaxes(title_text="Epochs", row=1, col=2)
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=2)
        
        return fig
    
    def save_training_history(self, path='training_history.npy'):
        """Save training history"""
        if self.history:
            np.save(path, self.history)
            return True
        return False