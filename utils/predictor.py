import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from PIL import Image
import cv2

class Predictor:
    def __init__(self, model, preprocessor, class_names):
        self.model = model
        self.preprocessor = preprocessor
        self.class_names = class_names
        
    def predict(self, image):
        """Predict class for a single image"""
        try:
            # Preprocess image
            processed_image = self.preprocessor.preprocess_single_image(image)
            
            if processed_image is None:
                return None, None
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)[0]
            
            # Get predicted class and confidence
            predicted_class_idx = np.argmax(predictions)
            predicted_class = self.class_names[predicted_class_idx]
            confidence = predictions[predicted_class_idx] * 100
            
            return predicted_class, confidence, predictions
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            return None, None, None
    
    def plot_probabilities(self, predictions):
        """Plot prediction probabilities"""
        fig = go.Figure(data=[
            go.Bar(
                x=self.class_names,
                y=predictions * 100,
                marker_color=['#667eea' if i == np.argmax(predictions) else '#2d3344' 
                             for i in range(len(predictions))],
                text=[f'{prob*100:.1f}%' for prob in predictions],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title='Class Probabilities',
            xaxis_title='Classes',
            yaxis_title='Probability (%)',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400,
            showlegend=False,
            yaxis=dict(range=[0, 100])
        )
        
        # Add threshold line at 50%
        fig.add_hline(y=50, line_dash="dash", line_color="gray", 
                     annotation_text="50% Threshold")
        
        st.plotly_chart(fig, use_container_width=True)
    
    def display_prediction_results(self, predicted_class, confidence, predictions, image):
        """Display prediction results in a nice format"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Prediction Result")
            
            # Create metric cards
            st.markdown(f"""
            <div style="background: #1a1f2e; padding: 1.5rem; border-radius: 1rem; border-left: 4px solid #667eea;">
                <h4 style="color: #a0aec0; margin-bottom: 0.5rem;">Predicted Class</h4>
                <p style="color: white; font-size: 2rem; font-weight: 700;">{predicted_class}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background: #1a1f2e; padding: 1.5rem; border-radius: 1rem; border-left: 4px solid #764ba2; margin-top: 1rem;">
                <h4 style="color: #a0aec0; margin-bottom: 0.5rem;">Confidence</h4>
                <p style="color: white; font-size: 2rem; font-weight: 700;">{confidence:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = confidence,
                title = {'text': "Confidence Score"},
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [None, 100], 'tickcolor': "white"},
                    'bar': {'color': "#667eea"},
                    'steps': [
                        {'range': [0, 50], 'color': "#2d3344"},
                        {'range': [50, 80], 'color': "#4a5568"},
                        {'range': [80, 100], 'color': "#667eea"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': confidence
                    }
                }
            ))
            
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': "white", 'family': "Arial"},
                height=200
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Input Image")
            st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Probability plot
        st.markdown("### Probability Distribution")
        self.plot_probabilities(predictions)
    
    def predict_batch(self, images):
        """Predict for multiple images"""
        predictions = []
        
        for image in images:
            pred_class, confidence, _ = self.predict(image)
            if pred_class:
                predictions.append({
                    'image': image,
                    'predicted_class': pred_class,
                    'confidence': confidence
                })
        
        return predictions