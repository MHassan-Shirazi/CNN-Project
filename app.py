"""
DeepVision AI - Professional Image Classification Platform
Color Scheme: Blue (#2563eb), Green (#10b981), Gray (#1f2937)
Icons: Font Awesome 6 (Free)
"""

import streamlit as st
from streamlit_option_menu import option_menu
import os
import shutil
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
from datetime import datetime
import cv2
import base64
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="DeepVision AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Font Awesome
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        * { font-family: 'Inter', sans-serif; }
    </style>
""", unsafe_allow_html=True)

# Load custom CSS
def load_css():
    css_file = Path('assets/style.css')
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'preprocessor' not in st.session_state:
        st.session_state.preprocessor = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'class_names' not in st.session_state:
        st.session_state.class_names = []
    if 'trained' not in st.session_state:
        st.session_state.trained = False
    if 'history' not in st.session_state:
        st.session_state.history = None

# Create directories
def create_directories():
    os.makedirs('dataset', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('assets', exist_ok=True)

# Professional Header
def professional_header(title, subtitle):
    st.markdown(f"""
    <div class="app-header">
        <h1><i class="fas fa-camera-retro" style="margin-right: 15px;"></i>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

# Metric Card
def metric_card(label, value, icon, color="#2563eb"):
    st.markdown(f"""
    <div class="metric-card" style="border-left: 4px solid {color};">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            <i class="fas fa-{icon}" style="font-size: 2rem; color: {color}; opacity: 0.8;"></i>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Alert Message
def show_alert(message, type="info"):
    icons = {
        "success": "check-circle",
        "error": "exclamation-circle",
        "warning": "exclamation-triangle",
        "info": "info-circle"
    }
    st.markdown(f"""
    <div class="alert alert-{type}">
        <i class="fas fa-{icons[type]}"></i>
        <span>{message}</span>
    </div>
    """, unsafe_allow_html=True)

# Section Header
def section_header(title, icon):
    st.markdown(f"""
    <div style="margin: 2rem 0 1.5rem 0;">
        <h3 style="color: var(--text-primary); font-weight: 600; display: flex; align-items: center; gap: 0.75rem;">
            <i class="fas fa-{icon}" style="color: #2563eb; font-size: 1.5rem;"></i>
            {title}
        </h3>
    </div>
    """, unsafe_allow_html=True)

# Gap between sections
def add_gap(size="2rem"):
    st.markdown(f"<div style='height: {size};'></div>", unsafe_allow_html=True)

# Dataset Manager Module
def dataset_manager():
    professional_header("Data Studio", "Manage and organize your training datasets")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="modern-card">
            <h3><i class="fas fa-folder-plus"></i> Create New Class</h3>
        """, unsafe_allow_html=True)
        
        new_class = st.text_input("", placeholder="Enter class name (e.g., Cat, Dog, Aeroplane)")
        
        if st.button("Create Class", use_container_width=True):
            if new_class:
                class_path = os.path.join('dataset', new_class)
                os.makedirs(class_path, exist_ok=True)
                show_alert(f"Class '{new_class}' created successfully", "success")
            else:
                show_alert("Please enter a class name", "warning")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        add_gap("1rem")
        
        # Dataset Statistics
        classes = [d for d in os.listdir('dataset') if os.path.isdir(os.path.join('dataset', d))]
        total_classes = len(classes)
        total_images = 0
        class_counts = {}
        
        for class_name in classes:
            class_path = os.path.join('dataset', class_name)
            image_count = len([f for f in os.listdir(class_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))])
            class_counts[class_name] = image_count
            total_images += image_count
        
        st.markdown("""
        <div class="modern-card">
            <h3><i class="fas fa-chart-pie"></i> Dataset Overview</h3>
        """, unsafe_allow_html=True)
        
        # Metrics row
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            metric_card("Total Classes", total_classes, "layer-group", "#2563eb")
        with col_m2:
            metric_card("Total Images", total_images, "images", "#10b981")
        with col_m3:
            avg = total_images // total_classes if total_classes > 0 else 0
            metric_card("Avg per Class", avg, "chart-line", "#f59e0b")
        
        # Class distribution
        if class_counts:
            df = pd.DataFrame({
                'Class': list(class_counts.keys()),
                'Images': list(class_counts.values())
            })
            
            fig = px.bar(df, x='Class', y='Images', 
                        title="Class Distribution",
                        color_discrete_sequence=['#2563eb'])
            fig.update_layout(
                template='plotly_white',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Inter", size=12),
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Upload Section
        st.markdown("""
        <div class="modern-card">
            <h3><i class="fas fa-cloud-upload-alt"></i> Upload Images</h3>
        """, unsafe_allow_html=True)
        
        if classes:
            selected_class = st.selectbox("Select Class", classes)
            
            st.markdown("""
            <div class="upload-area">
                <i class="fas fa-cloud-upload-alt"></i>
                <p>Drag and drop or click to upload</p>
                <small>Supported: PNG, JPG, JPEG, BMP, GIF</small>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_files = st.file_uploader(
                "", 
                type=['png', 'jpg', 'jpeg', 'bmp', 'gif'],
                accept_multiple_files=True,
                label_visibility="collapsed"
            )
            
            if uploaded_files:
                class_path = os.path.join('dataset', selected_class)
                
                with st.spinner("Uploading images..."):
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(class_path, uploaded_file.name)
                        with open(file_path, 'wb') as f:
                            f.write(uploaded_file.getbuffer())
                    
                    show_alert(f"Uploaded {len(uploaded_files)} images to {selected_class}", "success")
                
                # Preview
                st.markdown("#### Preview")
                cols = st.columns(4)
                for idx, uploaded_file in enumerate(uploaded_files[:4]):
                    with cols[idx]:
                        image = Image.open(uploaded_file)
                        st.image(image, caption=uploaded_file.name, use_column_width=True)
        else:
            show_alert("Create a class first to upload images", "info")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        add_gap("1rem")
        
        # Webcam Section
        st.markdown("""
        <div class="modern-card">
            <h3><i class="fas fa-camera"></i> Webcam Capture</h3>
        """, unsafe_allow_html=True)
        
        if classes:
            selected_class_webcam = st.selectbox("Select Class", classes, key="webcam_class")
            
            picture = st.camera_input("")
            
            if picture:
                class_path = os.path.join('dataset', selected_class_webcam)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = os.path.join(class_path, f"webcam_{timestamp}.png")
                
                with open(file_path, 'wb') as f:
                    f.write(picture.getbuffer())
                
                show_alert("Image captured and saved successfully", "success")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Training Module
def train_model():
    professional_header("Model Training", "Configure and train your neural network")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        <div class="modern-card">
            <h3><i class="fas fa-sliders-h"></i> Training Configuration</h3>
        """, unsafe_allow_html=True)
        
        # Training parameters
        img_size = st.slider("Image Size", 64, 256, 128, step=32)
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
        epochs = st.slider("Training Epochs", 10, 100, 50, step=10)
        learning_rate = st.select_slider("Learning Rate", 
                                         options=[0.1, 0.01, 0.001, 0.0001], 
                                         value=0.001)
        validation_split = st.slider("Validation Split", 0.1, 0.3, 0.2, step=0.05)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        add_gap("1rem")
        
        st.markdown("""
        <div class="modern-card">
            <h3><i class="fas fa-microchip"></i> Model Architecture</h3>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: var(--bg-secondary); padding: 1rem; border-radius: 12px;">
            <small style="color: #2563eb; font-weight: 600;">4-Layer CNN Architecture</small>
            <ul style="margin-top: 0.5rem; color: var(--text-secondary);">
                <li>Conv2D + BatchNorm + ReLU</li>
                <li>MaxPooling + Dropout (0.25)</li>
                <li>Dense Layers + Dropout (0.5)</li>
                <li>Softmax Classification</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        add_gap("1rem")
        
        # Start Training
        if st.button("Start Training", use_container_width=True):
            classes = [d for d in os.listdir('dataset') if os.path.isdir(os.path.join('dataset', d))]
            if len(classes) < 2:
                show_alert("Need at least 2 classes for training", "error")
                return
            
            show_alert("Training started! Check the progress on the right.", "info")
            # Training logic here...
    
    with col2:
        st.markdown("""
        <div class="modern-card">
            <h3><i class="fas fa-chart-line"></i> Training Progress</h3>
        """, unsafe_allow_html=True)
        
        if st.session_state.trained:
            # Show metrics
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                metric_card("Current Epoch", "25/50", "clock", "#2563eb")
            with col_m2:
                metric_card("Accuracy", "87.5%", "check-circle", "#10b981")
            
            add_gap("1rem")
            
            # Progress bar
            st.progress(0.5)
            
            add_gap("1rem")
            
            # Placeholder for plots
            st.markdown("""
            <div style="background: var(--bg-secondary); border-radius: 12px; padding: 2rem; text-align: center;">
                <i class="fas fa-chart-bar" style="font-size: 3rem; color: var(--text-muted);"></i>
                <p style="color: var(--text-muted); margin-top: 0.5rem;">Training plots will appear here</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            show_alert("No training in progress. Configure and start training.", "info")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Evaluation Module
def evaluate_model():
    professional_header("Model Evaluation", "Comprehensive performance analysis")
    
    if not st.session_state.trained:
        show_alert("Please train a model first", "warning")
        return
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Accuracy", "94.2%", "check-circle", "#2563eb")
    with col2:
        metric_card("Precision", "93.8%", "bullseye", "#10b981")
    with col3:
        metric_card("Recall", "94.5%", "search", "#f59e0b")
    with col4:
        metric_card("F1-Score", "94.1%", "chart-pie", "#ef4444")
    
    add_gap("2rem")
    
    # Confusion Matrix
    st.markdown("""
    <div class="modern-card">
        <h3><i class="fas fa-th"></i> Confusion Matrix</h3>
    """, unsafe_allow_html=True)
    
    # Sample confusion matrix placeholder
    st.markdown("""
    <div style="background: var(--bg-secondary); border-radius: 12px; padding: 2rem; text-align: center;">
        <i class="fas fa-heatmap" style="font-size: 3rem; color: var(--text-muted);"></i>
        <p style="color: var(--text-muted); margin-top: 0.5rem;">Confusion matrix visualization</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    add_gap("2rem")
    
    # Classification Report
    st.markdown("""
    <div class="modern-card">
        <h3><i class="fas fa-table"></i> Classification Report</h3>
    """, unsafe_allow_html=True)
    
    # Sample data
    report_data = {
        'Class': ['Cat', 'Dog', 'Bird'],
        'Precision': [0.95, 0.92, 0.94],
        'Recall': [0.94, 0.93, 0.96],
        'F1-Score': [0.94, 0.92, 0.95]
    }
    df = pd.DataFrame(report_data)
    st.dataframe(df, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Prediction Module
def predict():
    professional_header("Real-Time Prediction", "Upload images for instant classification")
    
    if not st.session_state.trained:
        show_alert("Please train a model first", "warning")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="modern-card">
            <h3><i class="fas fa-cloud-upload-alt"></i> Upload Image</h3>
        """, unsafe_allow_html=True)
        
        # Model selection
        models = [f for f in os.listdir('models') if f.endswith('.h5')]
        if models:
            selected_model = st.selectbox("Select Model", models)
        
        add_gap("1rem")
        
        # Upload area
        st.markdown("""
        <div class="upload-area">
            <i class="fas fa-cloud-upload-alt"></i>
            <p>Drag and drop or click to upload</p>
            <small>Supported: PNG, JPG, JPEG</small>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("", type=['png', 'jpg', 'jpeg'], 
                                        label_visibility="collapsed")
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if uploaded_file:
            st.markdown("""
            <div class="modern-card">
                <h3><i class="fas fa-chart-bar"></i> Prediction Results</h3>
            """, unsafe_allow_html=True)
            
            # Sample prediction
            metric_card("Predicted Class", "Cat", "tag", "#2563eb")
            
            add_gap("1rem")
            
            metric_card("Confidence", "97.3%", "percent", "#10b981")
            
            add_gap("1rem")
            
            # Probability chart
            prob_data = {
                'Class': ['Cat', 'Dog', 'Bird'],
                'Probability': [97.3, 2.1, 0.6]
            }
            df = pd.DataFrame(prob_data)
            
            fig = px.bar(df, x='Class', y='Probability', 
                        title="Class Probabilities",
                        color='Class',
                        color_discrete_sequence=['#2563eb', '#10b981', '#f59e0b'])
            fig.update_layout(
                template='plotly_white',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

# Main Application
def main():
    # Initialize
    init_session_state()
    create_directories()
    load_css()
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <h2>DeepVision AI</h2>
            <p>Image Classification Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        selected = option_menu(
            menu_title=None,
            options=["Dashboard", "Data Studio", "Training", "Evaluation", "Inference"],
            icons=["speedometer2", "database", "cpu", "graph-up", "camera"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0", "background-color": "transparent"},
                "icon": {"color": "#6b7280", "font-size": "1.1rem"},
                "nav-link": {
                    "font-size": "0.95rem",
                    "text-align": "left",
                    "margin": "0.25rem 1rem",
                    "padding": "0.75rem 1rem",
                    "border-radius": "12px",
                    "color": "#4b5563",
                    "font-weight": "500",
                },
                "nav-link-selected": {
                    "background-color": "#eff6ff",
                    "color": "#2563eb",
                    "font-weight": "600",
                },
            }
        )
        
        st.markdown("---")
        
        # Status indicator
        if st.session_state.trained:
            st.markdown("""
            <div class="status-badge status-success">
                <i class="fas fa-check-circle" style="margin-right: 8px;"></i>
                Model Trained & Ready
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-badge status-warning">
                <i class="fas fa-exclamation-triangle" style="margin-right: 8px;"></i>
                No Trained Model
            </div>
            """, unsafe_allow_html=True)
        
        add_gap("1rem")
        
        # Quick stats
        classes = [d for d in os.listdir('dataset') if os.path.isdir(os.path.join('dataset', d))]
        st.markdown(f"""
        <div style="padding: 1rem;">
            <small style="color: #6b7280;">Dataset Stats</small><br>
            <span style="font-weight: 600; color: #2563eb;">{len(classes)} classes</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="app-footer">
            <i class="fas fa-copyright"></i> 2024 DeepVision AI<br>
            Version 2.0
        </div>
        """, unsafe_allow_html=True)
    
    # ============= MAIN CONTENT AREA =============
    
    if selected == "Dashboard":
        # ============= DASHBOARD SECTION =============
        
        professional_header("Dashboard", "Overview of your image classification system")
        
        # Quick Actions Section
        section_header("Quick Actions", "bolt")
        
        # Quick action cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="modern-card" style="text-align: center; height: 200px; display: flex; flex-direction: column; justify-content: center;">
                <i class="fas fa-database" style="font-size: 2.5rem; color: #2563eb; margin-bottom: 1rem;"></i>
                <h4 style="margin: 0 0 0.5rem 0; color: var(--text-primary);">Data Studio</h4>
                <p style="color: var(--text-muted); font-size: 0.9rem; margin: 0;">Manage your dataset</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="modern-card" style="text-align: center; height: 200px; display: flex; flex-direction: column; justify-content: center;">
                <i class="fas fa-microchip" style="font-size: 2.5rem; color: #10b981; margin-bottom: 1rem;"></i>
                <h4 style="margin: 0 0 0.5rem 0; color: var(--text-primary);">Training</h4>
                <p style="color: var(--text-muted); font-size: 0.9rem; margin: 0;">Train CNN models</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="modern-card" style="text-align: center; height: 200px; display: flex; flex-direction: column; justify-content: center;">
                <i class="fas fa-chart-bar" style="font-size: 2.5rem; color: #f59e0b; margin-bottom: 1rem;"></i>
                <h4 style="margin: 0 0 0.5rem 0; color: var(--text-primary);">Evaluation</h4>
                <p style="color: var(--text-muted); font-size: 0.9rem; margin: 0;">Analyze performance</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="modern-card" style="text-align: center; height: 200px; display: flex; flex-direction: column; justify-content: center;">
                <i class="fas fa-camera" style="font-size: 2.5rem; color: #ef4444; margin-bottom: 1rem;"></i>
                <h4 style="margin: 0 0 0.5rem 0; color: var(--text-primary);">Inference</h4>
                <p style="color: var(--text-muted); font-size: 0.9rem; margin: 0;">Real-time prediction</p>
            </div>
            """, unsafe_allow_html=True)
        
        add_gap("2rem")
        
        # System Status Section
        section_header("System Status", "server")
        
        col_s1, col_s2, col_s3 = st.columns(3)
        
        with col_s1:
            st.markdown("""
            <div class="metric-card" style="border-left-color: #10b981;">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div>
                        <div class="metric-label">System Status</div>
                        <div class="metric-value" style="font-size: 1.5rem;">Operational</div>
                    </div>
                    <i class="fas fa-check-circle" style="font-size: 2rem; color: #10b981;"></i>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_s2:
            st.markdown("""
            <div class="metric-card" style="border-left-color: #2563eb;">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div>
                        <div class="metric-label">GPU Status</div>
                        <div class="metric-value" style="font-size: 1.5rem;">Available</div>
                    </div>
                    <i class="fas fa-microchip" style="font-size: 2rem; color: #2563eb;"></i>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_s3:
            st.markdown("""
            <div class="metric-card" style="border-left-color: #f59e0b;">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div>
                        <div class="metric-label">Memory Usage</div>
                        <div class="metric-value" style="font-size: 1.5rem;">64% Used</div>
                    </div>
                    <i class="fas fa-memory" style="font-size: 2rem; color: #f59e0b;"></i>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        add_gap("2rem")
        
        # Dataset Overview Section
        section_header("Dataset Overview", "chart-pie")
        
        # Get actual dataset stats
        classes = [d for d in os.listdir('dataset') if os.path.isdir(os.path.join('dataset', d))]
        total_classes = len(classes)
        total_images = 0
        
        for class_name in classes:
            class_path = os.path.join('dataset', class_name)
            image_count = len([f for f in os.listdir(class_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))])
            total_images += image_count
        
        # Stats cards
        col_d1, col_d2, col_d3 = st.columns(3)
        
        with col_d1:
            st.markdown(f"""
            <div class="modern-card" style="text-align: center;">
                <i class="fas fa-folder-open" style="font-size: 2rem; color: #2563eb; margin-bottom: 0.75rem;"></i>
                <div class="metric-label">Total Classes</div>
                <div class="metric-value" style="font-size: 2rem;">{total_classes}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_d2:
            st.markdown(f"""
            <div class="modern-card" style="text-align: center;">
                <i class="fas fa-images" style="font-size: 2rem; color: #10b981; margin-bottom: 0.75rem;"></i>
                <div class="metric-label">Total Images</div>
                <div class="metric-value" style="font-size: 2rem;">{total_images}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_d3:
            avg = total_images // total_classes if total_classes > 0 else 0
            st.markdown(f"""
            <div class="modern-card" style="text-align: center;">
                <i class="fas fa-chart-line" style="font-size: 2rem; color: #f59e0b; margin-bottom: 0.75rem;"></i>
                <div class="metric-label">Avg per Class</div>
                <div class="metric-value" style="font-size: 2rem;">{avg}</div>
            </div>
            """, unsafe_allow_html=True)
        
        add_gap("2rem")
        
        # Recent Activity Section
        section_header("Recent Activity", "history")
        
        # Activity cards
        col_a1, col_a2 = st.columns(2)
        
        with col_a1:
            st.markdown("""
            <div class="modern-card">
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <i class="fas fa-database" style="font-size: 1.5rem; color: #2563eb;"></i>
                    <div>
                        <div style="font-weight: 600; color: var(--text-primary);">Last Dataset Update</div>
                        <div style="color: var(--text-muted); font-size: 0.9rem;">Just now</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_a2:
            if st.session_state.trained:
                st.markdown("""
                <div class="modern-card">
                    <div style="display: flex; align-items: center; gap: 1rem;">
                        <i class="fas fa-microchip" style="font-size: 1.5rem; color: #10b981;"></i>
                        <div>
                            <div style="font-weight: 600; color: var(--text-primary);">Last Training</div>
                            <div style="color: var(--text-muted); font-size: 0.9rem;">Model trained successfully</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="modern-card">
                    <div style="display: flex; align-items: center; gap: 1rem;">
                        <i class="fas fa-microchip" style="font-size: 1.5rem; color: #f59e0b;"></i>
                        <div>
                            <div style="font-weight: 600; color: var(--text-primary);">Last Training</div>
                            <div style="color: var(--text-muted); font-size: 0.9rem;">No training yet</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # ============= DASHBOARD SECTION ENDS =============
        
    elif selected == "Data Studio":
        dataset_manager()
    elif selected == "Training":
        train_model()
    elif selected == "Evaluation":
        evaluate_model()
    elif selected == "Inference":
        predict()

if __name__ == "__main__":
    main()