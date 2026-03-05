import os
import cv2
import numpy as np
from PIL import Image
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf

class DataPreprocessor:
    def __init__(self, img_size=(128, 128)):
        self.img_size = img_size
        self.label_encoder = LabelEncoder()
        
    def load_and_preprocess_image(self, image_path):
        """Load and preprocess a single image"""
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                # Try with PIL if OpenCV fails
                img = Image.open(image_path)
                img = np.array(img)
                if len(img.shape) == 2:  # Grayscale
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 4:  # RGBA
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize
            img = cv2.resize(img, self.img_size)
            
            # Normalize
            img = img.astype(np.float32) / 255.0
            
            return img
        except Exception as e:
            st.error(f"Error loading image {image_path}: {str(e)}")
            return None
    
    def load_dataset(self, dataset_path='dataset'):
        """Load all images from dataset folder"""
        images = []
        labels = []
        class_names = []
        
        if not os.path.exists(dataset_path):
            return np.array(images), np.array(labels), class_names
        
        # Get all class folders
        classes = [d for d in os.listdir(dataset_path) 
                  if os.path.isdir(os.path.join(dataset_path, d))]
        class_names = sorted(classes)
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, class_name in enumerate(class_names):
            class_path = os.path.join(dataset_path, class_name)
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                img = self.load_and_preprocess_image(img_path)
                
                if img is not None:
                    images.append(img)
                    labels.append(class_name)
            
            # Update progress
            progress = (idx + 1) / len(classes)
            progress_bar.progress(progress)
            status_text.text(f"Loading class {idx+1}/{len(classes)}: {class_name}")
        
        progress_bar.empty()
        status_text.empty()
        
        return np.array(images), np.array(labels), class_names
    
    def prepare_data(self, images, labels, test_size=0.2, random_state=42):
        """Prepare data for training"""
        if len(images) == 0:
            return None, None, None, None, None, None
        
        # Encode labels
        labels_encoded = self.label_encoder.fit_transform(labels)
        num_classes = len(self.label_encoder.classes_)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels_encoded, test_size=test_size, 
            random_state=random_state, stratify=labels_encoded
        )
        
        # Convert labels to categorical
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)
        
        return X_train, X_test, y_train, y_test, num_classes, self.label_encoder.classes_
    
    def preprocess_single_image(self, image):
        """Preprocess a single image for prediction"""
        try:
            if isinstance(image, str):  # If image path
                img = cv2.imread(image)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:  # If image array
                img = np.array(image)
            
            # Resize
            img = cv2.resize(img, self.img_size)
            
            # Normalize
            img = img.astype(np.float32) / 255.0
            
            # Add batch dimension
            img = np.expand_dims(img, axis=0)
            
            return img
        except Exception as e:
            st.error(f"Error preprocessing image: {str(e)}")
            return None