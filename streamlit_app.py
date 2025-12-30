"""
Steel Defect Detection - Streamlit Web Interface (Simplified)

This version works without Grad-CAM for easier installation.
Still includes all core functionality:
- Single image upload and prediction
- Batch processing
- Confidence visualization
- Model performance metrics
"""

import streamlit as st
import torch
import torch.nn as nn
import timm
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import json

# Page configuration
st.set_page_config(
    page_title="Steel Defect Detection System",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
PROJECT_ROOT = Path(r"C:\Projects\CV_SDT")
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Configuration
CONFIG = {
    'model_name': 'convnextv2_tiny.fcmae_ft_in22k_in1k',
    'num_classes': 6,
    'image_size': 224,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

DEFECT_CLASSES = [
    'Crazing',
    'Inclusion',
    'Patches',
    'Pitted Surface',
    'Rolled-in Scale',
    'Scratches'
]

DEFECT_DESCRIPTIONS = {
    'Crazing': 'Fine cracks on the steel surface, often appearing in a network pattern.',
    'Inclusion': 'Non-metallic particles embedded in the steel surface.',
    'Patches': 'Irregular patches or spots on the surface due to coating issues.',
    'Pitted Surface': 'Small holes or pits caused by corrosion or manufacturing defects.',
    'Rolled-in Scale': 'Scale material pressed into the surface during rolling process.',
    'Scratches': 'Linear marks or scratches on the steel surface.'
}

DEFECT_COLORS = {
    'Crazing': '#e74c3c',
    'Inclusion': '#3498db',
    'Patches': '#2ecc71',
    'Pitted Surface': '#f39c12',
    'Rolled-in Scale': '#9b59b6',
    'Scratches': '#1abc9c'
}

@st.cache_resource
def load_model():
    """Load the trained model (cached for performance)."""
    model_files = list(MODELS_DIR.glob('best_model_*.pth'))
    if not model_files:
        st.error("No trained model found! Please train the model first.")
        st.stop()
    
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    
    model = timm.create_model(
        CONFIG['model_name'],
        pretrained=False,
        num_classes=CONFIG['num_classes']
    )
    
    checkpoint = torch.load(latest_model, map_location=CONFIG['device'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(CONFIG['device'])
    model.eval()
    
    return model, checkpoint

def get_transforms():
    """Get image preprocessing transforms."""
    return A.Compose([
        A.Resize(CONFIG['image_size'], CONFIG['image_size']),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def preprocess_image(image):
    """Preprocess uploaded image for model inference."""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    original = image.copy()
    
    transform = get_transforms()
    transformed = transform(image=image)
    tensor = transformed['image'].unsqueeze(0).to(CONFIG['device'])
    
    return tensor, original

def predict(model, image_tensor):
    """Make prediction on image."""
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(1)
    
    return predicted.item(), probabilities.cpu().numpy()[0]

def create_confidence_plot(probabilities):
    """Create interactive bar chart of confidence scores."""
    fig = go.Figure(data=[
        go.Bar(
            x=probabilities * 100,
            y=DEFECT_CLASSES,
            orientation='h',
            marker=dict(
                color=[DEFECT_COLORS[cls] for cls in DEFECT_CLASSES],
                line=dict(color='rgba(0,0,0,0.3)', width=1)
            ),
            text=[f'{p*100:.1f}%' for p in probabilities],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='Confidence Scores for Each Defect Type',
        xaxis_title='Confidence (%)',
        yaxis_title='Defect Class',
        height=400,
        showlegend=False,
        xaxis=dict(range=[0, 100])
    )
    
    return fig

def main():
    # Header
    st.title("Steel Defect Detection System")
    st.markdown("**Automated Quality Control for Hot-Rolled Steel Strips**")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("System Information")
        
        with st.spinner("Loading model..."):
            model, checkpoint = load_model()
        
        st.success("Model loaded successfully!")
        
        st.metric("Model", "ConvNeXt V2 Tiny")
        st.metric("Training Accuracy", f"{checkpoint.get('val_acc', 0):.2f}%")
        st.metric("Epochs Trained", checkpoint.get('epoch', 0) + 1)
        
        st.markdown("---")
        
        st.header("Settings")
        show_all_scores = st.checkbox("Show All Confidence Scores", value=True)
        confidence_threshold = st.slider("Confidence Threshold (%)", 0, 100, 70)
        
        st.markdown("---")
        
        st.header("Defect Types")
        for defect in DEFECT_CLASSES:
            with st.expander(f"{defect}"):
                st.write(DEFECT_DESCRIPTIONS[defect])
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["Single Image", "Batch Processing", "Model Performance"])
    
    # TAB 1: SINGLE IMAGE PREDICTION
    with tab1:
        st.header("Upload Steel Surface Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a steel surface image for defect detection"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
                
                # Image info
                img_array = np.array(image)
                st.caption(f"Size: {img_array.shape[1]}x{img_array.shape[0]} pixels")
            
            with st.spinner("Analyzing image..."):
                image_tensor, original_image = preprocess_image(image)
                predicted_class, probabilities = predict(model, image_tensor)
                
                predicted_defect = DEFECT_CLASSES[predicted_class]
                confidence = probabilities[predicted_class] * 100
            
            with col2:
                st.subheader("Prediction Result")
                
                # Big result card
                result_card = f"""
                <div style='background-color: {DEFECT_COLORS[predicted_defect]}20; 
                            padding: 30px; 
                            border-radius: 10px; 
                            border-left: 5px solid {DEFECT_COLORS[predicted_defect]};
                            margin: 20px 0;'>
                    <h2 style='color: {DEFECT_COLORS[predicted_defect]}; margin: 0;'>{predicted_defect}</h2>
                    <h1 style='margin: 10px 0;'>{confidence:.1f}%</h1>
                    <p style='margin: 0;'>{DEFECT_DESCRIPTIONS[predicted_defect]}</p>
                </div>
                """
                st.markdown(result_card, unsafe_allow_html=True)
                
                # Status indicator
                if confidence >= confidence_threshold:
                    st.success(f"High Confidence (>={confidence_threshold}%)")
                else:
                    st.warning(f"Low Confidence (<{confidence_threshold}%) - Manual review recommended")
            
            st.markdown("---")
            
            # Detailed metrics
            if show_all_scores:
                st.subheader("Detailed Confidence Analysis")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig = create_confidence_plot(probabilities)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.write("**Top 3 Predictions:**")
                    top3_idx = np.argsort(probabilities)[-3:][::-1]
                    for rank, idx in enumerate(top3_idx, 1):
                        st.metric(
                            f"{rank}. {DEFECT_CLASSES[idx]}",
                            f"{probabilities[idx]*100:.2f}%"
                        )
            
            # Action buttons
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Download Result"):
                    st.info("Result download feature - Coming soon!")
            
            with col2:
                if st.button("Export Report"):
                    st.info("Report export feature - Coming soon!")
            
            with col3:
                if st.button("Analyze Another"):
                    st.rerun()
    
    # TAB 2: BATCH PROCESSING
    with tab2:
        st.header("Batch Image Processing")
        st.write("Upload multiple images for automated defect detection")
        
        uploaded_files = st.file_uploader(
            "Choose images...",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            accept_multiple_files=True,
            help="Upload multiple steel surface images"
        )
        
        if uploaded_files:
            st.info(f"**{len(uploaded_files)} images uploaded**")
            
            if st.button("Process All Images", type="primary"):
                results = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {idx+1}/{len(uploaded_files)}: {file.name}")
                    
                    image = Image.open(file)
                    image_tensor, _ = preprocess_image(image)
                    predicted_class, probabilities = predict(model, image_tensor)
                    
                    conf = probabilities[predicted_class] * 100
                    status = 'Pass' if conf >= confidence_threshold else 'Review'
                    
                    results.append({
                        'Image': file.name,
                        'Detected Defect': DEFECT_CLASSES[predicted_class],
                        'Confidence': f"{conf:.2f}%",
                        'Status': status
                    })
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                status_text.text("Processing complete!")
                
                st.markdown("---")
                st.subheader("Batch Results")
                st.dataframe(results, use_container_width=True)
                
                # Summary statistics
                st.markdown("---")
                st.subheader("Summary Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                defect_counts = {}
                for result in results:
                    defect = result['Detected Defect']
                    defect_counts[defect] = defect_counts.get(defect, 0) + 1
                
                with col1:
                    st.metric("Total Images", len(results))
                
                with col2:
                    pass_count = sum(1 for r in results if 'Pass' in r['Status'])
                    st.metric("Pass Rate", f"{pass_count/len(results)*100:.1f}%")
                
                with col3:
                    review_count = len(results) - pass_count
                    st.metric("Need Review", review_count)
                
                with col4:
                    if defect_counts:
                        most_common = max(defect_counts, key=defect_counts.get)
                        st.metric("Most Common", most_common)
                
                # Defect distribution chart
                if defect_counts:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_pie = px.pie(
                            values=list(defect_counts.values()),
                            names=list(defect_counts.keys()),
                            title='Defect Distribution',
                            color_discrete_map=DEFECT_COLORS
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with col2:
                        fig_bar = go.Figure(data=[
                            go.Bar(
                                x=list(defect_counts.keys()),
                                y=list(defect_counts.values()),
                                marker_color=[DEFECT_COLORS[d] for d in defect_counts.keys()]
                            )
                        ])
                        fig_bar.update_layout(
                            title='Defect Count by Type',
                            xaxis_title='Defect Type',
                            yaxis_title='Count'
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)
    
    # TAB 3: MODEL PERFORMANCE
    with tab3:
        st.header("Model Performance Metrics")
        
        summary_path = RESULTS_DIR / 'evaluation_summary.json'
        
        if summary_path.exists():
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Confusion Matrix")
                cm_path = RESULTS_DIR / 'confusion_matrix.png'
                if cm_path.exists():
                    st.image(str(cm_path), use_container_width=True)
                else:
                    st.warning("Confusion matrix not found")
            
            with col2:
                st.subheader("Per-Class Metrics")
                metrics_path = RESULTS_DIR / 'per_class_metrics.png'
                if metrics_path.exists():
                    st.image(str(metrics_path), use_container_width=True)
                else:
                    st.warning("Metrics chart not found")
            
            st.markdown("---")
            
            st.subheader("Classification Report")
            report_path = RESULTS_DIR / 'classification_report.txt'
            if report_path.exists():
                with open(report_path, 'r') as f:
                    st.code(f.read(), language='text')
            else:
                st.warning("Classification report not found")
        else:
            st.warning("No evaluation results found. Run `python model_evaluation.py` first.")
            
            if st.button("Refresh"):
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p><strong>Steel Defect Detection System</strong> | Powered by ConvNeXt V2 | Tata Steel Internship Project</p>
            <p>Developed with PyTorch, Streamlit, and Albumentations</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()