# app.py

import streamlit as st
from streamlit_lottie import st_lottie
import requests
import json
import torch
import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from PIL import Image
import os
import time
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import base64

# Set page configuration
st.set_page_config(
    page_title="AI Flood Detection System",
    page_icon="🌊",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .header-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .logo-container {
        display: flex;
        align-items: center;
    }
    .logo {
        height: 60px;
        margin-right: 20px;
    }
    .title-container {
        flex-grow: 1;
        text-align: center;
    }
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: #e8f4f8;
        font-size: 1.2rem;
        margin-bottom: 0;
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .upload-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #667eea;
        margin: 1rem 0;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .processing-spinner {
        text-align: center;
        padding: 2rem;
        background: #f8f9fa;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .results-section {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .stSelectbox > div > div {
        background: white;
        border-radius: 5px;
    }
    
    .footer {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-top: 3rem;
        text-align: center;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

def load_lottie_url(url: str):
    """Load Lottie animation from URL with error handling."""
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

def build_model(in_channels):
    """Initializes the UNet model architecture with a flexible number of input channels."""
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=in_channels,
        classes=1,
    )
    return model

@st.cache_resource
def load_model(in_channels, model_type):
    """Load the trained PyTorch model with enhanced error handling."""
    model_paths = {
        'S1': "best_model_s1_only.pth",
        'S2': "best_model_s2_only.pth"
    }
    
    if model_type not in model_paths:
        st.error("❌ Invalid model type specified.")
        return None, torch.device("cpu")

    model_path = model_paths[model_type]
    model = build_model(in_channels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            st.success(f"✅ {model_type} model loaded successfully on {device}")
            return model, device
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            model.to(device)
            model.eval()
            st.warning(f"⚠️ Using random weights due to loading error")
            return model, device
    else:
        st.warning(f"⚠️ Model file not found. Using random weights.")
        model.to(device)
        model.eval()
        return model, device

def process_and_predict(uploaded_file, model, device, model_type):
    """Enhanced processing with better error handling and progress tracking."""
    file_path = None
    try:
        # Show processing steps
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("📁 Saving uploaded file...")
        progress_bar.progress(20)
        
        # Save uploaded file temporarily
        file_path = os.path.join("/tmp", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        status_text.text("🔍 Reading satellite data...")
        progress_bar.progress(40)
        
        with rasterio.open(file_path) as src:
            original_data = src.read().transpose(1, 2, 0)
            num_bands = original_data.shape[-1]
            profile = src.profile

        status_text.text("⚙️ Preprocessing data...")
        progress_bar.progress(60)
        
        # Enhanced preprocessing with validation
        if model_type == 'S1':
            if num_bands < 2:
                raise ValueError(f"S1 TIFF must contain at least 2 bands (VV and VH). Found {num_bands}.")
            
            sar_data = original_data[:, :, :2]
            sar_processed = np.nan_to_num(sar_data, nan=0.0, posinf=0.0, neginf=0.0)
            sar_processed = np.clip(sar_processed, -40.0, 0.0)
            input_processed = (sar_processed + 40) / 40.0
            original_for_viz = original_data[:, :, 0]

        elif model_type == 'S2':
            optical_data = original_data
            optical_processed = np.nan_to_num(optical_data, nan=0.0, posinf=0.0, neginf=0.0)
            optical_processed = np.clip(optical_processed, 0, 3000)
            input_processed = optical_processed / 3000.0
            original_for_viz = original_data

        status_text.text("🧠 Running AI inference...")
        progress_bar.progress(80)
        
        # Model inference
        input_tensor = torch.from_numpy(input_processed).permute(2, 0, 1).float().unsqueeze(0).to(device)
        
        model.eval()
        with torch.no_grad():
            prediction = model(input_tensor)

        # Apply sigmoid and threshold
        probability_mask = torch.sigmoid(prediction).squeeze().cpu().numpy()
        predicted_mask_binary = (probability_mask > 0.5).astype(np.uint8)

        status_text.text("✅ Processing complete!")
        progress_bar.progress(100)
        
        # Clear progress indicators
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()

        return original_for_viz, predicted_mask_binary, probability_mask, num_bands

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
        return None, None, None, None
    finally:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)

def create_enhanced_visualization(original_data, predicted_mask, probability_mask, model_type):
    """Create enhanced visualizations with better styling."""
    
    # Create subplot with better layout
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], hspace=0.3, wspace=0.3)
    
    # Style configuration
    plt.style.use('seaborn-v0_8')
    
    # Original image
    ax1 = fig.add_subplot(gs[0, 0])
    if model_type == 'S1':
        display_vv = np.clip(original_data, -40, 0)
        display_vv_normalized = ((display_vv - (-40)) / (0 - (-40)) * 255).astype(np.uint8)
        ax1.imshow(display_vv_normalized, cmap='gray')
        ax1.set_title('Original SAR (VV Band)', fontsize=14, fontweight='bold')
    elif model_type == 'S2':
        try:
            if original_data.shape[-1] >= 4:
                s2_rgb = original_data[:, :, [3, 2, 1]]
                s2_rgb = np.clip(s2_rgb, 0, 2000)
                s2_rgb_normalized = (s2_rgb / 2000.0 * 255).astype(np.uint8)
                ax1.imshow(s2_rgb_normalized)
                ax1.set_title('Original S2 (True Color)', fontsize=14, fontweight='bold')
            else:
                ax1.text(0.5, 0.5, 'Insufficient Bands', ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('S2 Image', fontsize=14, fontweight='bold')
        except Exception:
            ax1.text(0.5, 0.5, 'Error Loading S2', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('S2 Image', fontsize=14, fontweight='bold')
    
    ax1.axis('off')
    
    # Probability mask
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(probability_mask, cmap='Blues', vmin=0, vmax=1)
    ax2.set_title('Flood Intensity Map', fontsize=14, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, shrink=0.8, label='Intensity')
    
    # Binary mask
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(predicted_mask, cmap='Reds', interpolation='nearest')
    ax3.set_title('Detected Flood Mask', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # Statistics subplot
    ax4 = fig.add_subplot(gs[1, :])
    
    # Calculate statistics
    flood_pixels = np.sum(predicted_mask)
    total_pixels = predicted_mask.size
    flood_percentage = (flood_pixels / total_pixels) * 100
    
    # Create bar chart
    categories = ['Non-Flood', 'Flood']
    values = [total_pixels - flood_pixels, flood_pixels]
    colors = ['#2E86AB', '#F24236']
    
    bars = ax4.bar(categories, values, color=colors, alpha=0.8)
    ax4.set_title('Pixel Classification Statistics', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Number of Pixels')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + total_pixels*0.01,
                f'{value:,}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_interactive_overlay(original_data, predicted_mask, probability_mask, model_type):
    """Create interactive overlay with Plotly, showing true color or grayscale imagery."""
    
    # Prepare base image
    if model_type == 'S1':
        # For S1, we'll create a pseudo-color image using available bands
        if original_data.ndim == 2:
            # If only one band is available (e.g., VV)
            vv_band = np.clip(original_data, -40, 0)
            base_img = np.stack([vv_band, vv_band, vv_band], axis=-1)
        else:
            # If two bands are available (VV and VH)
            vv_band = np.clip(original_data[:, :, 0], -40, 0)
            vh_band = np.clip(original_data[:, :, 1], -40, 0)
        
        # Normalize to 0-255 range
        r = ((vv_band - (-40)) / (0 - (-40)) * 255).astype(np.uint8)
        g = ((vh_band - (-40)) / (0 - (-40)) * 255).astype(np.uint8) if original_data.ndim > 2 else r
        b = ((r + g) / 2).astype(np.uint8)
        
        base_img = np.stack([r, g, b], axis=-1)
        title = "SAR Pseudo-Color with Flood Overlay"
    elif model_type == 'S2':
        if original_data.ndim == 2:
            # If only one band is available, use it for grayscale
            base_img = np.stack([original_data, original_data, original_data], axis=-1)
        elif original_data.shape[-1] >= 3:
            # Use first three bands for RGB representation
            s2_rgb = original_data[:, :, :3]
            s2_rgb = np.clip(s2_rgb, 0, 3000)
            base_img = (s2_rgb / 3000.0 * 255).astype(np.uint8)
        else:
            # Fallback to grayscale if less than 3 bands
            base_img = np.stack([original_data[:,:,0], original_data[:,:,0], original_data[:,:,0]], axis=-1)
        title = "S2 Image with Flood Overlay"
    
    # Create figure
    fig = go.Figure()
    
    # Add base image
    fig.add_trace(go.Image(
        z=base_img,
        name='Base Image'
    ))
    
    # Add flood overlay
    flood_overlay = np.where(predicted_mask == 1, probability_mask, np.nan)
    fig.add_trace(go.Heatmap(
        z=flood_overlay,
        colorscale='Blues',
        opacity=0.6,
        showscale=True,
        colorbar=dict(title="Flood Probability"),
        name='Flood Areas'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="X Coordinate",
        yaxis_title="Y Coordinate",
        height=600,
        showlegend=False
    )
    
    return fig

def create_enhanced_metrics(predicted_mask, probability_mask):
    """Create enhanced metrics display."""
    
    # Calculate comprehensive statistics
    total_pixels = predicted_mask.size
    flood_pixels = np.sum(predicted_mask)
    flood_percentage = (flood_pixels / total_pixels) * 100
    
    # High confidence flood areas (probability > 0.8)
    high_conf_flood = np.sum(probability_mask > 0.8)
    high_conf_percentage = (high_conf_flood / total_pixels) * 100
    
    # Average flood probability
    avg_flood_prob = np.mean(probability_mask[predicted_mask == 1]) if flood_pixels > 0 else 0
    
    # Create metrics display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🌊 Flood Coverage</h3>
            <h2>{flood_percentage:.2f}%</h2>
            <p>{flood_pixels:,} pixels</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎯 High Confidence</h3>
            <h2>{high_conf_percentage:.2f}%</h2>
            <p>Probability > 80%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Avg. Probability</h3>
            <h2>{avg_flood_prob:.3f}</h2>
            <p>Flood areas only</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📐 Resolution</h3>
            <h2>{predicted_mask.shape[0]}×{predicted_mask.shape[1]}</h2>
            <p>Total: {total_pixels:,} pixels</p>
        </div>
        """, unsafe_allow_html=True)

def create_sidebar_info():
    """Create informative sidebar."""
    st.sidebar.markdown("## 📊 Model Information")
    
    st.sidebar.markdown("""
    ### 🛰️ Supported Satellites
    - **Sentinel-1**: SAR imagery (VV, VH bands)
    - **Sentinel-2**: Optical imagery (Multi-spectral)
    
    ### 🧠 Model Architecture
    - **Base**: U-Net with ResNet34 encoder
    - **Training**: Deep learning on satellite imagery
    - **Output**: Flood probability maps
    
    ### 📈 Performance Metrics
    - Accuracy: Model-dependent
    - Resolution: Input-dependent
    - Processing: Real-time inference
    """)
    
    st.sidebar.markdown("## ⚙️ Settings")
    
    # Add threshold slider
    threshold = st.sidebar.slider(
        "Flood Detection Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Adjust the probability threshold for flood detection"
    )
    
    # Add visualization options
    show_probability = st.sidebar.checkbox("Show Probability Map", value=True)
    show_overlay = st.sidebar.checkbox("Show Interactive Overlay", value=True)
    
    return threshold, show_probability, show_overlay

def main():
    """Enhanced main function with improved UI/UX."""
    
    # Stylized header with integrated logos
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="border: 2px solid #667eea; padding: 10px; border-radius: 10px;">
                <img src="data:image/png;base64,{}" style="height: 100px;">
            </div>
            <div style="text-align: center; flex-grow: 1;">
                <h1 style="color: #4a4a4a; font-size: 2.5rem; margin-bottom: 0.5rem;">🌊 AI-Enabled Flood Detection System 🛰️</h1>
                <p style="color: #666; font-size: 1.2rem; margin-bottom: 0;">Advanced satellite-based flood detection using deep learning</p>
            </div>
            <div style="border: 2px solid #667eea; padding: 10px; border-radius: 10px;">
                <img src="data:image/png;base64,{}" style="height: 100px;">
            </div>
        </div>
    </div>
    """.format(
        get_base64_encoded_image("afrdi_logo.png"),
        get_base64_encoded_image("logo_dsa.png")
    ), unsafe_allow_html=True)

    # Sidebar configuration
    threshold, show_probability, show_overlay = create_sidebar_info()
    
    # Main content area (centralized)
    st.markdown("""
    <div style="display: flex; justify-content: center;">
        <div style="max-width: 800px; width: 100%;">
            <div class="feature-card">
                <h3>🚀 Advanced AI Technology</h3>
                <p>Our system uses state-of-the-art U-Net architecture with ResNet34 encoder, trained on extensive satellite imagery datasets for accurate flood detection.</p>
            </div>
            <div class="feature-card">
                <h3>🛰️ Multi-Satellite Support</h3>
                <p>Compatible with both Sentinel-1 SAR and Sentinel-2 optical imagery, providing flexible flood detection capabilities across different weather conditions.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # File upload section
    st.markdown("""
    <div class="upload-section">
        <h2 style="text-align: center; color: #667eea;">📤 Upload Satellite Imagery</h2>
        <p style="text-align: center; color: #666;">Drag and drop your TIFF files or click to browse</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🛰️ Sentinel-1 SAR")
        uploaded_s1_file = st.file_uploader(
            "Upload Sentinel-1 TIFF", 
            type="tif", 
            key="s1_uploader",
            help="Upload Sentinel-1 SAR imagery (VV and VH bands)"
        )
        if uploaded_s1_file:
            st.success("✅ Sentinel-1 file uploaded successfully!")
            st.info(f"📁 File: {uploaded_s1_file.name}")
            st.info(f"📏 Size: {uploaded_s1_file.size / 1024 / 1024:.2f} MB")
    
    with col2:
        st.markdown("### 🛰️ Sentinel-2 Optical")
        uploaded_s2_file = st.file_uploader(
            "Upload Sentinel-2 TIFF", 
            type="tif", 
            key="s2_uploader",
            help="Upload Sentinel-2 optical imagery (multi-spectral bands)"
        )
        if uploaded_s2_file:
            st.success("✅ Sentinel-2 file uploaded successfully!")
            st.info(f"📁 File: {uploaded_s2_file.name}")
            st.info(f"📏 Size: {uploaded_s2_file.size / 1024 / 1024:.2f} MB")
    
    # Determine processing parameters
    uploaded_file = uploaded_s1_file or uploaded_s2_file
    model_type = 'S1' if uploaded_s1_file else 'S2' if uploaded_s2_file else None
    
    if uploaded_s1_file and uploaded_s2_file:
        st.warning("⚠️ Please upload either Sentinel-1 OR Sentinel-2 file, not both.")
        uploaded_file = None
        model_type = None
    
    # Processing section
    if uploaded_file and model_type:
        
        st.markdown("""
        <div class="processing-spinner">
            <h3>🔄 Processing Your Satellite Image</h3>
            <p>Our AI is analyzing the imagery for flood detection...</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Determine input channels
        determined_input_channels = None
        try:
            file_path_temp = os.path.join("/tmp", uploaded_file.name)
            with open(file_path_temp, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with rasterio.open(file_path_temp) as src:
                determined_input_channels = src.count
            
            st.info(f"📊 Detected {determined_input_channels} spectral bands")
            os.remove(file_path_temp)
            
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            determined_input_channels = None
        
        if determined_input_channels is not None:
            # Load model
            model, device = load_model(determined_input_channels, model_type)
            
            if model:
                # Process and predict
                original_data, predicted_mask, probability_mask, _ = process_and_predict(
                    uploaded_file, model, device, model_type
                )
                
                if all(x is not None for x in [original_data, predicted_mask, probability_mask]):
                    
                    # Apply custom threshold
                    if threshold != 0.5:
                        predicted_mask = (probability_mask > threshold).astype(np.uint8)
                    
                    # Results section
                    st.markdown("""
                    <div class="results-section">
                        <h2 style="color: #667eea;">🎉 Flood Detection Results</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Enhanced metrics
                    create_enhanced_metrics(predicted_mask, probability_mask)
                    
                    # Visualizations
                    tab1, tab2, tab3 = st.tabs(["📊 Detailed Analysis", "🗺️ Interactive Map", "📈 Statistics"])
                    
                    with tab1:
                        st.subheader("🔍 Comprehensive Flood Analysis")
                        if show_probability:
                            fig = create_enhanced_visualization(
                                original_data, predicted_mask, probability_mask, model_type
                            )
                            st.pyplot(fig)
                        else:
                            # Simple visualization
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                            
                            if model_type == 'S1':
                                display_vv = np.clip(original_data, -40, 0)
                                display_vv_normalized = ((display_vv - (-40)) / (0 - (-40)) * 255).astype(np.uint8)
                                ax1.imshow(display_vv_normalized, cmap='gray')
                                ax1.set_title('Original SAR')
                            else:
                                ax1.imshow(original_data[:, :, 0], cmap='gray')
                                ax1.set_title('Original S2')
                            
                            ax2.imshow(predicted_mask, cmap='Reds')
                            ax2.set_title('Flood Mask')
                            
                            ax1.axis('off')
                            ax2.axis('off')
                            st.pyplot(fig)
                    
                    with tab2:
                        st.subheader("🗺️ Interactive Flood Map")
                        if show_overlay:
                            try:
                                interactive_fig = create_interactive_overlay(
                                    original_data, predicted_mask, probability_mask, model_type
                                )
                                st.plotly_chart(interactive_fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error creating interactive map: {e}")
                        else:
                            st.info("Enable 'Show Interactive Overlay' in the sidebar to view this section.")
                    
                    with tab3:
                        st.subheader("📈 Detailed Statistics")
                        
                        # Create histogram of flood probabilities
                        fig_hist = px.histogram(
                            probability_mask.flatten(),
                            nbins=50,
                            title="Distribution of Flood Probabilities",
                            labels={'x': 'Probability', 'y': 'Frequency'},
                            color_discrete_sequence=['#667eea']
                        )
                        fig_hist.add_vline(x=threshold, line_dash="dash", line_color="red", 
                                         annotation_text=f"Threshold: {threshold}")
                        st.plotly_chart(fig_hist, use_container_width=True)
                        
                        # Summary statistics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("### 📊 Probability Statistics")
                            st.write(f"**Mean**: {np.mean(probability_mask):.4f}")
                            st.write(f"**Median**: {np.median(probability_mask):.4f}")
                            st.write(f"**Std Dev**: {np.std(probability_mask):.4f}")
                            st.write(f"**Min**: {np.min(probability_mask):.4f}")
                            st.write(f"**Max**: {np.max(probability_mask):.4f}")
                        
                        with col2:
                            st.markdown("### 🎯 Detection Summary")
                            total_pixels = predicted_mask.size
                            flood_pixels = np.sum(predicted_mask)
                            st.write(f"**Total Pixels**: {total_pixels:,}")
                            st.write(f"**Flood Pixels**: {flood_pixels:,}")
                            st.write(f"**Coverage**: {(flood_pixels/total_pixels)*100:.2f}%")
                            st.write(f"**Threshold Used**: {threshold}")
                    
                    # Success message
                    st.success(f"🎉 Flood detection complete! Found {np.sum(predicted_mask):,} flood pixels out of {predicted_mask.size:,} total pixels.")
                    
                    # Download section
                    st.markdown("### 💾 Download Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Create download button for binary mask
                        mask_pil = Image.fromarray(predicted_mask * 255)
                        st.download_button(
                            label="📥 Download Binary Mask",
                            data=mask_pil.tobytes(),
                            file_name=f"flood_mask_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                            mime="image/png"
                        )
                    
                    with col2:
                        # Create download button for probability map
                        prob_pil = Image.fromarray((probability_mask * 255).astype(np.uint8))
                        st.download_button(
                            label="📥 Download Probability Map",
                            data=prob_pil.tobytes(),
                            file_name=f"flood_probability_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                            mime="image/png"
                        )
                
                else:
                    st.error("❌ Failed to process the uploaded image. Please check the file format and try again.")
            else:
                st.error("❌ Failed to load the model. Please check model files and try again.")
        else:
            st.error("❌ Unable to determine the number of spectral bands in the uploaded file.")
    
    elif not uploaded_file:
        # Information section when no file is uploaded
        st.markdown("## 📋 How to Use This System")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h4>🛰️ Sentinel-1 SAR</h4>
                <ul>
                    <li>All-weather imaging capability</li>
                    <li>Requires VV and VH bands</li>
                    <li>Excellent for flood detection in cloudy conditions</li>
                    <li>Penetrates clouds and vegetation</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h4>🛰️ Sentinel-2 Optical</h4>
                <ul>
                    <li>High-resolution optical imagery</li>
                    <li>Multi-spectral bands (13 bands)</li>
                    <li>Clear visual interpretation</li>
                    <li>Best for clear weather conditions</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Usage instructions
        st.markdown("## 📖 Step-by-Step Guide")
        
        steps = [
            "📁 **Upload** your satellite imagery (TIFF format)",
            "⚙️ **Select** the appropriate satellite type (S1 or S2)",
            "🔧 **Adjust** detection threshold if needed (sidebar)",
            "🚀 **Wait** for AI processing to complete",
            "📊 **Review** results in multiple visualization tabs",
            "💾 **Download** flood maps and probability data"
        ]
        
        for i, step in enumerate(steps, 1):
            st.markdown(f"{i}. {step}")
        
        # Technical specifications
        st.markdown("## 🔧 Technical Specifications")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Model Architecture**
            - U-Net with ResNet34 encoder
            - Deep learning-based segmentation
            - Trained on satellite imagery
            """)
        
        with col2:
            st.markdown("""
            **Supported Formats**
            - TIFF files (.tif)
            - Multi-band satellite imagery
            - Geo-referenced data preferred
            """)
        
        with col3:
            st.markdown("""
            **Performance**
            - Real-time processing
            - High accuracy flood detection
            - Probability-based outputs
            """)
        
        # Sample data information
        st.markdown("## 📊 Data Access")
        st.info("""
        💡 **Tip**: Download sample Sentinel-1 and Sentinel-2 data from:
        - [Copernicus Open Access Hub](https9://scihub.copernicus.eu/)
        - [USGS Earth Explorer](https://earthexplorer.usgs.gov/)
        - [Google Earth Engine](https://earthengine.google.com/)
        """)
    
    # Footer section
    st.markdown("""
    <div class="footer">
        <h3>🌊 AI Flood Detection System</h3>
        <p>Powered by Deep Learning | Built by DSA Cohort 4 Team 2</p>
        <p>© 2025 - AFRDI  DSA Cohort 4</p>
        <p> Air Force Research and Development Institute | Defence Space Administration </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Additional information in expander
    with st.expander("ℹ️ Additional Information & Troubleshooting"):
        st.markdown("""
        ### 🔍 Troubleshooting Common Issues
        
        **File Upload Issues:**
        - Ensure TIFF files are not corrupted
        - Check file size (large files may take longer to process)
        - Verify the file contains the required spectral bands
        
        **Processing Errors:**
        - S1 files must contain at least 2 bands (VV, VH)
        - S2 files work best with all 13 bands
        - Ensure proper geo-referencing for best results
        
        **Model Performance:**
        - Results may vary based on image quality
        - Cloud coverage can affect S2 results
        - Urban areas may show different patterns
        
        ### 📚 References & Resources
        - [Sentinel-1 User Guide](https://sentinels.copernicus.eu/)
        - [Sentinel-2 User Guide](https://sentinels.copernicus.eu/)
        - [Flood Detection Techniques](https://www.mdpi.com/2072-4292/12/24/4082)
        
        ### 🛠️ Technical Support
        For technical issues or questions about the model:
        - Check input data format and specifications
        - Verify model files are properly loaded
        - Review processing logs for detailed error messages
        """)
    
    # Performance monitoring (optional)
    if st.checkbox("Show Performance Metrics", value=False):
        st.markdown("### 📊 System Performance")
        
        # Device information
        device_info = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.write(f"**Processing Device**: {device_info}")
        
        if torch.cuda.is_available():
            st.write(f"**GPU Memory**: {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB")
            st.write(f"**GPU Name**: {torch.cuda.get_device_name(0)}")
        
        # Memory usage
        import psutil
        memory_usage = psutil.virtual_memory()
        st.write(f"**System Memory**: {memory_usage.percent}% used")
        st.write(f"**Available Memory**: {memory_usage.available // 1024**2} MB")

def get_base64_encoded_image(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

# Application entry point
if __name__ == "__main__":
    main()
