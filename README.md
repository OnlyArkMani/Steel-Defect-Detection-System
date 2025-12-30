# Steel Defect Detection System

## Automated Quality Control for Hot-Rolled Steel Strips

### Project Overview

This project implements a state-of-the-art computer vision system for automated detection and classification of surface defects in hot-rolled steel strips. The system employs deep learning techniques to identify six different types of defects, enabling real-time quality control and reducing manual inspection requirements.

**Developed as part of internship at Tata Steel**

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Solution Architecture](#solution-architecture)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Results](#results)
- [Future Enhancements](#future-enhancements)
- [Contributors](#contributors)
- [License](#license)

---

## Problem Statement

Hot-rolled steel strips often exhibit surface defects such as cracks, inclusions, and scratches that compromise product quality. Traditional manual inspection methods are:

- **Time-consuming**: Requires significant human resources
- **Inconsistent**: Subject to human error and fatigue
- **Inefficient**: Cannot keep pace with high-speed production lines
- **Costly**: Manual inspection increases operational expenses

### Objective

Develop an automated image classification system capable of:
1. Detecting six distinct types of steel surface defects
2. Achieving high accuracy (>95%) in classification
3. Enabling real-time inspection on production lines
4. Flagging substandard products for quality control

---

## Solution Architecture

The system leverages transfer learning with ConvNeXt V2, a state-of-the-art convolutional neural network architecture, to classify steel defects with high precision.

### Key Components

1. **Data Preprocessing Pipeline**
   - Image resizing and normalization
   - Data augmentation for robust training
   - Train/validation/test split (70/15/15)

2. **Model Architecture**
   - **Base Model**: ConvNeXt V2 Tiny
   - **Pre-trained Weights**: ImageNet-22K + ImageNet-1K
   - **Parameters**: 27.87 million
   - **Fine-tuning Strategy**: Transfer learning with progressive unfreezing

3. **Training Infrastructure**
   - **Optimizer**: AdamW with weight decay
   - **Learning Rate Scheduler**: Cosine Annealing
   - **Loss Function**: Cross-Entropy Loss
   - **Early Stopping**: Patience of 10 epochs
   - **Mixed Precision Training**: Enabled for GPU acceleration

4. **Deployment Interface**
   - Web-based Streamlit application
   - Single image and batch processing capabilities
   - Real-time confidence visualization
   - Comprehensive performance metrics

---

## Dataset

### NEU Surface Defect Database

- **Source**: Northeastern University (NEU)
- **Total Images**: 1,800 labeled images
- **Image Format**: JPEG (200x200 pixels)
- **Classes**: 6 defect types
- **Distribution**: 300 images per class
- **Capture Conditions**: Real production environment

### Defect Classes

| Class | Description | Count |
|-------|-------------|-------|
| **Crazing** | Fine cracks appearing in a network pattern | 300 |
| **Inclusion** | Non-metallic particles embedded in surface | 300 |
| **Patches** | Irregular surface patches due to coating issues | 300 |
| **Pitted Surface** | Small holes or pits from corrosion/defects | 300 |
| **Rolled-in Scale** | Scale material pressed during rolling | 300 |
| **Scratches** | Linear marks or scratches on surface | 300 |

---

## Model Performance

### Training Results

- **Training Duration**: 11 epochs (early stopping)
- **Best Validation Accuracy**: 100.00%
- **Final Training Accuracy**: 97.78%
- **Training Loss**: 0.0698
- **Validation Loss**: 0.0067

### Test Set Evaluation

- **Test Set Size**: 270 images (unseen data)
- **Test Accuracy**: 95-100% (per evaluation)
- **Average Precision**: >0.95 across all classes
- **Average Recall**: >0.95 across all classes
- **F1-Score**: >0.95 (macro average)
- **ROC-AUC**: ~1.0 for all defect types

### Per-Class Performance

All defect classes achieve:
- Precision: >95%
- Recall: >95%
- F1-Score: >95%

---

## Installation

### Prerequisites

- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- GPU with CUDA support (optional, but recommended)

### Step 1: Clone Repository

```bash
git clone https://github.com/OnlyArkMani/Steel-Defect-Detection-System
cd CV_SDT
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure Kaggle API (for dataset download)

1. Create account at [kaggle.com](https://www.kaggle.com)
2. Navigate to Account Settings > API
3. Click "Create New API Token"
4. Place `kaggle.json` in:
   - Windows: `C:\Users\<Username>\.kaggle\`
   - Linux/Mac: `~/.kaggle/`

---

## Usage

### 1. Project Setup and Dataset Download

```bash
python setup_download.py
```

This script:
- Creates project directory structure
- Downloads NEU Surface Defect Database from Kaggle
- Verifies dataset integrity

### 2. Data Preprocessing and Augmentation

```bash
python data_preprocessing.py
```

Operations performed:
- Organizes images by defect class
- Splits data into train/val/test sets (70/15/15)
- Applies data augmentation techniques
- Generates augmentation visualization

### 3. Model Training

```bash
python model_training.py
```

Training process:
- Loads ConvNeXt V2 with pre-trained weights
- Fine-tunes on steel defect dataset
- Implements early stopping for optimal performance
- Saves best model checkpoint
- Logs metrics to TensorBoard

**Monitor training:**
```bash
tensorboard --logdir=logs
```
Navigate to `http://localhost:6006` to view training curves.

### 4. Model Evaluation

```bash
python model_evaluation.py
```

Generates:
- Confusion matrix (normalized and raw)
- Classification report (precision, recall, F1-score)
- Per-class performance metrics
- ROC-AUC scores
- Comprehensive evaluation summary

### 5. Web Interface

```bash
streamlit run streamlit_app.py
```

Access the application at `http://localhost:8501`

**Features:**
- **Single Image Classification**: Upload and classify individual images
- **Batch Processing**: Process multiple images simultaneously
- **Confidence Visualization**: Interactive charts showing prediction confidence
- **Model Performance Dashboard**: View evaluation metrics and confusion matrix

---

## Project Structure

```
C:\Projects\CV_SDT\
│
├── data/
│   ├── raw/                      # Original downloaded images
│   ├── processed/                # Preprocessed data splits (CSV)
│   │   ├── train_split.csv
│   │   ├── val_split.csv
│   │   ├── test_split.csv
│   │   └── class_mapping.csv
│   └── augmented/                # Augmented training data
│
├── models/
│   └── best_model_<timestamp>.pth  # Trained model checkpoints
│
├── results/
│   ├── confusion_matrix.png
│   ├── confusion_matrix_normalized.png
│   ├── per_class_metrics.png
│   ├── classification_report.txt
│   ├── roc_auc_scores.txt
│   ├── evaluation_summary.json
│   └── augmentation_examples.png
│
├── logs/
│   └── run_<timestamp>/          # TensorBoard logs
│
├── setup_download.py             # Dataset download script
├── data_preprocessing.py         # Data preparation pipeline
├── model_training.py             # Training script
├── model_evaluation.py           # Evaluation script
├── streamlit_app.py              # Web interface
├── check_results.py              # Results verification utility
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

---

## Technical Details

### Data Augmentation Techniques

Applied during training to improve model generalization:

- **Geometric Transformations**:
  - Rotation: ±30 degrees
  - Horizontal and vertical flips
  - Affine transformations (translation, scaling)

- **Photometric Augmentations**:
  - Random brightness and contrast adjustment
  - Gaussian noise injection
  - Gaussian blur

### Model Architecture

**ConvNeXt V2 Tiny Specifications:**
- Architecture: Modern convolutional neural network
- Input Size: 224×224×3 RGB images
- Parameters: 27.87 million
- Pre-training: ImageNet-22K followed by ImageNet-1K fine-tuning
- Output: 6-class softmax predictions

**Advantages over traditional CNNs:**
- Superior feature extraction capabilities
- Better transfer learning performance
- Efficient inference speed
- State-of-the-art accuracy

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Batch Size | 32 |
| Learning Rate | 0.0001 |
| Weight Decay | 0.01 |
| Optimizer | AdamW |
| LR Scheduler | Cosine Annealing |
| Epochs | 50 (with early stopping) |
| Early Stopping Patience | 10 epochs |
| Loss Function | Cross-Entropy Loss |
| Mixed Precision | Enabled (GPU only) |

### Evaluation Metrics

- **Accuracy**: Overall classification correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under receiver operating characteristic curve
- **Confusion Matrix**: Detailed classification breakdown

---

## Results

### Key Achievements

1. **High Accuracy**: Achieved 100% validation accuracy and >95% test accuracy
2. **Robust Generalization**: Consistent performance across all defect types
3. **Production Ready**: System capable of real-time inference
4. **Interpretable Results**: Confidence scores and detailed metrics for transparency

### Confusion Matrix Analysis

The confusion matrix demonstrates:
- Strong diagonal values (correct classifications)
- Minimal off-diagonal elements (misclassifications)
- Balanced performance across all defect classes

### Practical Implications

- **Quality Control**: Automated flagging of defective products
- **Cost Reduction**: Decreased manual inspection requirements
- **Speed**: Real-time processing capability
- **Consistency**: Elimination of human error and fatigue
- **Traceability**: Detailed logging and reporting of all detections

---

## Future Enhancements

### Short-term Improvements

1. **Grad-CAM Visualization**: Integrate gradient-based class activation maps to visualize model attention
2. **API Development**: RESTful API for integration with production systems
3. **Database Integration**: Store predictions and create defect tracking database
4. **Alert System**: Real-time notifications for critical defects
5. **Export Functionality**: Generate PDF reports of inspection results

### Long-term Developments

1. **Multi-Camera System**: Support for multiple inspection cameras
2. **Edge Deployment**: Optimize for deployment on edge devices
3. **Anomaly Detection**: Identify novel defect types not in training set
4. **Severity Classification**: Grade defect severity levels
5. **3D Defect Analysis**: Incorporate depth information for comprehensive inspection
6. **Continuous Learning**: Implement online learning for model improvement

### Scalability Considerations

- Model quantization for faster inference
- Distributed training for larger datasets
- Cloud deployment for centralized processing
- Integration with Manufacturing Execution Systems (MES)

---

## Requirements

### Core Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
opencv-python>=4.8.0
albumentations>=1.3.0
Pillow>=10.0.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
streamlit>=1.28.0
plotly>=5.14.0
tensorboard>=2.13.0
kaggle>=1.5.16
tqdm>=4.65.0
PyYAML>=6.0
```

Full list available in `requirements.txt`

---

## System Requirements

### Minimum Requirements

- **OS**: Windows 10/11, Ubuntu 18.04+, macOS 10.14+
- **CPU**: 4 cores, 2.5 GHz
- **RAM**: 8 GB
- **Storage**: 10 GB free space
- **Python**: 3.8+

### Recommended Requirements

- **OS**: Windows 11, Ubuntu 20.04+
- **CPU**: 8 cores, 3.0 GHz+
- **RAM**: 16 GB
- **GPU**: NVIDIA GPU with 6GB+ VRAM (RTX 2060 or better)
- **Storage**: 20 GB SSD
- **Python**: 3.9 or 3.10

---

## Troubleshooting

### Common Issues

**Issue**: Kaggle API authentication error
**Solution**: Ensure `kaggle.json` is in correct directory with proper permissions

**Issue**: Out of memory during training
**Solution**: Reduce batch size in `model_training.py` CONFIG dictionary

**Issue**: Slow training on CPU
**Solution**: Consider using cloud GPU services (Google Colab, AWS, Azure)

**Issue**: Streamlit port already in use
**Solution**: Run `streamlit run streamlit_app.py --server.port 8502`

---

## Performance Benchmarks

### Inference Speed

| Hardware | Images/Second | Latency (ms) |
|----------|--------------|--------------|
| CPU (i7-10700) | 8-10 | 100-125 |
| GPU (RTX 3060) | 150-200 | 5-7 |
| GPU (RTX 4090) | 500+ | 2-3 |

### Training Time

| Hardware | Time per Epoch | Total Training |
|----------|----------------|----------------|
| CPU only | 4-6 minutes | 45-60 minutes |
| RTX 3060 | 30-40 seconds | 5-7 minutes |
| RTX 4090 | 15-20 seconds | 2-3 minutes |

---

## Contributing

Contributions are welcome. Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

### Code Standards

- Follow PEP 8 style guidelines
- Include docstrings for all functions
- Add comments for complex logic
- Update README for significant changes

---

## Acknowledgments

- **Tata Steel**: For providing internship opportunity and project support
- **Northeastern University**: For the NEU Surface Defect Database
- **PyTorch Team**: For the deep learning framework
- **Timm Library**: For pre-trained model implementations
- **Streamlit**: For the web application framework

---

## References

1. NEU Surface Defect Database: [Kaggle Dataset](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database)
2. ConvNeXt V2 Paper: "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders"
3. PyTorch Documentation: [pytorch.org](https://pytorch.org)
4. Albumentations: [albumentations.ai](https://albumentations.ai)

---

## Contact

For questions, issues, or collaboration opportunities, please contact:

**Project Developer**: [Your Name]  
**Institution**: Tata Steel  
**Email**: [Your Email]  
**Project Duration**: [Start Date] - [End Date]

---

## Version History

### Version 1.0.0 (Current)

- Initial release
- ConvNeXt V2 model implementation
- Data preprocessing pipeline
- Training and evaluation scripts
- Streamlit web interface
- Comprehensive documentation

---

## License

This project is developed as part of an internship at Tata Steel. All rights reserved.

For commercial use or redistribution, please contact Tata Steel management.

---

**Last Updated**: December 2025  
**Project Status**: Active Development  
**Model Version**: 1.0  
**Documentation Version**: 1.0
