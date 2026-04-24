# Attention Classification using Eye Tracking and Machine Learning

A hybrid CNN-LSTM model for real-time attention classification through eye gaze analysis, achieving 97.65% test accuracy. This project combines synthetic and real eye-tracking data to create a robust attention monitoring system.

## 📋 Overview

This research implements a deep learning approach to classify attention levels using eye movement patterns. The system processes sequential eye images to distinguish between attentive and non-attentive states, with applications in educational monitoring, clinical assessment (particularly ADHD), and productivity tracking.

## 🚀 Key Features

- **Hybrid Architecture**: CNN-LSTM model capturing both spatial and temporal features of eye movements
- **Multi-dataset Training**: Combines synthetic (UnityEyes) and real-world (MPIIGaze) data
- **Real-time Implementation**: Webcam-based live attention monitoring
- **Comprehensive Analysis**: Detailed performance metrics and threshold optimization
- **Data Logging**: CSV output for continuous behavioral analysis

## 🛠️ Technology Stack

- **Deep Learning**: TensorFlow, Keras
- **Computer Vision**: OpenCV
- **Data Processing**: NumPy, Pandas, SciPy
- **Model Evaluation**: Scikit-learn
- **Programming Language**: Python

## 📊 Model Performance

- **Test Accuracy**: 97.65%
- **Validation Accuracy**: ~85%
- **Optimal Threshold**: 0.50 (84.6% accuracy)
- **Training**: Stable convergence with minimal overfitting

## 📁 Dataset Information

### [UnityEyes (Synthetic)](https://kaggle.com/datasets/912502af5f374eda590b7965fbcc0e3a27081f5e6b08fed685c6aa6cb6cec457)

- 1 million+ synthetic eye images
- Perfect annotations: 3D gaze vectors, pupil size, lighting details
- Used for initial model training

### [MPIIGaze (Real-world)](https://www.kaggle.com/datasets/dhruv413/mpiigaze/data)

- Real eye images under natural conditions
- Gaze coordinates and head pose data
- Used for fine-tuning and real-world adaptation

## 🏗️ Model Architecture

```
Input → TimeDistributed(CNN) → LSTM → Dense(Sigmoid) → Output
```

- **CNN Layers**: Extract spatial features (pupil position, iris contours)
- **LSTM Layer**: Captures temporal dynamics of eye movements
- **Output**: Binary classification (Attentive/Not Attentive)

## 📥 Installation

```bash
git clone https://github.com/pmalaquias/BCC326_TP_PDI
cd BCC326_TP_PDI

# Install dependencies
pip install tensorflow opencv-python numpy pandas scikit-learn scipy
```

## 📈 Results

The model demonstrates:

- High generalization capability on unseen data
- Balanced performance across both classes
- Robustness to threshold variations around 0.50
- Stable training convergence

## 🎮 Real-time Application

The live implementation features:

- Real-time webcam processing
- Dynamic attention status display
- Probability confidence scores
- Continuous duration logging to CSV

## 👥 Contributors

- [Felipe Peret Moraes Sasdelli](https://github.com/felipeperet)
- [Iago Izidório Lacerda](https://github.com/iagoizi)
- [Pedro Igor de Souza Malaquias](https://github.com/pmalaquias)

## 🔮 Future Work

- Expand dataset with diverse real-world scenarios
- Integrate multimodal data (EEG, facial expressions)
- Optimize for mobile and VR platforms
- Clinical validation in ADHD assessment

## 📝 License

This project is available for academic and research purposes. Please cite appropriately if used in research.

---

_For detailed methodology, results analysis, and references, contact us for the full project report._
