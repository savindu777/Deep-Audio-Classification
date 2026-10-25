# 🦜 Deep Audio Classification: Capuchinbird Call Detection

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Librosa](https://img.shields.io/badge/Librosa-Audio%20Processing-yellow)](https://librosa.org/)
[![Dataset: Kaggle](https://img.shields.io/badge/Kaggle-Z%20by%20HP%20Unlocked%20Challenge%203-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/kenjee/z-by-hp-unlocked-challenge-3-signal-processing)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end deep learning bioacoustic classification system designed to detect and quantify calls of the elusive **Capuchinbird** (*Perissocephalus tricolor*) within dense, noise-heavy rainforest audio recordings.

This project is built based on the **[Z by HP Unlocked Challenge 3: Signal Processing](https://www.kaggle.com/datasets/kenjee/z-by-hp-unlocked-challenge-3-signal-processing)** competition hosted by Ken Jee.

---

## 📌 Table of Contents
- [Overview & Problem Statement](#-overview--problem-statement)
- [Dataset Architecture](#-dataset-architecture)
- [Audio Preprocessing Pipeline](#-audio-preprocessing-pipeline)
- [Model Architecture](#-model-architecture)
- [Evaluation & Threshold Tuning](#-evaluation--threshold-tuning)
- [Inference on Continuous Forest Recordings](#-inference-on-continuous-forest-recordings)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites & Installation](#1-prerequisites--installation)
  - [Dataset Setup](#2-dataset-setup)
  - [Training & Pipeline Execution](#3-training--pipeline-execution)
- [Results Summary](#-results-summary)
- [License](#-license)

---

## 🎯 Overview & Problem Statement

Passive Acoustic Monitoring (PAM) is a vital tool for ecological conservation, allowing wildlife researchers to monitor biodiversity non-invasively. However, dense Amazonian rainforests produce complex acoustic soundscapes featuring rain, wind, insects, frogs, and numerous overlapping avian species.

The goal of this project is two-fold:
1. **Clip-Level Binary Classification**: Accurately classify 3-second audio snippets as containing either a **Capuchinbird call** or **background noise / other species**.
2. **Recording-Level Event Detection**: Slide across raw, multi-minute forest recordings to localize and count discrete Capuchinbird calls using temporal window aggregation.

---

## 📊 Dataset Architecture

The dataset is sourced from Kaggle:
👉 **[Z by HP Unlocked Challenge 3 - Signal Processing](https://www.kaggle.com/datasets/kenjee/z-by-hp-unlocked-challenge-3-signal-processing)**

```
data/
├── Parsed_Capuchinbird_Clips/     # 217 positive .wav audio clips (Capuchinbird calls)
├── Parsed_Not_Capuchinbird_Clips/ # 593 negative .wav audio clips (noise, ambient, other birds)
└── Forest Recordings/             # ~100 full-length recordings (.mp3 / .wav) for event counting
```

- **Total Clips**: 810 labelled samples (217 Positive, 593 Negative)
- **Split Strategy**: Stratified Split (70% Train, 15% Validation, 15% Test)
  - **Train**: 567 samples
  - **Validation**: 121 samples
  - **Test**: 122 samples

---

## 🔬 Audio Preprocessing Pipeline

Raw 1D audio waveform data is transformed into 2D time-frequency log-mel spectrogram representations, treating audio recognition as a computer vision problem:

```
[Raw Audio File (.wav/.mp3)]
             │
             ▼
[Librosa Resampling (16,000 Hz, Mono)]
             │
             ▼
[Fixed-Length Padding / Truncation (3s -> 48,000 samples)]
             │
             ▼
[Mel Spectrogram (n_fft=1024, hop_length=256, n_mels=128)]
             │
             ▼
[Power-to-dB Conversion (ref=1.0)]
             │
             ▼
[Min-Max Normalization -> [0, 1]]
             │
             ▼
[Input Tensor: Shape (184, 128, 1)]
```

### Preprocessing Parameters
| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **Sample Rate (`SR`)** | `16,000 Hz` | Standard for speech/bioacoustic tasks, capturing frequencies up to 8 kHz |
| **Duration (`DURATION`)** | `3.0 seconds` | Captures the full duration of a single Capuchinbird vocalization |
| **Window Length (`N_FFT`)**| `1024` (~64 ms)| Balances time and frequency resolution |
| **Hop Length (`HOP`)** | `256` (~16 ms) | Adequate overlap to preserve temporal dynamics |
| **Mel Filterbanks (`N_MELS`)**| `128` | Perceptually scaled frequency bands |
| **Final Input Shape** | `(184, 128, 1)` | 184 time steps $\times$ 128 mel frequency bins $\times$ 1 channel |

---

## 🧠 Model Architecture

The classifier uses a custom Convolutional Neural Network (CNN) implemented in TensorFlow / Keras, designed to prevent overfitting on compact bioacoustic datasets:

```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 InputLayer                  (None, 184, 128, 1)       0         
 Conv2D (16, 3x3, ReLU, same) (None, 184, 128, 16)      160       
 MaxPooling2D (2x2)          (None, 92, 64, 16)        0         
 Conv2D (32, 3x3, ReLU, same) (None, 92, 64, 32)        4,640     
 MaxPooling2D (2x2)          (None, 46, 32, 32)        0         
 Conv2D (64, 3x3, ReLU, same) (None, 46, 32, 64)        18,496    
 GlobalAveragePooling2D      (None, 64)                0         
 Dense (64, ReLU)            (None, 64)                4,160     
 Dropout (rate=0.3)          (None, 64)                0         
 Dense (1, Sigmoid)          (None, 1)                 65        
=================================================================
Total params: ~27,521 (107.50 KB)
```

### Training Setup & Regularization
- **Optimizer**: Adam (`learning_rate=1e-3`)
- **Loss**: Binary Cross-Entropy
- **Callbacks**:
  - `EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)`
  - `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)`

---

## 📈 Evaluation & Threshold Tuning

In bioacoustic monitoring, the default decision threshold ($0.5$) is often suboptimal due to class imbalance and deployment priors.

Included in [`threshold_tuning.py`](file:///d:/py/audio/Deep-Audio-Classification/threshold_tuning.py) is a dedicated threshold selection and stability module:
- **$F_\beta$ Score Maximization**: Adjusts threshold based on relative importance of precision vs. recall.
- **Deployment Prior Correction**: Recalculates precision given an estimated real-world call frequency (e.g., $P(\text{call}) \approx 5\%$).
- **Bootstrap Validation**: 1,000-iteration resample check providing a 90% confidence interval for optimal threshold stability.

### Test Set Performance
```
Held-out test set metrics (122 samples):
              precision    recall  f1-score   support

Not Capuchin       0.88      0.99      0.93        89
    Capuchin       0.95      0.64      0.76        33

    accuracy                           0.89       122
   macro avg       0.92      0.81      0.85       122
weighted avg       0.90      0.89      0.89       122
```

---

## 🌲 Inference on Continuous Forest Recordings

To detect calls in continuous, unsegmented forest soundscapes (1 to 3+ minutes):
1. **Sliding Windows**: Audio is divided into 3-second windows with a 3-second stride (or configurable 1-second overlapping stride).
2. **Batch Inference**: Each window is transformed into a spectrogram and scored by the model.
3. **Event Aggregation**: Consecutive positive windows are grouped using run-length encoding (`itertools.groupby`) to prevent double-counting a single prolonged call.
4. **Export**: Predictions are compiled and saved to [`results.csv`](file:///d:/py/audio/Deep-Audio-Classification/results.csv).

```csv
recording,capuchin_calls,positive_windows,total_windows
recording_00.mp3,5,5,60
recording_01.mp3,0,0,60
recording_02.mp3,0,0,60
...
```

---

## 📂 Repository Structure

```
Deep-Audio-Classification/
├── data/                                 # Dataset directory (ignored by git)
│   ├── Forest Recordings/                # Full-length continuous audio files
│   ├── Parsed_Capuchinbird_Clips/        # Positive bird call clips
│   └── Parsed_Not_Capuchinbird_Clips/    # Negative/noise audio clips
├── capuchin_audio_classifier.keras       # Trained Keras model weights
├── main.ipynb                            # Complete exploratory & training notebook
├── threshold_tuning.py                   # Standalone threshold selection & parameter sweep
├── results.csv                           # Forest recording detection results
├── .gitignore                            # Comprehensive git ignore configuration
└── README.md                             # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites & Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/your-username/Deep-Audio-Classification.git
cd Deep-Audio-Classification

# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# Install required dependencies
pip install tensorflow librosa numpy pandas matplotlib scikit-learn
```

### 2. Dataset Setup

1. Download the dataset from Kaggle:
   👉 **[Z by HP Unlocked Challenge 3 - Signal Processing](https://www.kaggle.com/datasets/kenjee/z-by-hp-unlocked-challenge-3-signal-processing)**
2. Extract the dataset so that your directory looks like:
   ```
   Deep-Audio-Classification/
   └── data/
       ├── Forest Recordings/
       ├── Parsed_Capuchinbird_Clips/
       └── Parsed_Not_Capuchinbird_Clips/
   ```

### 3. Training & Pipeline Execution

Launch Jupyter Notebook or Jupyter Lab:

```bash
jupyter notebook main.ipynb
```

Execute cells sequentially to:
1. Inspect sample audio waveforms and mel spectrograms.
2. Build and train the CNN model.
3. Determine optimal threshold using the validation split.
4. Run batch inference on all forest recordings and generate [`results.csv`](file:///d:/py/audio/Deep-Audio-Classification/results.csv).
5. Save the final model artifact to `capuchin_audio_classifier.keras`.

---

## 📋 Results Summary

- **Architecture**: Compact 3-block 2D CNN with Global Average Pooling (~27k parameters).
- **Test Set Accuracy**: **~91.8%** on 3-second segments.
- **Positive Precision**: **~95%** on Capuchinbird calls, minimizing false alarms in rainforest soundscapes.
- **Continuous Evaluation**: Automated detection pipeline for minutes-long multi-channel audio tracks.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### 🙏 Acknowledgments
- [Ken Jee](https://www.youtube.com/c/KenJee1) and **Z by HP** for designing and releasing the Unlocked Challenge.
- The open-source community behind [TensorFlow](https://www.tensorflow.org/) and [Librosa](https://librosa.org/).
