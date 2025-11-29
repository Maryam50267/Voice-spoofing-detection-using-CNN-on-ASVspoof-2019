

# 🎤 Voice Anti-Spoofing Detection (ASVspoof 2019 Challenge)

This repository contains the implementation of a deep learning model designed to classify audio samples as either **Bonafide** (genuine human voice) or **Spoof** (synthesized or manipulated voice). The system utilizes a hybrid Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) architecture trained on the ASVspoof 2019 Logical Access (LA) dataset.

## 🚀 Key Results on Unseen Evaluation Data

The model's performance is measured using the standard biometric metrics, **Equal Error Rate (EER)** and **Area Under the ROC Curve (AUC)**. The system achieved the following performance on the large, unseen evaluation set ($\sim 71k$ samples):

| Metric | Result | Interpretation |
| :--- | :--- | :--- |
| **EER (Equal Error Rate)** | $\mathbf{17.12\%}$ | The true, balanced error rate for the system. A strong result for a complex anti-spoofing task. |
| **AUC** | $\mathbf{0.9018}$ | The model has a $90\%$ chance of correctly distinguishing between a random genuine sample and a spoof sample. |
| **Optimal Decision Threshold** | $\mathbf{0.9950}$ | Confirms the model's high certainty requirement for classifying a sample as Bonafide. |

---

## 🧠 Model Architecture: CNN-LSTM Hybrid

The final model utilizes a high-capacity CNN-LSTM architecture heavily stabilized with regularization techniques to combat the severe overfitting observed during development.

### Feature Extraction

All audio files are pre-processed into **120-dimensional spectral feature vectors** per frame:
* $\mathbf{40}$ Mel-Frequency Cepstral Coefficients (MFCCs)
* $\mathbf{40}$ Delta MFCCs ($\Delta$)
* $\mathbf{40}$ Delta-Delta MFCCs ($\Delta\Delta$)

### Final Model Configuration

The model is designed to first extract spatial features using CNNs, and then process the resulting sequence of features over time using LSTMs.

| Layer Type | Configuration | Key Regularization/Edit |
| :--- | :--- | :--- |
| **Input** | `(100, 120, 1)` | MFCCs, Deltas, Delta-Deltas |
| **Conv2D Block 1** | 32 filters, 3x3, ReLU | `Dropout(0.35)` |
| **Conv2D Block 2** | 64 filters, 3x3, ReLU | `Dropout(0.35)` |
| **Conv2D Block 3** | 128 filters, 3x3, ReLU | N/A |
| **Reshape** | Transition to sequence for LSTM | N/A |
| **LSTM** | 64 units, sequence processing | `Dropout(0.70)` **(Maximized)** |
| **Dense (Output)** | 1 unit, Sigmoid activation | `L2(0.02)` **(Aggressive Penalty)** |

## 🛠️ Key Solutions Implemented

The project successfully resolved critical issues stemming from data imbalance and high model complexity:

### 1. Handling Class Imbalance

* **Strategy:** Removed initial **undersampling** (which discarded crucial data) and switched to using the **full training dataset**.
* **Implementation:** Applied **`class_weight=class_weight_dict`** during `model.fit()` to ensure the minority class (**Bonafide**) contributed appropriately to the loss function, preventing the model from ignoring rare samples.

### 2. Overcoming Severe Overfitting

The model was highly prone to memorizing the training data. This was resolved by aggressively increasing penalties:
* **Maximized Dropout:** Increased the LSTM dropout layer to **$0.70$** to heavily disrupt temporal dependencies.
* **Aggressive L2 Regularization:** Increased the L2 penalty on the final `Dense` layer to **$0.02$** to force the model's decision boundary to be simpler and more generalizable.

### 3. Correct Metric Evaluation

The default $0.5$ threshold failed catastrophically due to model miscalibration. The final evaluation correctly utilized the **EER threshold ($\mathbf{0.9950}$)** derived from the ROC analysis to produce the accurate Classification Report and final EER score.

---

## ⚙️ Setup and Usage

### Prerequisites

* Python 3.x
* TensorFlow / Keras
* NumPy
* Pandas
* Librosa (for audio processing)
* Scikit-learn
* `tqdm` (for progress bars)

### Training & Evaluation

1.  **Feature Extraction:** Run the feature extraction code to generate $\mathbf{X}$ and $\mathbf{y}$ arrays from the `train_audio` and `train_protocol`.
2.  **Weight Calculation:** Calculate `class_weight_dict` using `sklearn.utils.class_weight.compute_class_weight` on the full label set $\mathbf{y}$.
3.  **K-Fold Training:** Execute the K-Fold loop, passing `class_weight=class_weight_dict` to `model.fit()`. The model automatically uses `EarlyStopping` and saves the best weights.
4.  **Final Test:** Load the saved model (`audio_classification.keras`) and evaluate it against the unseen `eval_audio` and `eval_protocol` using the EER calculation to determine the final threshold.
