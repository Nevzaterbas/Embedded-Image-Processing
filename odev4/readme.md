# Student Information

Nevzat Tarık Erbaş - 150720058



# 1. QUESTION
Application of Single-Neuron Model for Digit Identification (Ref: Sec. 10.9)

---
## 2.1 Methodology and Data Preprocessing In this study, a binary classification framework based on logistic regression (a single neuron) was implemented to distinguish the digit '0' from other numerals.

Dataset Acquisition: The MNIST dataset served as the source, where training and testing images were parsed from the IDX file format and restructured into Numpy arrays for processing.

Feature Engineering: To optimize the input vector, the model utilized 7 Hu Moments extracted via the cv2.HuMoments function in OpenCV, rather than processing the raw 28x28 pixel intensity matrices. This approach provides geometric descriptors that remain invariant despite scaling or rotation variations.

Normalization: Input features were standardized using Z-score normalization, utilizing the mean and standard deviation derived solely from the training set to ensure numerical stability and faster convergence.

Target Encoding: The classification targets were binarized, labeling the digit '0' as Class 0 and all other digits collectively as Class 1 ('non-zero').

![alt text](<photo1.png) 
## 2.2 Network Configuration and Training Protocol The architecture consists of a solitary neuron employing a Sigmoid activation function to estimate class probabilities.

Optimization: The Adam optimizer was selected with a learning rate set to 0.001.

Objective Function: The model minimized the Binary Crossentropy loss.

Handling Imbalance: To counteract the dataset imbalance (where non-zero digits significantly outnumber zeros), a class weight of 8 was applied to the minority class (digit '0').

Training Cycle: The model underwent training for a total of 50 epochs.

2.3 Experimental Results Post-training evaluation was conducted on the test set, with performance metrics detailed in the Confusion Matrix (Figure 1).

Figure 1: Confusion Matrix for Single Neuron Training

Analysis of the matrix reveals the following:

The system successfully identified 944 samples of the target digit '0'.

False negatives were minimized to 36 instances.

The model demonstrated high precision in the 'non-zero' category, correctly classifying 7529 samples.

These findings suggest that utilizing a compact, single-neuron architecture in conjunction with low-dimensional Hu moment features is a highly efficient strategy for this specific binary classification task.


# 2. QUESTION
Multi-Class Digit Recognition with MLP (Ref: Sec. 11.8)

### 2.4 Methodology This study expanded the scope to classify all 10 MNIST digits (0-9). Consistent with the previous experiment, the feature vector consisted of 7 standardized Hu Moments, while the target labels were preserved in their original multi-class format. 
![alt text](<Screenshot 2026-01-07 220831.png>)
### 2.5 Model Architecture A Multi-Layer Perceptron (MLP) was implemented to handle the 10-class problem:

Input: 7 neurons receiving the Hu moments.

Hidden Layers: Two sequential dense layers, each with 100 neurons and ReLU activation.

Output: 10 neurons using Softmax activation to output class probabilities.

### 2.6 Training Configuration The model utilized the Adam optimizer (lr=0.001) and minimized Sparse Categorical Crossentropy loss. To prevent overfitting and ensure optimal weight retention, EarlyStopping and ModelCheckpoint callbacks were applied within a maximum 1000-epoch limit.

### 2.7 Results The Confusion Matrix (Figure 2) illustrates the model's performance. High diagonal values (e.g., 1097 correct predictions for digit '1') indicate successful generalization across all classes. These results confirm that the MLP architecture is effective for multi-class recognition even with a compact feature set.