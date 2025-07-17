# 🎯Objective
The primary objective of this project is to classify elephant vocalizations into various emotional categories such as calmness, stress, aggression, and alertness. This is achieved by performing signal processing on audio recordings of elephant calls, extracting relevant acoustic features, and applying machine learning classifiers to accurately predict the emotional state of the elephant.

Emotions are inferred based on distinct types of elephant vocalizations such as cry rumble, baroo, blast, roar, roar rumble, and trumpet. Each of these vocal types carries unique acoustic signatures that correspond to specific emotional cues. By analyzing these differences in frequency content and signal behavior, the system is able to distinguish between various emotional states.

This approach offers a non-invasive, audio-based solution to monitor elephant behavior and emotional well-being in both captive and wild environments, aiding conservationists and wildlife officials in early detection of distress or agitation.

## 📂 Dataset Description
The dataset used in this project is a real-world collection of elephant vocalizations sourced from [ElephantVoices](https://www.elephantvoices.org/)  . The audio recordings(.WAV files) were pre-labeled with corresponding emotional states. These audio files were processed and converted into CSV files after extracting features such as MFCCs (Mel Frequency Cepstral Coefficients), Chroma features, Spectral Contrast, and Zero-Crossing Rate . Each record in the CSV dataset corresponds to an audio sample with its associated emotion label. The dataset was then divided into training and testing sets for model development and evaluation.
Dataset CSV format - [CSV DATASET](https://1drv.ms/x/c/1d593e6a448c4948/EaJ8Kvh4KbVHrV7vz4RggxUBT6Fz2I4GtmldcSezaYQPJA?e=J1pdF7)

## 🔧Technologies and Tools 
The project was developed using Python and incorporates various libraries and manual algorithms for processing and classification:
*  Librosa: For loading audio files and extracting features such as MFCCs, spectral contrast, Chroma, and Zero-Crossing Rate.
*  NumPy & Pandas: For numerical computation, data handling, and CSV file management.
*  Matplotlib: For plotting time-domain and frequency-domain signal graphs.
*  Scikit-learn: For implementing machine learning models such as Decision Tree and Random Forest classifiers, along with model evaluation tools like confusion matrices and classification reports.

Manual Implementations:
*  Fast Fourier Transform (FFT)
*  Butterworth high-pass filter implementation using Blackman window and sinc function for denoising.

## ⚙️Signal Processing Pipeline

* High-Pass Filtering:
A manually implemented Butterworth high-pass filter is applied to remove low-frequency noise from the raw audio. The filter is designed using the sinc function and a Blackman window to smooth the response and ensure accurate isolation of relevant high-frequency features.

* Frequency Domain Analysis (FFT):
A manual implementation of the Fast Fourier Transform (FFT) is used to convert the time-domain audio signal into the frequency domain. This helps in identifying frequency patterns present in different emotional vocalizations.

## 🚀 Feature Extraction:
Acoustic features are extracted from the filtered audio:
1) MFCCs: Represent the short-term power spectrum of sound.

2) Chroma: Represents the 12 different pitch classes.

3) Spectral Contrast: Measures the contrast between spectral peaks and valleys.

4) Zero-Crossing Rate: Indicates the rate at which the signal changes sign (used to detect noisy/percussive content).

5) Normalization: The dataset is normalized using Max Abs Scaling, which scales each value by dividing it with the maximum absolute value in the dataset, ensuring consistent and effective feature scaling for the models.


## 🚀Machine Learning Models
Two machine learning models were trained and evaluated to classify emotional states from extracted features:

- Random Forest Classifier: A robust ensemble model using multiple decision trees. It achieved higher accuracy and better generalization by reducing overfitting. Final accuracy: 64%.

- Decision Tree Classifier: A simple, interpretable model that splits the data into decision paths. Although it performed slightly lower than Random Forest, it provided clear insight into decision-making. Final accuracy: 55%.


## 📊 Evaluation and Performance Metrics 

- Precision: Measures how many predicted positives were actually correct.

- Recall: Measures how many actual positives were captured.

- F1-Score: Harmonic mean of precision and recall.

- Confusion Matrix: Shows true vs predicted labels for all classes.



## 📈 Visualization 

- Time-Domain Plot: Displays both original and filtered waveforms for understanding noise reduction.
  
- Frequency-Domain Plot (FFT): Shows signal strength across frequency bins.
  
- Confusion Matrices and classification reports for both models are printed for detailed performance analysis.


## ✅ Result and Inference:
Random Forest Classifier achieved an accuracy of 64% on the test data.
Decision Tree Classifier achieved an accuracy of 55% on the test data.

The input audio was classified as:

🔹 roar_rumble by Random Forest

🔹 blast by Decision Tree

## 🧠 Conclusion
Future improvements can focus on deep learning integration and increasing the dataset size for better generalization and more accurate classification across all elephant vocal types.

🐘 This project showcases the practical application of acoustic signal processing and machine learning in the field of wildlife monitoring. By analyzing the vocalizations of elephants, it offers a non-invasive and effective method for understanding their emotional states. The system serves as a digital tool that can assist veterinarians, conservationists, and forest officials in monitoring elephant well-being and behavior. Furthermore, it lays the groundwork for future advancements, such as the development of real-time emotion monitoring systems and the integration of more advanced deep learning models to enhance accuracy and scalability.
