# 🎯Objective
The primary objective of this project is to classify elephant vocalizations into various emotional categories such as calmness, stress, aggression, and alertness. This is achieved by performing signal processing on audio recordings of elephant calls, extracting relevant acoustic features, and applying machine learning classifiers to accurately predict the emotional state of the elephant.

Emotions are inferred based on distinct types of elephant vocalizations such as cry rumble, baroo, blast, roar, roar rumble, and trumpet. Each of these vocal types carries unique acoustic signatures that correspond to specific emotional cues. By analyzing these differences in frequency content and signal behavior, the system is able to distinguish between various emotional states.

This approach offers a non-invasive, audio-based solution to monitor elephant behavior and emotional well-being in both captive and wild environments, aiding conservationists and wildlife officials in early detection of distress or agitation.

## 📂 Dataset Description
The dataset used in this project is a real-world collection of elephant vocalizations sourced from [ElephantVoices](https://www.elephantvoices.org/)  . The audio recordings(.WAV files) were pre-labeled with corresponding emotional states. These audio files were processed and converted into CSV files after extracting features such as MFCCs (Mel Frequency Cepstral Coefficients), Chroma features, Spectral Contrast, and Zero-Crossing Rate . Each record in the CSV dataset corresponds to an audio sample with its associated emotion label. The dataset was then divided into training and testing sets for model development and evaluation.
Dataset CSV format - [features](https://1drv.ms/x/c/1d593e6a448c4948/EeBO5c060vpLvvR16kUfJroBmjecgtzpeNcGLJCzbzsbvQ?e=IwB5EZ)
