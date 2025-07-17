import librosa
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def manual_fft(x):
    N = len(x)
    log2_N = int(np.ceil(np.log2(N)))

    # Zero-padding to the nearest power of 2
    padded_size = 2 ** log2_N
    x = np.pad(x, (0, padded_size - N), 'constant')

    assert len(x) == padded_size, "Error in zero-padding"

    # Bit-reversal permutation
    for i in range(padded_size):
        j = int('{:0{width}b}'.format(i, width=log2_N)[::-1], 2)
        if i < j:
            x[i], x[j] = x[j], x[i]

    # Cooley-Tukey decimation-in-time radix-2 FFT
    for m in range(2, padded_size + 1, 2):
        offset = padded_size // m
        for start in range(0, padded_size, m):
            butterfly(x, start, offset, m // 2, padded_size)

    return x.astype(complex)

def butterfly(arr, start, offset, m, N):
    end = start + m
    even_part = arr[start:end]

    # Check if end:end+m is non-empty before performing the butterfly step
    if end < len(arr):
        odd_part = arr[end:end+m] * np.exp(-2j * np.pi * np.arange(m) / m)

        arr[start:end] = even_part + odd_part
        arr[end:end+m] = even_part - odd_part

def highpass_butterworth_manual(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist

    # Design a high-pass Butterworth filter manually
    x = np.linspace(0, order, order + 1)
    h = np.sinc(2 * normal_cutoff * (x - order / 2.0))
    # Manually implement the Blackman window
    blackman_window = 0.42 - 0.5 * np.cos(2 * np.pi * x / order) + 0.08 * np.cos(4 * np.pi * x / order)
    h *= blackman_window

    h /= np.sum(h)

    # Apply the filter to the data using convolution
    filtered_data = manual_convolution(data, h)

    return filtered_data
def manual_convolution(signal, kernel):
    result = np.zeros_like(signal)

    for i in range(len(signal)):
        for j in range(len(kernel)):
            if i - j >= 0:
                result[i] += signal[i - j] * kernel[j]

    return result
def plot_signal(original_data, filtered_data, sample_rate, title):
    time = np.arange(0, len(original_data)) / sample_rate
    plt.figure(figsize=(12, 4))
    plt.plot(time, original_data, label='Original Signal', color='black')
    plt.plot(time, filtered_data, label='Filtered Signal', color='grey')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_fft(data, sample_rate, title):
    N = len(data)
    T = 1 / sample_rate
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    yf = 2.0/N * np.abs(manual_fft(data)[:N//2])

    plt.figure(figsize=(12, 4))
    plt.plot(xf, yf)
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

def extract_features(audio, sample_rate, mfcc=True, chroma=True, mel=True):
    # Plot the original signal
    plot_signal(audio, np.zeros_like(audio), sample_rate, 'Original Signals')

    # Apply high-pass filter manually
    cutoff_frequency = 50.0  # Adjust the cutoff frequency as needed
    filtered_audio = highpass_butterworth_manual(audio, cutoff_frequency, sample_rate)

    # Plot the filtered signal
    plot_signal(audio, filtered_audio, sample_rate, 'Original and Filtered Signals')

    features = []

    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=filtered_audio, sr=sample_rate, n_mfcc=13), axis=1)
        features.extend(mfccs)

    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(y=filtered_audio, sr=sample_rate), axis=1)
        features.extend(chroma)

    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=filtered_audio, sr=sample_rate), axis=1)
        features.extend(mel)

    return features

# Function to save features to a CSV file
def save_features_to_csv(features, output_csv):
    df = pd.DataFrame([features], columns=[f'feature_{i}' for i in range(len(features))])
    df.to_csv(output_csv, index=False)

# Function to predict the class of input audio
def predict_audio_class(audio_features, classifier):
    return classifier.predict([audio_features])

# Load the audio file and extract features
audio_file_path = r"D:\SEM 3\signal processing\project\elephant_growl\cry_rumble_3.wav"
output_csv_path = 'audio_features.csv'

audio, sample_rate = librosa.load(audio_file_path, res_type='kaiser_fast', duration=3)
audio_features = extract_features(audio, sample_rate)

# Save the features to a CSV file
save_features_to_csv(audio_features, output_csv_path)

# Load the synthetic dataset
data = pd.read_csv(r"D:\SEM 3\signal processing\project\elephant_growl\emotion_dataset2.csv")

# Assume 'labels' is the column containing the emotion labels
X = data.drop('label', axis=1)
y = data['label']

# Normalize the data
X = np.array(X) / np.max(np.abs(X))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier on the training set
clf_random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
clf_random_forest.fit(X_train, y_train)

# Train a Decision Tree Classifier on the training set
clf_decision_tree = DecisionTreeClassifier(random_state=42)
clf_decision_tree.fit(X_train, y_train)

# Predict the class of the input audio using Random Forest Classifier
predicted_class_random_forest = predict_audio_class(audio_features, clf_random_forest)

# Print confusion matrix and classification report for Random Forest Classifier on the test set
y_pred_random_forest = clf_random_forest.predict(X_test)
print("\nConfusion Matrix (Random Forest):")
print(confusion_matrix(y_test, y_pred_random_forest))

print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_pred_random_forest,zero_division=1))

# Predict the class of the input audio using Decision Tree Classifier
predicted_class_decision_tree = predict_audio_class(audio_features, clf_decision_tree)

# Print confusion matrix and classification report for Decision Tree Classifier on the test set
y_pred_decision_tree = clf_decision_tree.predict(X_test)
print("\nConfusion Matrix (Decision Tree):")
print(confusion_matrix(y_test, y_pred_decision_tree))

print("\nClassification Report (Decision Tree):")
print(classification_report(y_test, y_pred_decision_tree,zero_division=1))

print(f"The predicted class for the input audio (Decision Tree) is: {predicted_class_decision_tree[0]}")
print(f"The predicted class for the input audio (Random Forest) is: {predicted_class_random_forest[0]}")
