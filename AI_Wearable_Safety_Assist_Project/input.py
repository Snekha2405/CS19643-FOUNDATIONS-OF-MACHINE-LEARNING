import wfdb
import numpy as np
from main import record_audio
from scipy.signal import butter, filtfilt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import os

# Step 1: Load ECG data
record = wfdb.rdrecord('data/101')  # Ensure 100.dat, 100.hea are in 'data/'
signal = record.p_signal[:, 0]  # Channel 0 (Lead II)

# Step 2: Bandpass filter
def bandpass_filter(data, lowcut=0.5, highcut=50.0, fs=360, order=5):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)

filtered_signal = bandpass_filter(signal)

# Step 3: Normalize
normalized = (filtered_signal - np.min(filtered_signal)) / (np.max(filtered_signal) - np.min(filtered_signal))

# Step 4: Segment the signal into windows
window_size = 180
segments = [normalized[i:i + window_size] for i in range(0, len(normalized) - window_size, window_size)]
segments = np.array(segments)

# Step 5: Load annotations and create binary labels
ann = wfdb.rdann('data/101', 'atr')

def label_segments(ann, num_segments, window_size):
    y = []
    arrhythmia_symbols = ['V', 'A', 'L', '!']  # Customize based on use case
    for i in range(num_segments):
        start = i * window_size
        end = start + window_size
        abnormal = any(
            sym in arrhythmia_symbols
            for s, sym in zip(ann.sample, ann.symbol)
            if start <= s < end
        )
        y.append(1 if abnormal else 0)
    return np.array(y)

y = label_segments(ann, len(segments), window_size)

# Step 6: Reshape data for model input
X = segments.reshape(-1, 180, 1)

# Step 7: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Build CNN model
model = Sequential([
    Conv1D(32, 3, activation='relu', input_shape=(180, 1)),
    MaxPooling1D(2),
    Conv1D(64, 3, activation='relu'),
    MaxPooling1D(2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 9: Train model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Step 10: Save model
os.makedirs("models", exist_ok=True)
model.save("models/ecg_model.h5")

# Step 11: Load model and predict
model = load_model("models/ecg_model.h5")
import matplotlib.pyplot as plt

# Store predictions for each segment
predictions = []

epred = model.predict(segments[1].reshape(1, 180, 1))[0][0]

# Step 12: Simulate real-time detection (Stop after first abnormal)
for i in range(100):
    print(f"Checking segment {i}")
    pred = model.predict(segments[i].reshape(1, 180, 1))[0][0]
    predictions.append(pred)
    print(f"Prediction score: {pred}")

    if pred > epred:
        print("🚨 Abnormal ECG detected! Sending alert!")
        record_audio()
        break  # Stop checking further segments
    else:
        print("✅ Normal ECG.")

# Plot the predictions
plt.figure(figsize=(12, 6))
plt.plot(predictions, label="Prediction Score")
plt.axhline(y=epred, color='r', linestyle='--', label=f"Threshold ({epred})")
plt.xlabel("Segment Index")
plt.ylabel("Prediction Score")
plt.title("Abnormal Data Detection in ECG Segments")
plt.legend()

# Highlight abnormal segments
abnormal_segments = [i for i, pred in enumerate(predictions) if pred > epred]
for segment in abnormal_segments:
    plt.axvline(x=segment, color='orange', linestyle='--', alpha=0.7)

plt.show()