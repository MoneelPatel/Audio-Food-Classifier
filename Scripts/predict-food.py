import os
import sys
import numpy as np
import torch
import torch.nn as nn
import librosa

# Configuration
SAMPLE_RATE = 22050
DURATION = 5  # seconds per clip
N_MFCC = 40
N_FFT = 2048
HOP_LENGTH = 512
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_ENHANCED_FEATURES = True  # Set this to match your trained model

# Expected food categories - updated to include new foods
EXPECTED_FOOD_CLASSES = [
    'Apple', 'Banana', 'Carrot', 'Chips', 'Fries', 
    'Grape', 'Peanut Butter', 'Popcorn', 'Rice', 'Sandwich'
]

# Enhanced model architecture
class EnhancedAudioModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(EnhancedAudioModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        
        # CNN layers to process spectral features
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout_cnn = nn.Dropout(0.3)
        
        # LSTM to capture temporal dependencies
        self.lstm1 = nn.LSTM(64, hidden_dim, layer_dim, batch_first=True, bidirectional=True)
        self.bn_lstm = nn.BatchNorm1d(hidden_dim * 2)  # * 2 for bidirectional
        self.dropout_lstm = nn.Dropout(0.3)
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * 2, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 64)
        self.bn_fc2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, output_dim)
        
    def forward(self, x):
        # x shape: (batch_size, features, time_steps)
        
        # CNN layers
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout_cnn(x)
        
        # Reshape for LSTM - transpose time and features
        x = x.permute(0, 2, 1)  # Now shape: (batch_size, time_steps, features)
        
        # LSTM layer
        x, _ = self.lstm1(x)  # Shape: (batch_size, time_steps, hidden_dim*2)
        
        # Attention mechanism
        attention_weights = torch.softmax(self.attention(x), dim=1)
        x = torch.sum(attention_weights * x, dim=1)  # Weighted sum across time steps
        
        # Apply batch normalization
        x = self.bn_lstm(x)
        x = self.dropout_lstm(x)
        
        # Fully connected layers
        x = torch.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

def extract_enhanced_features(file_path, use_enhanced=True, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH):
    try:
        # Load audio file
        audio, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # Ensure consistent length
        if len(audio) < SAMPLE_RATE * DURATION:
            audio = np.pad(audio, (0, SAMPLE_RATE * DURATION - len(audio)))
        else:
            audio = audio[:SAMPLE_RATE * DURATION]
        
        # Extract standard MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        
        if use_enhanced:
            # Extract additional features
            # Spectral contrast captures the difference between peaks and valleys in the spectrum
            contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate, n_bands=6, fmin=200.0)
            
            # Chroma features relate to the 12 different pitch classes
            chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate, n_chroma=12, n_fft=n_fft)
            
            # Zero-crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio, frame_length=n_fft, hop_length=hop_length)
            
            # RMS energy
            rms = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=hop_length)
            
            # Spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate, roll_percent=0.85)
            
            # Spectral bandwidth
            bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
            
            # Spectral flatness
            flatness = librosa.feature.spectral_flatness(y=audio, n_fft=n_fft, hop_length=hop_length)
            
            # Combine all features
            features = np.vstack([
                mfccs,          # Shape (n_mfcc, time)
                contrast,       # Shape (n_bands+1, time)
                chroma,         # Shape (n_chroma, time)
                zcr,            # Shape (1, time)
                rms,            # Shape (1, time) 
                rolloff,        # Shape (1, time)
                bandwidth,      # Shape (1, time)
                flatness        # Shape (1, time)
            ])
        else:
            features = mfccs
        
        # Normalize features
        features = (features - np.mean(features, axis=1, keepdims=True)) / (np.std(features, axis=1, keepdims=True) + 1e-10)
        
        return features
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

def get_food_classes(data_dir='WAV_Clips'):
    # First try to get classes from directory
    food_classes = []
    
    # Check if the directory exists
    if os.path.exists(data_dir):
        # Get all food categories
        for food_category in os.listdir(data_dir):
            food_path = os.path.join(data_dir, food_category)
            if os.path.isdir(food_path):
                food_classes.append(food_category)
        
        food_classes.sort()  # Ensure consistent ordering
    
    # If no classes found or directory doesn't exist, use expected classes
    if not food_classes:
        print("Couldn't detect food classes from directory. Using expected classes list.")
        food_classes = EXPECTED_FOOD_CLASSES
    
    return food_classes

def load_model(model_path):
    print(f"Loading model from {model_path}...")
    # Load the saved model
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Full model with metadata
        food_classes = checkpoint['food_classes']
        input_shape = checkpoint['input_shape']
        
        # Get enhanced_features flag if available
        global USE_ENHANCED_FEATURES
        if 'enhanced_features' in checkpoint:
            USE_ENHANCED_FEATURES = checkpoint['enhanced_features']
            print(f"Setting enhanced_features to {USE_ENHANCED_FEATURES} based on model")
        
        # Initialize model
        hidden_dim = 128
        layer_dim = 2
        output_dim = len(food_classes)
        
        model = EnhancedAudioModel(input_shape[0], hidden_dim, layer_dim, output_dim).to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model with input shape {input_shape}")
    else:
        # Just the state dict (older format)
        food_classes = get_food_classes()
        model = torch.jit.load(model_path)
    
    print(f"Model loaded. Recognized food classes: {food_classes}")
    return model, food_classes

def predict_food(model_path, audio_path):
    # Load the model
    model, food_classes = load_model(model_path)
    
    # Set model to evaluation mode
    model.eval()
    
    # Extract features from the audio file
    print(f"Extracting features from {audio_path}...")
    features = extract_enhanced_features(audio_path, use_enhanced=USE_ENHANCED_FEATURES)
    
    if features is None:
        print(f"Could not extract features from {audio_path}")
        return
    
    print(f"Feature shape: {features.shape}")
    
    # Reshape for model input
    features_tensor = torch.FloatTensor(features).unsqueeze(0).to(DEVICE)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(features_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = food_classes[predicted.item()]
    confidence_value = confidence.item() * 100
    
    print(f"\nPrediction for {os.path.basename(audio_path)}:")
    print(f"Predicted food: {predicted_class}")
    print(f"Confidence: {confidence_value:.2f}%")
    
    # Show top 5 predictions (expanded for more foods)
    num_predictions = min(5, len(food_classes))
    top_values, top_indices = torch.topk(probabilities, num_predictions, dim=1)
    print(f"\nTop {num_predictions} predictions:")
    for i in range(num_predictions):
        idx = top_indices[0][i].item()
        value = top_values[0][i].item() * 100
        print(f"{i+1}. {food_classes[idx]}: {value:.2f}%")
        
    # Show similar food comparisons
    print("\nComparisons between similar foods:")
    similar_pairs = [
        ('Fries', 'Chips'),
        ('Rice', 'Peanut Butter'),
        ('Rice', 'Sandwich'),
        ('Banana', 'Apple'),
        ('Peanut Butter', 'Sandwich')
    ]
    
    for food1, food2 in similar_pairs:
        if food1 in food_classes and food2 in food_classes:
            idx1 = food_classes.index(food1)
            idx2 = food_classes.index(food2)
            prob1 = probabilities[0][idx1].item() * 100
            prob2 = probabilities[0][idx2].item() * 100
            print(f"{food1} vs {food2}: {prob1:.2f}% vs {prob2:.2f}% (difference: {abs(prob1-prob2):.2f}%)")
    
    return predicted_class, confidence_value

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_food.py <audio_file_path> [model_path]")
        print("Default model path is 'food_audio_classifier_full.pth'")
        return
    
    audio_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else 'food_audio_classifier_full.pth'
    
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        return
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
    
    predict_food(model_path, audio_path)

if __name__ == "__main__":
    main()