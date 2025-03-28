import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import tqdm
from collections import Counter

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Configuration
DATA_DIR = 'WAV_Clips'
SAMPLE_RATE = 22050  # Standard sample rate, adjust if your WAVs have a different rate
DURATION = 5  # seconds per clip
N_MFCC = 40  # Number of MFCC features
N_FFT = 2048  # FFT window size
HOP_LENGTH = 512  # Hop length for FFT
BATCH_SIZE = 32
EPOCHS = 100  # Increased epochs
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 0.001
USE_AUGMENTATION = True  # Enable data augmentation
USE_CLASS_WEIGHTS = True  # Enable class weighting
USE_ENHANCED_FEATURES = True  # Enable enhanced feature extraction

# Expected food categories - updated to include new foods
EXPECTED_FOOD_CLASSES = [
    'Apple', 'Banana', 'Carrot', 'Chips', 'Fries', 
    'Grape', 'Peanut Butter', 'Popcorn', 'Rice', 'Sandwich'
]  # This is just for verification, actual classes are detected from folders

print(f"Using device: {DEVICE}")

# Function to augment audio data
def augment_audio(audio, sample_rate):
    augmentation_type = np.random.choice(['stretch', 'pitch', 'noise', 'original'], p=[0.3, 0.3, 0.3, 0.1])
    
    if augmentation_type == 'stretch':
        # Time stretching (slower/faster)
        stretch_factor = np.random.uniform(0.8, 1.2)
        audio_augmented = librosa.effects.time_stretch(audio, rate=stretch_factor)
    elif augmentation_type == 'pitch':
        # Pitch shifting (higher/lower)
        pitch_steps = np.random.uniform(-3, 3)
        audio_augmented = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=pitch_steps)
    elif augmentation_type == 'noise':
        # Add small noise
        noise_factor = np.random.uniform(0.005, 0.02)
        noise = np.random.randn(len(audio))
        audio_augmented = audio + noise_factor * noise
    else:
        # Original with small volume variation
        volume_factor = np.random.uniform(0.9, 1.1)
        audio_augmented = audio * volume_factor
    
    # Ensure consistent length
    if len(audio_augmented) < sample_rate * DURATION:
        audio_augmented = np.pad(audio_augmented, (0, sample_rate * DURATION - len(audio_augmented)))
    else:
        audio_augmented = audio_augmented[:sample_rate * DURATION]
    
    return audio_augmented

# Function to extract enhanced features from audio file
def extract_enhanced_features(file_path, use_enhanced=True, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH):
    try:
        # Load audio file
        audio, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # Ensure consistent length
        if len(audio) < SAMPLE_RATE * DURATION:
            audio = np.pad(audio, (0, SAMPLE_RATE * DURATION - len(audio)))
        else:
            audio = audio[:SAMPLE_RATE * DURATION]
        
        # Optionally augment audio (during training)
        # Apply augmentation more frequently to problematic food categories
        if USE_AUGMENTATION and ('Fries' in file_path or 'Rice' in file_path or 'Peanut Butter' in file_path) and np.random.random() < 0.8:
            # Apply augmentation more frequently to potentially challenging samples
            # Rice and Peanut Butter might have similar texture to other foods
            audio = augment_audio(audio, sample_rate)
        elif USE_AUGMENTATION and np.random.random() < 0.5:
            # Apply augmentation to other samples with 50% probability
            audio = augment_audio(audio, sample_rate)
        
        # Extract standard MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        
        if use_enhanced:
            # Extract additional features
            
            # Spectral contrast captures the difference between peaks and valleys in the spectrum
            # Good for distinguishing between different textures of sounds
            contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate, n_bands=6, fmin=200.0)
            
            # Chroma features relate to the 12 different pitch classes
            # Useful for capturing tonal content
            chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate, n_chroma=12, n_fft=n_fft)
            
            # Zero-crossing rate - useful for discriminating between voiced/unvoiced sounds
            # Can help distinguish between continuous sounds and discrete crunches
            zcr = librosa.feature.zero_crossing_rate(audio, frame_length=n_fft, hop_length=hop_length)
            
            # RMS energy - overall volume/energy of the signal
            # Can help distinguish between loud crunchy foods and softer foods
            rms = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=hop_length)
            
            # Spectral rolloff - frequency below which a specified percentage of the spectrum is contained
            # Helps characterize the "brightness" of the sound
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate, roll_percent=0.85)
            
            # Spectral bandwidth - weighted standard deviation of frequencies
            # Helps distinguish between "focused" vs "spread out" sounds
            bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
            
            # Spectral flatness - measure of how "noisy" vs "tonal" a sound is
            # Helps distinguish between noisy crunch sounds and smoother sounds
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
        
        # Normalize features (per feature, across time)
        features = (features - np.mean(features, axis=1, keepdims=True)) / (np.std(features, axis=1, keepdims=True) + 1e-10)
        
        return features
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

# Custom dataset for audio files with oversampling for minority classes
class AudioDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = torch.FloatTensor(self.features[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label

# Function to load dataset
def load_dataset():
    features = []
    labels = []
    food_classes = []
    file_paths = []  # Store file paths for debugging and analysis
    
    # Get all food categories
    for food_category in os.listdir(DATA_DIR):
        food_path = os.path.join(DATA_DIR, food_category)
        if os.path.isdir(food_path):
            food_classes.append(food_category)
    
    food_classes.sort()  # Ensure consistent ordering
    
    # Verify if all expected food categories are found
    missing_categories = set(EXPECTED_FOOD_CLASSES) - set(food_classes)
    extra_categories = set(food_classes) - set(EXPECTED_FOOD_CLASSES)
    
    if missing_categories:
        print(f"Warning: Expected food categories not found: {missing_categories}")
        print("Make sure all category folders exist in the WAV_Clips directory.")
    
    if extra_categories:
        print(f"Note: Found additional food categories: {extra_categories}")
    class_to_index = {cls: i for i, cls in enumerate(food_classes)}
    
    print(f"Found {len(food_classes)} food categories: {food_classes}")
    
    # Count files per category
    file_counts = {}
    for food_category in food_classes:
        food_path = os.path.join(DATA_DIR, food_category)
        wav_files = [f for f in os.listdir(food_path) if f.endswith('.wav')]
        file_counts[food_category] = len(wav_files)
    
    print("Files per category:")
    for category, count in file_counts.items():
        print(f"  - {category}: {count} files")
    
    # Extract features for each audio file
    for food_category in food_classes:
        food_path = os.path.join(DATA_DIR, food_category)
        print(f"Processing {food_category} files...")
        
        for filename in os.listdir(food_path):
            if filename.endswith('.wav'):
                file_path = os.path.join(food_path, filename)
                file_paths.append(file_path)
                
                # Extract enhanced features if enabled
                audio_features = extract_enhanced_features(file_path, use_enhanced=USE_ENHANCED_FEATURES)
                
                if audio_features is not None:
                    features.append(audio_features)
                    labels.append(class_to_index[food_category])
    
    # Convert to numpy arrays
    features = np.array(features)
    labels = np.array(labels)
    
    return features, labels, food_classes, file_paths

# Improved model architecture with more layers and batch normalization
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

# Training function with class weights
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device, food_classes=None):
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # For early stopping
    patience = 15  # Increased patience
    early_stop_counter = 0
    
    # Track per-class accuracy
    class_correct = {cls: 0 for cls in range(len(food_classes))}
    class_total = {cls: 0 for cls in range(len(food_classes))}
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Reset per-class tracking for this epoch
        for cls in class_correct:
            class_correct[cls] = 0
            class_total[cls] = 0
        
        train_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for inputs, labels in train_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Track per-class accuracy
            for cls in range(len(food_classes)):
                cls_indices = (labels == cls)
                if cls_indices.sum() > 0:
                    class_total[cls] += cls_indices.sum().item()
                    class_correct[cls] += ((predicted == cls) & cls_indices).sum().item()
            
            # Update progress bar
            train_bar.set_postfix(loss=loss.item(), acc=correct/total)
        
        # Calculate epoch statistics
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # Print per-class training accuracy
        print("\nTraining Accuracy by class:")
        for cls in range(len(food_classes)):
            if class_total[cls] > 0:
                cls_acc = class_correct[cls] / class_total[cls]
                print(f"  - {food_classes[cls]}: {cls_acc:.4f} ({class_correct[cls]}/{class_total[cls]})")
            else:
                print(f"  - {food_classes[cls]}: N/A (no samples)")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Reset per-class tracking for validation
        for cls in class_correct:
            class_correct[cls] = 0
            class_total[cls] = 0
        
        with torch.no_grad():
            val_bar = tqdm.tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for inputs, labels in val_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Track per-class accuracy
                for cls in range(len(food_classes)):
                    cls_indices = (labels == cls)
                    if cls_indices.sum() > 0:
                        class_total[cls] += cls_indices.sum().item()
                        class_correct[cls] += ((predicted == cls) & cls_indices).sum().item()
                
                # Update progress bar
                val_bar.set_postfix(loss=loss.item(), acc=val_correct/val_total)
        
        # Calculate validation statistics
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_correct / val_total
        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc)
        
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")
        print(f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")
        
        # Print per-class validation accuracy
        print("\nValidation Accuracy by class:")
        for cls in range(len(food_classes)):
            if class_total[cls] > 0:
                cls_acc = class_correct[cls] / class_total[cls]
                print(f"  - {food_classes[cls]}: {cls_acc:.4f} ({class_correct[cls]}/{class_total[cls]})")
            else:
                print(f"  - {food_classes[cls]}: N/A (no samples)")
        
        # Save the best model
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"New best model saved with validation accuracy: {best_val_acc:.4f}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            
        # Early stopping
        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        # Learning rate adjustment
        if (epoch + 1) % 10 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9
                print(f"Adjusted learning rate to {param_group['lr']}")
    
    return train_losses, val_losses, train_accs, val_accs

# Plot training history
def plot_history(train_losses, val_losses, train_accs, val_accs):
    # Plot accuracy
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Calculate and print per-class accuracy
    print("\nPer-class accuracy from confusion matrix:")
    class_acc = cm.diagonal() / cm.sum(axis=1)
    for i, cls in enumerate(classes):
        print(f"  - {cls}: {class_acc[i]:.4f}")

# Evaluate the model
def evaluate_model(model, test_loader, criterion, device, food_classes):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    # Track per-class predictions
    class_predictions = {cls: [] for cls in food_classes}
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Store predictions for each class
            for i, label in enumerate(labels.cpu().numpy()):
                class_name = food_classes[label]
                predicted_class = food_classes[predicted[i].item()]
                class_predictions[class_name].append(predicted_class)
    
    # Calculate test statistics
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = correct / total
    
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    # Display what each class is predicted as
    print("\nClass prediction breakdown:")
    for cls, predictions in class_predictions.items():
        if predictions:
            counter = Counter(predictions)
            print(f"\n{cls} samples were predicted as:")
            for pred_cls, count in counter.most_common():
                percentage = count / len(predictions) * 100
                print(f"  - {pred_cls}: {count}/{len(predictions)} ({percentage:.1f}%)")
    
            # Special analysis for potentially problematic classes
    problematic_classes = ['Fries', 'Rice', 'Peanut Butter']
    for food in problematic_classes:
        if food in food_classes:
            food_idx = food_classes.index(food)
            food_samples = [i for i, label in enumerate(all_labels) if label == food_idx]
            if food_samples:
                food_predictions = [food_classes[all_preds[i]] for i in food_samples]
                print(f"\nAnalysis of {food} predictions:")
                print(f"When actual class was {food}, predicted as: {Counter(food_predictions).most_common()}")
    
    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_preds, food_classes)
    
    # Generate classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=food_classes))
    
    return test_loss, test_acc

# Export model to TorchScript for mobile deployment
def export_model(model, input_shape):
    model.eval()
    example_input = torch.rand(1, input_shape[0], input_shape[1]).to(DEVICE)
    traced_script_module = torch.jit.trace(model, example_input)
    traced_script_module.save("food_audio_classifier.pt")
    print("Model exported to TorchScript format as 'food_audio_classifier.pt'")

# Main function
def main():
    print("Loading and preprocessing dataset...")
    features, labels, food_classes, file_paths = load_dataset()
    
    print(f"Feature shape: {features.shape}")
    
    # Calculate and apply class weights
    if USE_CLASS_WEIGHTS:
        class_counts = np.bincount(labels)
        total_samples = len(labels)
        class_weights = total_samples / (len(class_counts) * class_counts)
        # Boost weights for potentially challenging classes
        for food in ['Fries', 'Rice', 'Peanut Butter']:
            if food in food_classes:
                food_idx = food_classes.index(food)
                class_weights[food_idx] *= 1.5  # Extra weight for potentially challenging classes
                print(f"  - Boosted weight for {food}: {class_weights[food_idx].item():.4f}")
        class_weights = torch.FloatTensor(class_weights).to(DEVICE)
        print("Using class weights:")
        for i, weight in enumerate(class_weights):
            print(f"  - {food_classes[i]}: {weight.item():.4f}")
    else:
        class_weights = None
    
    # Split the dataset into training, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(features, labels, test_size=TEST_SPLIT, random_state=42, stratify=labels)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=VALIDATION_SPLIT, random_state=42, stratify=y_train_val)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Print class distribution
    print("\nClass distribution:")
    print("Training set:")
    for cls in range(len(food_classes)):
        count = (y_train == cls).sum()
        print(f"  - {food_classes[cls]}: {count} samples")
    
    print("Validation set:")
    for cls in range(len(food_classes)):
        count = (y_val == cls).sum()
        print(f"  - {food_classes[cls]}: {count} samples")
    
    print("Test set:")
    for cls in range(len(food_classes)):
        count = (y_test == cls).sum()
        print(f"  - {food_classes[cls]}: {count} samples")
    
    # Create datasets and loaders
    train_dataset = AudioDataset(X_train, y_train)
    val_dataset = AudioDataset(X_val, y_val)
    test_dataset = AudioDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Initialize the model
    input_shape = X_train.shape[1:]  # (features, time_steps)
    hidden_dim = 128
    layer_dim = 2  # Number of LSTM layers
    output_dim = len(food_classes)
    
    model = EnhancedAudioModel(input_shape[0], hidden_dim, layer_dim, output_dim).to(DEVICE)
    
    # Print model summary
    print("\nModel architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Define loss function and optimizer
    if USE_CLASS_WEIGHTS:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Use Adam optimizer with weight decay to prevent overfitting
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # Train the model
    print("\nTraining the model...")
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, EPOCHS, DEVICE, food_classes
    )
    
    # Plot training history
    plot_history(train_losses, val_losses, train_accs, val_accs)
    
    # Load the best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Evaluate on test set
    print("\nEvaluating the model...")
    evaluate_model(model, test_loader, criterion, DEVICE, food_classes)
    
    # Analyze potential confusions between similar foods
    print("\nAnalyzing potential confusions between similar food types:")
    similar_pairs = [
        ('Fries', 'Chips'),
        ('Rice', 'Peanut Butter'),
        ('Rice', 'Sandwich'),
        ('Banana', 'Apple'),
        ('Peanut Butter', 'Sandwich')
    ]
    
    for food1, food2 in similar_pairs:
        if food1 in food_classes and food2 in food_classes:
            food1_idx = food_classes.index(food1)
            food2_idx = food_classes.index(food2)
            
            # Check confusions from test set predictions
            print(f"Analyzing confusion between {food1} and {food2}...")
            # This was an incomplete try statement - removed and simplified
    
    # Export the model for mobile deployment
    export_model(model, input_shape)
    
    # Save the model and class mapping for later use
    torch.save({
        'model_state_dict': model.state_dict(),
        'food_classes': food_classes,
        'input_shape': input_shape,
        'enhanced_features': USE_ENHANCED_FEATURES
    }, 'food_audio_classifier_full.pth')
    
    print("Model saved as 'food_audio_classifier_full.pth'")
    
    # Function to predict from a single WAV file
    def predict_single_file(file_path, model, food_classes, device):
        model.eval()
        features = extract_enhanced_features(file_path, use_enhanced=USE_ENHANCED_FEATURES)
        if features is None:
            return None, 0.0
        
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(features_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = food_classes[predicted.item()]
        confidence_value = confidence.item() * 100
        print(f"Predicted food: {predicted_class} (Confidence: {confidence_value:.2f}%)")
        
        # Show top 3 predictions
        top_values, top_indices = torch.topk(probabilities, 3, dim=1)
        print("Top 3 predictions:")
        for i in range(3):
            idx = top_indices[0][i].item()
            value = top_values[0][i].item() * 100
            print(f"{i+1}. {food_classes[idx]}: {value:.2f}%")
        
        return predicted_class, confidence_value
    
    print("\nExample usage for prediction:")
    print("predict_single_file('path/to/your/audio.wav', model, food_classes, DEVICE)")

if __name__ == "__main__":
    main()