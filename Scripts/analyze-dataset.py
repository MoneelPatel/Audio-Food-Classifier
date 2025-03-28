import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Configuration
DATA_DIR = 'WAV_Clips'
SAMPLE_RATE = 22050
DURATION = 5  # seconds per clip
N_MFCC = 40
N_FFT = 2048
HOP_LENGTH = 512

# Expected food categories - updated to include new foods
EXPECTED_FOOD_CLASSES = [
    'Apple', 'Banana', 'Carrot', 'Chips', 'Fries', 
    'Grape', 'Peanut Butter', 'Popcorn', 'Rice', 'Sandwich'
]

def extract_features(file_path, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH):
    try:
        # Load audio file
        audio, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # Ensure consistent length
        if len(audio) < SAMPLE_RATE * DURATION:
            audio = np.pad(audio, (0, SAMPLE_RATE * DURATION - len(audio)))
        else:
            audio = audio[:SAMPLE_RATE * DURATION]
        
        # Extract various features
        result = {}
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        result['mfccs'] = mfccs
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)[0]
        result['spectral_centroid'] = np.mean(spectral_centroid)
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)[0]
        result['spectral_bandwidth'] = np.mean(spectral_bandwidth)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)[0]
        result['spectral_rolloff'] = np.mean(spectral_rolloff)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio, frame_length=n_fft, hop_length=hop_length)[0]
        result['zero_crossing_rate'] = np.mean(zcr)
        
        # RMS energy
        rms = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=hop_length)[0]
        result['rms_energy'] = np.mean(rms)
        
        # Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
        result['mel_spectrogram'] = mel_spectrogram
        
        return result
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

def analyze_dataset():
    # Get all food categories
    food_classes = []
    for food_category in os.listdir(DATA_DIR):
        food_path = os.path.join(DATA_DIR, food_category)
        if os.path.isdir(food_path):
            food_classes.append(food_category)
    
    food_classes.sort()
    print(f"Found {len(food_classes)} food categories: {food_classes}")
    
    # Verify if all expected food categories are found
    missing_categories = set(EXPECTED_FOOD_CLASSES) - set(food_classes)
    extra_categories = set(food_classes) - set(EXPECTED_FOOD_CLASSES)
    
    if missing_categories:
        print(f"Warning: Expected food categories not found: {missing_categories}")
        print("Make sure all category folders exist in the WAV_Clips directory.")
    
    if extra_categories:
        print(f"Note: Found additional food categories: {extra_categories}")
        
    # Add analysis for potential acoustic similarities
    print("\nAnalyzing potential acoustic similarities between food categories...")
    potential_similar_pairs = [
        ('Fries', 'Chips'),
        ('Rice', 'Peanut Butter'),
        ('Rice', 'Sandwich'),
        ('Banana', 'Apple'),
        ('Peanut Butter', 'Sandwich')
    ]
    
    # Verify if all expected food categories are found
    missing_categories = set(EXPECTED_FOOD_CLASSES) - set(food_classes)
    extra_categories = set(food_classes) - set(EXPECTED_FOOD_CLASSES)
    
    if missing_categories:
        print(f"Warning: Expected food categories not found: {missing_categories}")
        print("Make sure all category folders exist in the WAV_Clips directory.")
    
    if extra_categories:
        print(f"Note: Found additional food categories: {extra_categories}")
    
    # Count files per category
    file_counts = {}
    for food_category in food_classes:
        food_path = os.path.join(DATA_DIR, food_category)
        wav_files = [f for f in os.listdir(food_path) if f.endswith('.wav')]
        file_counts[food_category] = len(wav_files)
    
    print("Files per category:")
    for category, count in file_counts.items():
        print(f"  - {category}: {count} files")
        
    # Check for data imbalance
    min_count = min(file_counts.values()) if file_counts else 0
    max_count = max(file_counts.values()) if file_counts else 0
    
    if max_count > min_count * 1.5:  # If any category has 50% more samples than the smallest
        print("\nWarning: Dataset is imbalanced. Consider:")
        print("  - Adding more samples to smaller categories")
        print("  - Using data augmentation for underrepresented classes")
        print("  - Applying class weights during training")
    
    # Plot file distribution
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(file_counts.keys()), y=list(file_counts.values()))
    plt.xlabel('Food Category')
    plt.ylabel('Number of Files')
    plt.title('File Distribution per Category')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('file_distribution.png')
    plt.close()
    
    # Extract features for visualization
    features_data = []
    labels = []
    mfccs_data = []
    
    print("Extracting features for visualization...")
    for food_category in food_classes:
        food_path = os.path.join(DATA_DIR, food_category)
        wav_files = [f for f in os.listdir(food_path) if f.endswith('.wav')]
        
        # Limit to max 10 files per category for visualization
        sample_files = wav_files[:10]
        
        for filename in sample_files:
            file_path = os.path.join(food_path, filename)
            features = extract_features(file_path)
            
            if features:
                # Store scalar features for PCA/t-SNE
                feature_vector = [
                    features['spectral_centroid'],
                    features['spectral_bandwidth'],
                    features['spectral_rolloff'],
                    features['zero_crossing_rate'],
                    features['rms_energy']
                ]
                features_data.append(feature_vector)
                labels.append(food_category)
                
                # Store MFCCs for later visualization
                mfccs_data.append((food_category, features['mfccs'], features['mel_spectrogram']))
    
    # Convert to numpy arrays
    features_data = np.array(features_data)
    labels = np.array(labels)
    
    # Analyze feature similarity between categories
    if len(features_data) > 0:
        feature_means = {}
        unique_categories = np.unique(labels)
        
        for category in unique_categories:
            # Category is already a string (like 'Apple'), not an index
            indices = labels == category
            category_features = features_data[indices]
            feature_means[category] = np.mean(category_features, axis=0)
        
        # Calculate feature similarity between pairs
        print("\nFeature similarity between potentially similar food pairs:")
        for food1, food2 in potential_similar_pairs:
            if food1 in feature_means and food2 in feature_means:
                # Calculate cosine similarity
                f1 = feature_means[food1]
                f2 = feature_means[food2]
                similarity = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))
                print(f"  - {food1} vs {food2}: Similarity = {similarity:.4f} (higher means more similar sounds)")
    
    # Visualize feature distributions
    feature_names = [
        'Spectral Centroid', 
        'Spectral Bandwidth', 
        'Spectral Rolloff', 
        'Zero Crossing Rate', 
        'RMS Energy'
    ]
    
    # Create a DataFrame for easier plotting
    df = pd.DataFrame(features_data, columns=feature_names)
    df['Food Category'] = labels
    
    print("Generating feature distribution plots...")
    # Box plots for each feature by category
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(feature_names):
        plt.subplot(2, 3, i+1)
        sns.boxplot(x='Food Category', y=feature, data=df)
        plt.xticks(rotation=45)
        plt.title(f'{feature} by Food Category')
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png')
    plt.close()
    
    # Pairplot of features
    print("Generating pairplot...")
    g = sns.pairplot(df, hue='Food Category', height=2.5)
    g.fig.suptitle('Pairwise Feature Relationships', y=1.02)
    plt.savefig('feature_pairplot.png')
    plt.close()
    
    # Dimensionality reduction for visualization
    print("Performing dimensionality reduction...")
    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features_data)
    
    plt.figure(figsize=(10, 8))
    for category in np.unique(labels):
        indices = labels == category
        plt.scatter(pca_result[indices, 0], pca_result[indices, 1], label=category)
    
    plt.title('PCA of Audio Features')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.tight_layout()
    plt.savefig('pca_visualization.png')
    plt.close()
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(features_data)
    
    plt.figure(figsize=(10, 8))
    for category in np.unique(labels):
        indices = labels == category
        plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1], label=category)
    
    plt.title('t-SNE of Audio Features')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.tight_layout()
    plt.savefig('tsne_visualization.png')
    plt.close()
    
    # Sample spectrograms and MFCCs
    print("Generating sample spectrograms and MFCCs...")
    # Randomly select one example per category
    selected_indices = {}
    for i, (category, _, _) in enumerate(mfccs_data):
        if category not in selected_indices:
            selected_indices[category] = i
    
    # Plot MFCCs and Mel spectrograms
    for category, idx in selected_indices.items():
        _, mfccs, mel_spec = mfccs_data[idx]
        
        plt.figure(figsize=(12, 8))
        
        # Plot MFCC
        plt.subplot(2, 1, 1)
        librosa.display.specshow(mfccs, x_axis='time', sr=SAMPLE_RATE)
        plt.colorbar()
        plt.title(f'MFCC - {category}')
        
        # Plot Mel spectrogram
        plt.subplot(2, 1, 2)
        librosa.display.specshow(
            librosa.power_to_db(mel_spec, ref=np.max), 
            y_axis='mel', 
            x_axis='time', 
            sr=SAMPLE_RATE
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Mel Spectrogram - {category}')
        
        plt.tight_layout()
        plt.savefig(f'spectrogram_{category}.png')
        plt.close()
    
    print("\nAnalysis complete! Generated visualizations:")
    print("- file_distribution.png")
    print("- feature_distributions.png")
    print("- feature_pairplot.png")
    print("- pca_visualization.png")
    print("- tsne_visualization.png")
    for category in selected_indices:
        print(f"- spectrogram_{category}.png")

if __name__ == "__main__":
    analyze_dataset()