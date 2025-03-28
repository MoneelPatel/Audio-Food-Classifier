import os
import wave
import numpy as np
from pydub import AudioSegment
from pathlib import Path

def split_wav_file(input_path, output_base_dir, clip_length_seconds=5):
    """
    Split a WAV file into smaller clips of specified length.
    
    Args:
        input_path (str): Path to the input WAV file
        output_base_dir (str): Base directory to save the output clips
        clip_length_seconds (int): Length of each clip in seconds
    """
    # Get the filename without extension
    filename = os.path.basename(input_path)
    name_without_ext = os.path.splitext(filename)[0]
    
    # Create a subfolder named after the food (original filename)
    food_subfolder = os.path.join(output_base_dir, name_without_ext)
    os.makedirs(food_subfolder, exist_ok=True)
    
    # Load the audio file
    audio = AudioSegment.from_wav(input_path)
    
    # Get the total duration in milliseconds
    total_duration = len(audio)
    clip_length_ms = clip_length_seconds * 1000
    
    # Split the audio file into clips
    clip_number = 1
    for start_ms in range(0, total_duration, clip_length_ms):
        # Calculate end time (make sure it doesn't exceed the audio length)
        end_ms = min(start_ms + clip_length_ms, total_duration)
        
        # Extract the clip
        clip = audio[start_ms:end_ms]
        
        # Define the output filename
        output_filename = f"{name_without_ext}_{clip_number}.wav"
        output_path = os.path.join(food_subfolder, output_filename)
        
        # Export the clip
        clip.export(output_path, format="wav")
        
        print(f"Created clip: {output_path}")
        clip_number += 1

def main():
    # Define the input directory
    input_dir = "wav"
    output_base_dir = "wav_clips"
    
    # Create the base output directory if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Find all WAV files in the input directory
    wav_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith('.wav')]
    
    if not wav_files:
        print(f"No WAV files found in {input_dir}")
        return
    
    # Process each WAV file
    for wav_file in wav_files:
        print(f"Processing: {wav_file}")
        split_wav_file(wav_file, output_base_dir)
        print(f"Finished processing: {wav_file}")
    
    print(f"All files processed. Clips saved in subfolders under {output_base_dir}")

if __name__ == "__main__":
    main()