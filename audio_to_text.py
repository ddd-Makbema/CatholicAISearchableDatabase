import os
import whisper
import logging
import warnings
import numpy as np
from pydub import AudioSegment
import webrtcvad
import time

# Suppress the FutureWarning from PyTorch
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Suppress the UserWarning from Whisper
warnings.filterwarnings("ignore", category=UserWarning, module="whisper")

# Set up logging to log both to console and a file
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(),  # Output to console
                        logging.FileHandler("transcription_log.txt", mode='a')  # Log to a file
                    ])

# Load the Whisper model and log if it loaded successfully
try:
    model = whisper.load_model("base")
    print("Whisper model loaded successfully.")
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    logging.error(f"Error loading Whisper model: {e}")
    exit(1)  # Exit the program if model loading fails

# Function to detect speech using WebRTC VAD
def detect_speech(audio_path):
    vad = webrtcvad.Vad(3)  # Sensitivity (0 - 3), 3 is most aggressive for detecting speech

    # Load the audio file using pydub and convert to mono for better analysis
    try:
        audio = AudioSegment.from_mp3(audio_path).set_channels(1).set_frame_rate(16000)
    except Exception as e:
        logging.error(f"Error loading or processing audio file {audio_path}: {e}")
        print(f"Error loading or processing audio file {audio_path}: {e}")
        return []

    # Limit the processing to the first 6 minutes (360000 ms) of the audio for music removal
    audio = audio[:360000]  # Truncate the audio to 6 minutes (360000 ms)
    
    samples = np.array(audio.get_array_of_samples())

    # Create a generator that yields chunks of the audio in 30 ms intervals
    frame_duration = 30  # milliseconds
    frame_size = int(16000 * frame_duration / 1000)  # 16kHz frame size

    frames = [samples[i:i + frame_size] for i in range(0, len(samples), frame_size)]

    speech_segments = []
    
    # Track time to detect if processing takes too long
    start_time = time.time()

    total_frames = len(frames)  # Total number of frames

    # Detect speech in each frame
    for i, frame in enumerate(frames):
        # Log progress every 500 frames
        if i % 500 == 0:
            logging.info(f"Processing frame {i}/{len(frames)}")

        if len(frame) < frame_size:  # Skip frames that are smaller than the expected size
            logging.warning(f"Skipping frame {i} due to insufficient length.")
            continue

        try:
            is_speech = vad.is_speech(frame.tobytes(), 16000)  # Check if the frame contains speech
            if is_speech:
                speech_segments.append((i * frame_duration, (i + 1) * frame_duration))
        except Exception as e:
            logging.error(f"Error processing frame {i}: {e}")
            continue

        # Check if the processing is taking too long
        if time.time() - start_time > 60:  # If processing takes more than 60 seconds
            logging.error("Processing took too long, exiting loop.")
            break

    return speech_segments

# Function to trim audio to speech segments
def trim_speech_from_audio(audio_path, speech_segments):
    audio = AudioSegment.from_mp3(audio_path)

    segments_to_keep = []
    for start, end in speech_segments:
        start_ms = start
        end_ms = end
        segment = audio[start_ms:end_ms]
        
        # Check if segment is not empty
        if len(segment) > 0:
            segments_to_keep.append(segment)
        else:
            logging.warning(f"Skipping empty segment from {start_ms} to {end_ms}")

    # Concatenate all speech segments into one audio file
    if segments_to_keep:
        trimmed_audio = segments_to_keep[0]
        for i, segment in enumerate(segments_to_keep[1:], start=1):
            logging.info(f"Concatenating segment {i}/{len(segments_to_keep)} of length {len(segment)} ms.")
            trimmed_audio += segment  # Use += to append segments properly
    else:
        trimmed_audio = AudioSegment.empty()  # If no segments, return an empty audio segment

    return trimmed_audio

# Function to transcribe the audio file
def transcribe_audio(file_path):
    full_path = os.path.abspath(file_path)
    print(f"Attempting to access file: {full_path}")  # Debug: print the full file path
    logging.info(f"Attempting to access file: {full_path}")  # Log the full file path

    # Check if the file exists
    if not os.path.exists(full_path):
        logging.error(f"File does not exist: {full_path}")
        print(f"File does not exist: {full_path}")
        return None  # Skip the file if it doesn't exist

    # Check if the file is empty
    if os.path.getsize(full_path) == 0:
        logging.error(f"File is empty: {full_path}")
        print(f"File is empty: {full_path}")
        return None

    # Detect speech in the audio (use the first 6 minutes to remove music)
    speech_segments = detect_speech(full_path)
    if not speech_segments:
        print(f"No speech detected in {full_path}")
        logging.error(f"No speech detected in {full_path}")
        return None

    # Trim the audio to only keep the speech segments
    trimmed_audio = trim_speech_from_audio(full_path, speech_segments)

    # Save the trimmed audio temporarily (optional, for debugging purposes)
    temp_audio_path = "trimmed_audio.wav"
    trimmed_audio.export(temp_audio_path, format="wav")

    # Transcribe the full audio (original MP3 file) using Whisper, not just the trimmed one
    try:
        print(f"Attempting to transcribe: {full_path}")  # Debug: Attempt to transcribe file
        result = model.transcribe(full_path)  # Transcribe the full MP3 file
        os.remove(temp_audio_path)  # Clean up the temporary file (if created)
        return result['text']
    except Exception as e:
        logging.error(f"Error transcribing {full_path}: {e}")
        print(f"Error transcribing {full_path}: {e}")
        return None

# Function to batch transcribe all MP3 files in a directory
def batch_transcribe(directory):
    # Check if the directory exists
    if not os.path.exists(directory):
        logging.error(f"Directory does not exist: {directory}")
        print(f"Directory does not exist: {directory}")
        return

    # Walk through the directory and its subdirectories
    for subdir, _, files in os.walk(directory):
        print(f"Checking directory: {subdir}")  # Debug: check what directory we are scanning

        # Check if there are any files in the directory
        if not files:
            print(f"No files found in directory: {subdir}")

        for filename in files:
            if filename.endswith(".mp3"):
                file_path = os.path.join(subdir, filename)
                print(f"Found MP3 file: {file_path}")  # Debug: check the file found

                # Ensure the file exists before attempting to transcribe
                if not os.path.exists(file_path):
                    logging.error(f"Skipping {file_path} - file does not exist")
                    print(f"Skipping {file_path} - file does not exist")
                    continue

                # Transcribe the audio
                text = transcribe_audio(file_path)

                if text is not None:
                    txt_file_path = file_path.replace(".mp3", ".txt")

                    # Save the transcription in a text file with UTF-8 encoding
                    with open(txt_file_path, "w", encoding="utf-8") as f:
                        f.write(text)

                    # Log the file conversion
                    logging.info(f"Transcribed: {file_path} -> {txt_file_path}")
                    print(f"Transcribed: {file_path} -> {txt_file_path}")
                else:
                    print(f"Skipping transcription for {file_path}, as it failed.")

# Use the function with a parent directory containing subdirectories
batch_transcribe(r"C:\Users\Daniel Keefe\Documents\CatholicAIProject\AudiotoTextAPI\custom\Youtube Homily Downloads")
