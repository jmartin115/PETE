import threading
import sounddevice as sd
import caller_transcriber
import operator_transcriber
import whisper
import webrtcvad
import re
import os
import numpy as np
import scipy.io.wavfile as wav
from functools import partial


# Replace with actual device IDs from `sd.query_devices()`
MIC_DEVICE = 1  # Your Microphone Array (AMD Audio Dev)

SAMPLE_RATE = 16000 
CHANNELS = 2

# Define callback function to process incoming audio
def callback(indata, frames, time, status):
    if status:
        print(status)  # Print any stream errors
    print(indata)  # Process or analyze the audio data here

# Function to handle microphone input and transcription for the caller mic
def handle_microphone_input_caller():
    model = whisper.load_model("base")
    vad = webrtcvad.Vad(int(3))

    # Set up the audio parameters
    sample_rate = 16000  # Standard sample rate (can be adjusted)
    channels = 2  # Mono audio (can be adjusted)

    # Select the device
    device_id = 11

    if device_id is None:
        print(f"Device not found.")
    else:
        # Set up the audio stream
        
        # Wrap the callback to include extra arguments
        callback_with_extra_args = partial(caller_transcriber.audio_callback, model=model, vad=vad)

        # Use the wrapped callback
        with sd.OutputStream(callback=partial(caller_transcriber.audio_callback, model=model, vad=vad),
                       device=device_id,
                       channels=channels, 
                       samplerate=sample_rate,
                       dtype=np.float32,
                       latency='high',):
            print("Streaming audio and processing 30 ms chunks indefinitely...")
            sd.sleep(1000000)  # Keep the stream open indefinitely




# Main function to start the threads for both microphones
def main():

    # Starting two threads for two microphones (selecting by ID)
    mic2_thread = threading.Thread(target=handle_microphone_input_caller)  # Using caller mic

    
    mic2_thread.start()
    
    # Join threads to wait for their completion
    
    mic2_thread.join()

if __name__ == "__main__":
    main()

