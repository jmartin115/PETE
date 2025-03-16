import sounddevice as sd
import numpy as np
import contextlib
import wave
import webrtcvad
import collections
import numpy as np
from scipy.io.wavfile import write
import whisper

def transcribe_speech(audio, model):
    # Load your audio file
    audio_file = audio

    # Transcribe the audio
    result = model.transcribe(audio_file)
    return result
    # Output the transcribed text
    # print("Transcription:")
    # print(result['text'])
def read_wave(path):
    """Reads a .wav file.

    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 2
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate == 48000
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, 48000
    

def write_wave(path, audio, sample_rate):
    """Writes a .wav file.

    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)

import numpy as np
from scipy.io.wavfile import write

def frames_to_wav(frames, model, sample_rate=44100):
    """Converts a list of Frame objects into a .wav file."""
    # Reconstruct audio data from the frames
    audio_data = b''.join([frame.bytes for frame in frames])  # Combine all frame bytes
    output_file = 'current_caller_chunk.wav'
    # Convert the audio data back into numpy array (assuming int16 format)
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    
    # Write the audio data to a .wav file
    write(output_file, sample_rate, audio_array)
    result = transcribe_speech(output_file, model)


# Frame class to store audio data in chunks
class Frame(object):
    """Represents a 'frame' of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

    def __repr__(self):
        return f"Frame(timestamp={self.timestamp}, duration={self.duration}, bytes={len(self.bytes)})"

#Audio configuration
SAMPLE_RATE = 16000 # 48 kHz sample rate
FRAME_DURATION_MS = 30  # 30 ms frames
CHANNELS = 2 # Stereo audio

FRAME_SIZE = int(SAMPLE_RATE * 0.03)  # 480 samples for 30ms at 16kHz # 480 samples

# Callback function to handle incoming audio data
def audio_callback(indata, frames, time, status, model, vad):
    
    if status:
        print(status, flush=True)

    sample_rate = SAMPLE_RATE
    frame_duration_ms = FRAME_DURATION_MS
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    padding_duration_ms = 1500
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []

    # Chunk size for 30 ms (based on the sample rate)
    chunk_size = int(sample_rate * 0.03)  # 30 ms
    
    # Get the current timestamp (the time at which this chunk was received)
    timestamp = time.inputBufferAdcTime  # Timestamp from the sounddevice callback
    
    # Process incoming audio in chunks of 30 ms
    num_chunks = frames // chunk_size  # How many 30 ms chunks in the current frame
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size

        # Convert stereo to mono (if needed)
        if indata.shape[1] > 1:  
            chunk = indata.mean(axis=1)  # Average across channels
        else:
            chunk = indata[:, 0]  # Use first channel if already mono

        # Convert float32 PCM [-1, 1] to int16 PCM [-32768, 32767]
        chunk = (chunk * 32768).astype(np.int16)


        
        # Create a Frame object for the chunk
        frame = Frame(bytes=chunk.tobytes(), timestamp=timestamp, duration=0.03)

        
        # Process or store the Frame (for example, print it or append it to a list)
        is_speech = vad.is_speech(frame.bytes, sample_rate)
        
        # Process the audio frame (e.g., save, analyze, or stream it)
        # print(f"Captured frame of size {len(audio_frame)} bytes")
         # sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                # sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                frames_to_wav(voiced_frames, model)
                ring_buffer.clear()
                voiced_frames = []
    frames_to_wav(voiced_frames, model)

