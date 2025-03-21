import collections
import contextlib
import sys
import os
import wave
import pyaudio

import webrtcvad

import pyaudio

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


class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

#Audio configuration
SAMPLE_RATE = 48000 # 48 kHz sample rate
FRAME_DURATION_MS = 30  # 30 ms frames
CHANNELS = 1 # Stereo audio
FORMAT = pyaudio.paInt16  # 16-bit PCM

FRAME_SIZE = int(SAMPLE_RATE * (FRAME_DURATION_MS / 1000))  # 480 samples

# Initialize PyAudio
p = pyaudio.PyAudio()

mic1_index = 1

# Open stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,  # Enable input for recording
                input_device_index=mic1_index,
                frames_per_buffer=FRAME_SIZE)  # Buffer size matches frame size

def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad):
    """Filters out non-voiced audio frames.
    
    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.

    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.

    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.

    Arguments:

    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).

    Returns: A generator that yields PCM audio data.
    """
    print(f"Recording audio in {FRAME_DURATION_MS} ms frames...")
            
    sample_rate = SAMPLE_RATE
    frame_duration_ms = FRAME_DURATION_MS
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0

    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    while stream.is_active():
        audio_frame = stream.read(FRAME_SIZE, exception_on_overflow=False)
        frame = Frame(audio_frame, timestamp, duration)
        timestamp += duration
        
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
                # sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
         #if triggered:
            # sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
    # sys.stdout.write('\n')
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])

    # Close stream and terminate PyAudio
    stream.stop_stream()
    stream.close()
    p.terminate()
