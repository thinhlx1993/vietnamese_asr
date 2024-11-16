import pyaudio
import os
import io
import time
import threading
import queue
import numpy as np
import wave
import subprocess

# Constants for recording
format = pyaudio.paInt16  # 16-bit per sample
sample_rate = 16000
chunk = 1024  # Record in chunks of 1024 samples
channels = 1
record_seconds = 5  # Duration of each recording

audio = pyaudio.PyAudio()
stream = audio.open(format=format, channels=channels, rate=sample_rate, input=True, frames_per_buffer=chunk)

# Queue to hold audio buffers for transcription
audio_queue = queue.Queue()

# Function to save audio buffer to WAV file and send it for transcription
def save_and_transcribe_audio():
    while True:
        # Wait for the next audio buffer
        audio_buffer = audio_queue.get()
        if audio_buffer is None:  # Sentinel to exit the thread
            break
        
        # Generate filename based on current time
        timestamp = int(time.time())
        wav_filename = f"recorded_audio_{timestamp}.wav"

        # Save the audio buffer as a .wav file
        with wave.open(wav_filename, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(format))
            wf.setframerate(sample_rate)
            wf.writeframes(audio_buffer.read())
        
        # Now, send the WAV file for transcription using the API
        command = [
            "curl", "--location", "https://api.voicesplitter.com/api/v1/uploads",
            "--form", f'file=@"{wav_filename}"'
        ]

        # Run the command and capture the output
        result = subprocess.run(command, capture_output=True, text=True)
        
        # Print the transcription result
        print(f"Transcription for {wav_filename}:")
        unicode_escape = result.stdout.encode('utf-8').decode('unicode_escape')
        print(unicode_escape)
        if len(unicode_escape) < 10:
            os.remove(wav_filename)

        # Mark this task as done
        audio_queue.task_done()

# Start the transcription thread
transcription_thread = threading.Thread(target=save_and_transcribe_audio, daemon=True)
transcription_thread.start()

try:
    while True:
        # Record audio to an in-memory buffer
        audio_buffer = io.BytesIO()
        frames = []

        for _ in range(0, int(sample_rate / chunk * record_seconds)):
            data = stream.read(chunk)
            frames.append(data)

        # Write to buffer and add to the queue
        audio_buffer.write(b''.join(frames))
        audio_buffer.seek(0)

        # Put the buffer in the queue for saving and transcription
        audio_queue.put(audio_buffer)

        # Pause briefly between recordings if desired (optional)
        # time.sleep(1)  # Adjust the delay as needed

except KeyboardInterrupt:
    print("Stopping recording...")

finally:
    # Stop recording and terminate resources
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Signal the transcription thread to exit
    audio_queue.put(None)
    transcription_thread.join()
