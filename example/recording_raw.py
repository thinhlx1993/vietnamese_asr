import pyaudio
import os
import io
import time
import threading
import queue
import wave
import subprocess
from pydub import AudioSegment
import tempfile

# Constants for recording
record_format = pyaudio.paInt16  # 16-bit per sample
sample_rate = 16000  # Output sample rate (16000 Hz)
chunk = 1024  # Record in chunks of 512 samples
channels = 1
record_seconds = 10  # Duration of each recording

audio = pyaudio.PyAudio()
os.makedirs("data", exist_ok=True)

# Open the audio stream for recording
p = pyaudio.PyAudio()
stream = p.open(format=record_format,  # audio format
                channels=channels,  # number of channels (1 for mono)
                rate=sample_rate_in,  # sample rate
                input=True,  # input stream
                frames_per_buffer=chunk)  # buffer size

# Queue to hold audio buffers for transcription
audio_queue = queue.Queue(maxsize=1)

# Save the audio buffer as a .wav file in-memory
def save_wav_buffer(audio_buffer, filename="temp.wav"):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(record_format))
        wf.setframerate(sample_rate_in)  # Input sample rate (44100 Hz)
        wf.writeframes(audio_buffer.read())

# Convert the WAV file to 16000Hz using pydub
def convert_and_resample_wav(wav_filename, target_sample_rate):
    # Load the file using pydub
    audio_segment = AudioSegment.from_wav(wav_filename)

    # Convert to the desired sample rate
    audio_segment = audio_segment.set_frame_rate(target_sample_rate)

    return audio_segment

# Function to save audio buffer to WAV file and send it for transcription
def save_and_transcribe_audio():
    while True:
        # Wait for the next audio buffer
        audio_buffer = audio_queue.get()
        if audio_buffer is None:  # Sentinel to exit the thread
            break

        # Generate filename based on current time
        timestamp = int(time.time())
        wav_filename = f"data/recorded_audio_{timestamp}.wav"

        # Save the audio buffer to a temporary WAV file
        save_wav_buffer(audio_buffer, wav_filename)

        try:
            # Convert and resample to 16000 Hz
            audio_segment = convert_and_resample_wav(wav_filename, sample_rate_out)

            # Save the resampled audio to file
            audio_segment.export(wav_filename, format="wav")

            # Send the WAV file for transcription using the API
            command = [
                "curl", "--location", "https://api.voicesplitter.com/api/v1/uploads",
                "--form", f'file=@"{wav_filename}"'
            ]

            # Run the command and capture the output
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Process transcription result
            unicode_escape = result.stdout.decode('utf-8')
            # Decode the unicode escape sequences (e.g., "\u01b0" becomes "ư")
            decoded_text = unicode_escape.encode('utf-8').decode('unicode_escape')

            if len(unicode_escape) < 10:
                os.remove(wav_filename)
            elif "1 per 1 second" in decoded_text:
                print("Server overload")
                os.remove(wav_filename)
            else:
                print(f"Transcription for {wav_filename}: {decoded_text}")

        except Exception as e:
            print(f"Error processing audio file {wav_filename}: {e}")
            os.remove(wav_filename)  # Cleanup on error

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

        # # Put the buffer in the queue for saving and transcription
        audio_queue.put(audio_buffer)

        # # Pause briefly between recordings if desired (optional)
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
