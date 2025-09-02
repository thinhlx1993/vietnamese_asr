import pyaudio
import wave

# Parameters
device_index = 1  # specify the index of the input device you want to use
sample_rate = 44100  # sample rate in Hz
duration = 10  # duration of the recording in seconds
output_filename = "output.wav"  # output filename

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open stream for recording
stream = p.open(format=pyaudio.paInt16,  # audio format
                channels=1,  # number of channels (1 for mono)
                rate=sample_rate,  # sample rate
                input=True,  # input stream
                input_device_index=device_index,  # input device index
                frames_per_buffer=512)  # smaller buffer size

print("Recording...")

# Record for the specified duration
frames = []
for _ in range(0, int(sample_rate / 512 * duration)):
    data = stream.read(512)
    frames.append(data)

# Stop and close the stream
print("Recording finished.")
stream.stop_stream()
stream.close()
p.terminate()

# Save the recorded audio to a file
with wave.open(output_filename, 'wb') as wf:
    wf.setnchannels(1)  # mono audio
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))  # sample width in bytes
    wf.setframerate(sample_rate)  # sample rate
    wf.writeframes(b''.join(frames))  # write frames to file

print(f"Audio saved to {output_filename}")
