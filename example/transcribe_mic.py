import pyaudio
import io
import time
import threading
import queue
import numpy as np
import torch
import nemo.collections.asr as nemo_asr
import torchaudio

from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig

format = pyaudio.paInt16  # 16-bit per sample
sample_rate = 16000
chunk = 1024  # Record in chunks of 1024 samples
channels = 1
record_seconds = 5  # Duration of each recording

audio = pyaudio.PyAudio()
stream = audio.open(format=format, channels=channels, rate=sample_rate, input=True, frames_per_buffer=chunk)

# Load the ASR model from a .nemo file
model = nemo_asr.models.ASRModel.restore_from("./vietnamese_asr_confome_ctc.nemo")
model.sample_rate = sample_rate

decoding_cfg = CTCDecodingConfig()
RESULTS_DIR = "models"
decoding_cfg.strategy = "flashlight"
decoding_cfg.beam.search_type = "flashlight"
decoding_cfg.beam.kenlm_path = f'{RESULTS_DIR}/interpolated_lm_vi.bin'
decoding_cfg.beam.flashlight_cfg.lexicon_path=f'{RESULTS_DIR}/interpolated_lm_vi.lexicon'
decoding_cfg.beam.beam_size = 32
decoding_cfg.beam.beam_alpha = 0.2
decoding_cfg.beam.beam_beta = 0.2
decoding_cfg.beam.flashlight_cfg.beam_size_token = 32
decoding_cfg.beam.flashlight_cfg.beam_threshold = 25.0

model.change_decoding_strategy(decoding_cfg)


# Queue to hold audio buffers for transcription
audio_queue = queue.Queue()

def transcribe_audio():
    while True:
        # Wait for the next audio buffer
        audio_buffer = audio_queue.get()
        if audio_buffer is None:  # Sentinel to exit the thread
            break

        # Convert buffer to waveform format expected by the model
        waveform = np.frombuffer(audio_buffer.read(), dtype=np.int16).astype(np.float32) / 32768.0  # Normalize
        waveform = torch.tensor(waveform).unsqueeze(0)  # Add batch dimension

        # Resample to 16kHz if needed
        if sample_rate != model.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=model.sample_rate)
            waveform = resampler(waveform)

        # Transcribe the audio and print the result
        transcription = model.transcribe([waveform.squeeze(0).numpy()])
        print("Transcription:", transcription)

        # Mark this task as done
        audio_queue.task_done()

# Start the transcription thread
transcription_thread = threading.Thread(target=transcribe_audio, daemon=True)
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

        # Put the buffer in the queue for transcription
        audio_queue.put(audio_buffer)

        # Pause briefly between recordings if desired (optional)
        time.sleep(1)  # Adjust the delay as needed

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
