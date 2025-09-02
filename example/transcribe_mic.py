import time
import numpy as np
import torch
import nemo.collections.asr as nemo_asr
import torchaudio
import os
from datetime import datetime

from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig
from respeaker_mic import ReSpeakerMic

sample_rate = 16000
chunk = 1024  # Record in chunks of 1024 samples
record_seconds = 10  # Duration of each recording

# Suppress NeMo and PyTorch progress bars
os.environ['TQDM_DISABLE'] = '1'

# Create transcription folder if it doesn't exist
transcription_dir = "example/transcription"
os.makedirs(transcription_dir, exist_ok=True)

# Initialize ReSpeaker microphone
respeaker_mic = ReSpeakerMic(rate=sample_rate, frames_size=chunk)

# Load the ASR model from a .nemo file
model = nemo_asr.models.ASRModel.restore_from("models/vietnamese_asr_confome_ctc.nemo")
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

def get_daily_filename():
    """Get the filename for today's transcription file"""
    today = datetime.now()
    return f"{today.strftime('%d%m%Y')}.txt"

def save_transcription(text):
    """Save transcription to today's file with timestamp"""
    filename = get_daily_filename()
    filepath = os.path.join(transcription_dir, filename)
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Format: timestamp|transcription text
    line = f"{timestamp}|{text}\n"
    
    # Append to file
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(line)

def transcribe_audio(waveform_data):
    """Transcribe waveform data"""
    # Convert to tensor and add batch dimension
    waveform = torch.tensor(waveform_data).unsqueeze(0)

    # Resample to 16kHz if needed
    if sample_rate != model.sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=model.sample_rate)
        waveform = resampler(waveform)

    # Transcribe the audio and return result
    transcription = model.transcribe([waveform.squeeze(0).numpy()], verbose=False)
    return transcription

# Start ReSpeaker microphone
respeaker_mic.start()

print("Start ReSpeaker microphone")

try:
    while True:
        # Clear previous recording data
        respeaker_mic.clear_audio_buffer()
        
        # Record for the specified duration
        time.sleep(record_seconds)
        
        # Get all recorded audio as a single waveform
        with respeaker_mic.lock:
            if respeaker_mic.recording_frames:
                # Combine all frames into one audio buffer
                all_audio_data = b''.join(respeaker_mic.recording_frames)
                audio_array = np.frombuffer(all_audio_data, dtype='int16')
                
                # Extract voice channel
                if respeaker_mic.channels > 1:
                    voice_channel_data = audio_array[respeaker_mic.voice_channel::respeaker_mic.channels]
                else:
                    voice_channel_data = audio_array
                
                # Normalize to float32 [-1, 1]
                waveform_data = voice_channel_data.astype(np.float32) / 32768.0
                
                # Transcribe the recorded audio
                transcription = transcribe_audio(waveform_data)
                
                # Extract and display only the transcription text
                if transcription and len(transcription) > 0:
                    text = transcription[0].text
                    if text.strip() and len(text) > 1:
                        print(text)
                        # Save transcription to daily file
                        save_transcription(text)

except KeyboardInterrupt:
    pass

finally:
    # Stop ReSpeaker microphone completely
    respeaker_mic.stop()
