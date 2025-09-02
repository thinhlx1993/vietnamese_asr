import time
import numpy as np
import torch
import nemo.collections.asr as nemo_asr
import torchaudio
import os
import sys
import threading
import queue
from datetime import datetime

from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig
from respeaker_mic import ReSpeakerMic

sample_rate = 16000
chunk = 1024  # Record in chunks of 1024 samples
record_seconds = 10  # Duration of each recording

# Suppress NeMo and PyTorch progress bars
os.environ['TQDM_DISABLE'] = '1'

# Create transcription and recording folders if they don't exist
transcription_dir = "transcription"
recording_dir = "recording"
os.makedirs(transcription_dir, exist_ok=True)
os.makedirs(recording_dir, exist_ok=True)

# Initialize ReSpeaker microphone
print("Initializing ReSpeaker microphone...", flush=True)
respeaker_mic = ReSpeakerMic(rate=sample_rate, frames_size=chunk)

# Load the ASR model from a .nemo file
print("Loading Vietnamese ASR model...", flush=True)
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
print("Model loaded successfully!", flush=True)

def get_daily_filename():
    """Get the filename for today's transcription file"""
    today = datetime.now()
    return f"{today.strftime('%d%m%Y')}.txt"

def save_transcription(text, audio_data, cycle_count):
    """Save transcription to today's file with integer timestamp and save WAV file"""
    filename = get_daily_filename()
    filepath = os.path.join(transcription_dir, filename)
    
    # Get current timestamp as integer (Unix timestamp)
    timestamp = int(time.time())
    
    # Format: timestamp|transcription text
    line = f"{timestamp}|{text}\n"
    
    # Append to file
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(line)
    
    # Save WAV file with timestamp-only filename
    wav_filename = f"{timestamp}.wav"
    wav_filepath = os.path.join(recording_dir, wav_filename)
    
    # Save the audio data as WAV file using the ReSpeakerMic method
    respeaker_mic.save_voice_channel_wav(wav_filepath)
    
    print(f"Cycle {cycle_count}: Audio saved to {wav_filename}", flush=True)

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

# Global queue for audio processing (limit size to prevent memory issues)
audio_queue = queue.Queue(maxsize=3)
processing_active = threading.Event()

def background_transcription_worker():
    """Background worker thread for processing transcriptions"""
    while processing_active.is_set():
        try:
            # Get audio data from queue with timeout
            audio_item = audio_queue.get(timeout=1.0)
            if audio_item is None:  # Shutdown signal
                break
                
            cycle_count, all_audio_data, waveform_data = audio_item
            
            print(f"Cycle {cycle_count}: Processing audio...", flush=True)
            
            # Transcribe the recorded audio
            transcription = transcribe_audio(waveform_data)
            
            # Extract and display only the transcription text
            if transcription and len(transcription) > 0:
                text = transcription[0].text
                if text.strip() and len(text) > 10:
                    print(f"Cycle {cycle_count}: TRANSCRIPTION: {text}", flush=True)
                    # Save transcription to daily file
                    save_transcription(text, all_audio_data, cycle_count)
                else:
                    print(f"Cycle {cycle_count}: No speech detected", flush=True)
            else:
                print(f"Cycle {cycle_count}: Transcription failed", flush=True)
            
            print(f"Cycle {cycle_count}: Completed", flush=True)
            print("-" * 30, flush=True)
            
            audio_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in background transcription: {e}", flush=True)

# Start background transcription worker
print("Starting background transcription worker...", flush=True)
processing_active.set()
transcription_thread = threading.Thread(target=background_transcription_worker, daemon=True)
transcription_thread.start()

# Start ReSpeaker microphone
print("Starting ReSpeaker microphone...", flush=True)
respeaker_mic.start()
print("ReSpeaker microphone started successfully!", flush=True)
print(f"Recording {record_seconds}-second audio chunks with background processing...", flush=True)
print("=" * 50, flush=True)

try:
    cycle_count = 0
    while True:
        cycle_count += 1
        print(f"Cycle {cycle_count}: Recording {record_seconds} seconds...", flush=True)
        
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
                
                # Add to processing queue (non-blocking)
                try:
                    audio_queue.put((cycle_count, all_audio_data, waveform_data), block=False)
                    print(f"Cycle {cycle_count}: Audio queued for background processing", flush=True)
                except queue.Full:
                    print(f"Cycle {cycle_count}: Processing queue full, skipping this audio", flush=True)
                    
            else:
                print(f"Cycle {cycle_count}: No audio data captured", flush=True)

except KeyboardInterrupt:
    print("Stopping service...", flush=True)

finally:
    # Stop background processing
    print("Stopping background transcription worker...", flush=True)
    processing_active.clear()
    
    # Add shutdown signal to queue
    try:
        audio_queue.put(None, block=False)
    except queue.Full:
        pass
    
    # Wait for transcription thread to finish (with timeout)
    transcription_thread.join(timeout=5.0)
    
    # Stop ReSpeaker microphone completely
    print("Stopping ReSpeaker microphone...", flush=True)
    respeaker_mic.stop()
    print("Service stopped.", flush=True)
