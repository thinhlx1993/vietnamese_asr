import time
import numpy as np
import torch
import nemo.collections.asr as nemo_asr
import torchaudio
import os
import sys
import threading
import queue
import wave
import pyaudio
from datetime import datetime

from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig

sample_rate = 16000
chunk = 1024  # Record in chunks of 1024 samples
MIN_RECORD_SECONDS = 10  # Keep recording for at least 10 seconds
VAD_CHECK_INTERVAL = 1.0  # Check VAD every 1 second

# PyAudio configuration
RESPEAKER_RATE = 16000
RESPEAKER_WIDTH = 2  # 16-bit audio
RESPEAKER_INDEX = 1  # Input device index (run getDeviceInfo.py to get index)
# Note: Even with 1_channel_firmware.bin, the device may report 6 channels
# Channel 0 contains the processed audio for ASR

# Import USB tuning module for VAD (required)
import usb.core
import usb.util
from tuning import Tuning

# Suppress NeMo and PyTorch progress bars
os.environ['TQDM_DISABLE'] = '1'

# Create transcription and recording folders if they don't exist
transcription_dir = "transcription"
recording_dir = "recording"
os.makedirs(transcription_dir, exist_ok=True)
os.makedirs(recording_dir, exist_ok=True)

# Initialize PyAudio
print("Initializing PyAudio...", flush=True)
p = pyaudio.PyAudio()

# Find ReSpeaker device and detect channel count
device_index = None
device_channels = None
respeaker_found = False

print(f"Searching for ReSpeaker device...", flush=True)
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    name = dev['name']
    input_channels = dev['maxInputChannels']
    print(f"Device {i}: {name} (max input channels: {input_channels})", flush=True)
    if 'ReSpeaker' in name:
        respeaker_found = True
        if input_channels > 0:
            # Found ReSpeaker with input channels
            device_index = i
            device_channels = input_channels
            print(f"Found ReSpeaker device at index {i}: {name}", flush=True)
            print(f"Device supports {input_channels} input channel(s)", flush=True)
            break
        else:
            print(f"Found ReSpeaker device at index {i} but it has 0 input channels (output only)", flush=True)

# If ReSpeaker found but has no input channels, try PulseAudio or default device
if respeaker_found and (device_index is None or device_channels is None or device_channels <= 0):
    print("ReSpeaker device found but has no direct input access.", flush=True)
    print("Trying PulseAudio or default device (which should route to ReSpeaker)...", flush=True)
    
    # Try PulseAudio device (usually device 6)
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        name = dev['name']
        input_channels = dev['maxInputChannels']
        if ('pulse' in name.lower() or 'default' in name.lower()) and input_channels > 0:
            device_index = i
            device_channels = input_channels
            print(f"Using {name} at index {i} ({input_channels} input channels)", flush=True)
            print("Note: This should route to ReSpeaker through PulseAudio", flush=True)
            break

# Fallback to default index if still not found
if device_index is None or device_channels is None or device_channels <= 0:
    print(f"Trying default device index {RESPEAKER_INDEX}...", flush=True)
    try:
        dev = p.get_device_info_by_index(RESPEAKER_INDEX)
        input_channels = dev['maxInputChannels']
        if input_channels > 0:
            device_index = RESPEAKER_INDEX
            device_channels = input_channels
            print(f"Using device at index {RESPEAKER_INDEX}: {dev['name']} ({input_channels} channels)", flush=True)
        else:
            print(f"Device at index {RESPEAKER_INDEX} has no input channels", flush=True)
    except Exception as e:
        print(f"Error accessing device at index {RESPEAKER_INDEX}: {e}", flush=True)

# Final validation
if device_index is None:
    print("ERROR: Could not find a suitable input device.", flush=True)
    sys.exit(1)

if device_channels is None or device_channels <= 0:
    print(f"WARNING: Device reports {device_channels} channels. Using default configuration...", flush=True)
    # Try common ReSpeaker configurations
    device_channels = 6  # Default to 6 channels (most common for ReSpeaker)
    print(f"Will attempt to use {device_channels} channels", flush=True)

# Determine actual channel count to use
# Try 1 channel first (for 1_channel_firmware.bin), fallback to device's max channels
actual_channels = None
voice_channel = 0  # Channel 0 contains processed audio for ASR

print(f"Attempting to determine optimal channel count (device reports {device_channels} channels)...", flush=True)

# Try opening with 1 channel first
try:
    test_stream = p.open(
        rate=RESPEAKER_RATE,
        format=p.get_format_from_width(RESPEAKER_WIDTH),
        channels=1,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=chunk
    )
    test_stream.stop_stream()
    test_stream.close()
    actual_channels = 1
    print("Successfully opened test stream with 1 channel", flush=True)
except Exception as e:
    print(f"Could not open test stream with 1 channel: {e}", flush=True)
    # Try with device's max channels (typically 6)
    if device_channels >= 6:
        actual_channels = 6
        print(f"Will use 6 channels (will extract channel 0 for ASR)", flush=True)
    elif device_channels >= 1:
        actual_channels = device_channels
        print(f"Will use {actual_channels} channels (device max)", flush=True)
    else:
        print(f"ERROR: Device reports invalid channel count: {device_channels}", flush=True)
        sys.exit(1)

# Final validation before opening stream
if actual_channels is None or actual_channels <= 0:
    print(f"ERROR: Invalid channel count ({actual_channels}). Cannot proceed.", flush=True)
    print(f"Debug info: device_channels={device_channels}, actual_channels={actual_channels}", flush=True)
    sys.exit(1)

# Open PyAudio stream with detected channel count
print(f"Opening audio stream (channels={actual_channels}, rate={RESPEAKER_RATE})...", flush=True)
stream = p.open(
    rate=RESPEAKER_RATE,
    format=p.get_format_from_width(RESPEAKER_WIDTH),
    channels=actual_channels,
    input=True,
    input_device_index=device_index,
    frames_per_buffer=chunk
)
print("PyAudio stream opened successfully!", flush=True)

# Initialize USB VAD (required)
print("Initializing USB VAD...", flush=True)
dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
if dev is None:
    print("ReSpeaker USB device not found. Please ensure the device is connected.", flush=True)
    sys.exit(1)

try:
    mic_tuning = Tuning(dev)
    mic_tuning.set_vad_threshold(15)
    print("* USB VAD initialized successfully", flush=True)
except Exception as e:
    print(f"* Failed to initialize USB VAD: {e}", flush=True)
    sys.exit(1)

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

def save_transcription(text, cycle_count, timestamp_str=None):
    """Save transcription to today's file with integer timestamp"""
    filename = get_daily_filename()
    filepath = os.path.join(transcription_dir, filename)
    
    # Get current timestamp as integer (Unix timestamp)
    timestamp = int(time.time())
    
    # Format: timestamp|transcription text
    line = f"{timestamp}|{text}\n"
    
    # Append to file
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(line)
    
    print(f"Cycle {cycle_count}: Transcription saved", flush=True)

def save_audio_data_as_wav(audio_data, filepath, sample_rate=16000):
    """Save raw audio data as mono WAV file"""
    # Convert audio_data to numpy array
    audio_array = np.frombuffer(audio_data, dtype='int16')
    
    # Save as mono WAV file
    with wave.open(filepath, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit audio
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_array.tobytes())
    
    return filepath

def check_voice_activity():
    """Check if voice is detected using USB VAD"""
    try:
        return mic_tuning.is_voice()
    except Exception as e:
        print(f"* USB VAD error: {e}", flush=True)
        return 0

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
                if text.strip() and len(text) >=0:
                    print(f"Cycle {cycle_count}: TRANSCRIPTION: {text}", flush=True)
                    # Save transcription to daily file
                    save_transcription(text, cycle_count)
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

# Start PyAudio stream
print("Starting audio stream...", flush=True)
stream.start_stream()
print("Audio stream started successfully!", flush=True)
print("* Ready — waiting for voice...", flush=True)
print("=" * 50, flush=True)

try:
    cycle_count = 0
    recording = False
    record_start_time = None
    last_vad_check = 0
    recording_frames = []
    
    while True:
        current_time = time.time()
        
        # Check VAD every second when not recording
        if not recording:
            if current_time - last_vad_check >= VAD_CHECK_INTERVAL:
                vad = check_voice_activity()
                last_vad_check = current_time
                
                if vad == 1:
                    print("* Voice detected — start recording...", flush=True)
                    recording = True
                    record_start_time = current_time
                    recording_frames = []  # Clear previous frames
                else:
                    # Not recording, just sleep and continue
                    time.sleep(0.1)
                    continue
        
        # When recording, continuously read audio data
        elif recording:
            # Read audio chunk from stream
            try:
                data = stream.read(chunk, exception_on_overflow=False)
                recording_frames.append(data)
            except Exception as e:
                print(f"* Error reading audio: {e}", flush=True)
                continue
            
            # Record for minimum duration
            elapsed_time = current_time - record_start_time
            
            if elapsed_time < MIN_RECORD_SECONDS:
                # Still recording minimum duration
                continue
            else:
                # Minimum duration reached, stop recording
                print("* Voice ended and min duration reached — stop recording.", flush=True)
                
                cycle_count += 1
                
                # Generate unique filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(recording_dir, f"recording_{timestamp}.wav")
                
                # Process recorded frames
                if recording_frames:
                    # Combine all frames into one audio buffer
                    all_audio_data = b''.join(recording_frames)
                    audio_array = np.frombuffer(all_audio_data, dtype='int16')
                    
                    # Extract voice channel (channel 0) if multi-channel
                    if actual_channels > 1:
                        voice_channel_data = audio_array[voice_channel::actual_channels]
                    else:
                        voice_channel_data = audio_array
                    
                    # Calculate actual recorded duration
                    total_samples = len(voice_channel_data)
                    calculated_duration = total_samples / sample_rate
                    
                    print(f"* Audio buffer contains {calculated_duration:.2f} seconds of data", flush=True)
                    
                    # Save audio file as mono (voice channel only)
                    wf = wave.open(filename, 'wb')
                    wf.setnchannels(1)  # Save as mono
                    wf.setsampwidth(2)  # 16-bit audio
                    wf.setframerate(sample_rate)
                    wf.writeframes(voice_channel_data.tobytes())
                    wf.close()
                    
                    print(f"* Saved: {filename}", flush=True)
                    
                    # Normalize to float32 [-1, 1] for transcription
                    waveform_data = voice_channel_data.astype(np.float32) / 32768.0
                    
                    # Add to processing queue (non-blocking)
                    try:
                        audio_queue.put((cycle_count, all_audio_data, waveform_data), block=False)
                        print(f"* Transcription started in background for: {filename}", flush=True)
                    except queue.Full:
                        print(f"* Processing queue full, skipping transcription for: {filename}", flush=True)
                else:
                    print(f"* No audio data captured", flush=True)
                
                recording = False
                last_vad_check = current_time
                recording_frames = []  # Clear frames for next recording

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
    
    # Stop PyAudio stream
    print("Stopping audio stream...", flush=True)
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("Service stopped.", flush=True)
