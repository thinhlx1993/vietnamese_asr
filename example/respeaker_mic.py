#!/usr/bin/env python3

import threading
import sys
import time
import numpy as np
import pyaudio
import wave
import io
import queue
from pixel_ring import pixel_ring

pixel_ring.off()  # Turn off the LED


class ReSpeakerMic:
    def __init__(self, rate=16000, frames_size=1024):
        self.rate = rate
        self.frames_size = frames_size
        self.channels = 6  # ReSpeaker 4 Mic Array with 6_channels_firmware.bin
        # Channel mapping for 6_channels_firmware.bin (official Seeed documentation):
        # Channel 0: processed audio for ASR (Voice channel)
        # Channel 1-4: 4 microphones' raw data  
        # Channel 5: playback
        self.voice_channel = 0  # Channel 0 contains processed audio for ASR
        print("Using 6_channels_firmware.bin (6 channels)")
        print("Channel 0: Processed audio for ASR (Voice channel)")
        print("Channel 1-4: Raw microphone data")
        print("Channel 5: Playback data")
        
        self.pyaudio_instance = pyaudio.PyAudio()
        self.stop_event = threading.Event()
        
        # Find the ReSpeaker device
        device_index = None
        print("Searching for ReSpeaker 4 Mic Array device...")
        for i in range(self.pyaudio_instance.get_device_count()):
            dev = self.pyaudio_instance.get_device_info_by_index(i)
            name = dev['name']
            input_channels = dev['maxInputChannels']
            print(f"Device {i}: {name} with {input_channels} input channels")
            if 'ReSpeaker 4 Mic Array' in name and input_channels >= self.channels:
                device_index = i
                print(f"Found ReSpeaker device at index {i}")
                break
        
        if device_index is None:
            # List all available input devices for debugging
            print("\nAvailable input devices:")
            for i in range(self.pyaudio_instance.get_device_count()):
                dev = self.pyaudio_instance.get_device_info_by_index(i)
                if dev['maxInputChannels'] > 0:
                    print(f"  {i}: {dev['name']} (channels: {dev['maxInputChannels']})")
            
            raise RuntimeError(
                "ReSpeaker 4 Mic Array device not found! "
                "Please ensure the device is connected and has proper permissions. "
                "Check the device list above for available input devices."
            )
        
        # Open audio stream
        self.stream = self.pyaudio_instance.open(
            start=False,
            format=pyaudio.paInt16,
            input_device_index=device_index,
            channels=self.channels,
            rate=int(self.rate),
            frames_per_buffer=int(self.frames_size),
            stream_callback=self._callback,
            input=True
        )
        
        self.audio_data = []
        self.lock = threading.Lock()
        self.recording_frames = []
        self.is_recording = False
    
    def _callback(self, in_data, frame_count, time_info, status):
        if self.is_recording:
            with self.lock:
                self.audio_data.append(in_data)
                self.recording_frames.append(in_data)
        return None, pyaudio.paContinue
    
    def start(self):
        """Start audio capture"""
        self.stream.start_stream()
        self.stop_event.clear()
        self.is_recording = True
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_recording(self):
        """Stop recording but keep PyAudio instance alive"""
        self.is_recording = False
        self.stop_event.set()
        self.stream.stop_stream()
    
    def stop(self):
        """Stop everything and terminate PyAudio"""
        self.stop_recording()
        self.stream.close()
        self.pyaudio_instance.terminate()
    
    def get_voice_channel_audio(self):
        """Get the latest voice channel audio (channel 0 - processed for ASR) as a buffer"""
        with self.lock:
            if not self.audio_data:
                return None
            
            # Get the latest audio data
            latest_data = self.audio_data[-1]
            audio_array = np.frombuffer(latest_data, dtype='int16')
            
            # Extract only the voice channel
            if self.channels > 1:
                voice_channel_data = audio_array[self.voice_channel::self.channels]
            else:
                voice_channel_data = audio_array
            
            # Convert to buffer
            buffer = io.BytesIO()
            buffer.write(voice_channel_data.tobytes())
            buffer.seek(0)
            
            return buffer
    
    def get_voice_channel_waveform(self):
        """Get the latest voice channel audio (channel 0 - processed for ASR) as normalized waveform"""
        with self.lock:
            if not self.audio_data:
                return None
            
            # Get the latest audio data
            latest_data = self.audio_data[-1]
            audio_array = np.frombuffer(latest_data, dtype='int16')
            
            # Extract only the voice channel
            if self.channels > 1:
                voice_channel_data = audio_array[self.voice_channel::self.channels]
            else:
                voice_channel_data = audio_array
            
            # Normalize to float32 [-1, 1]
            waveform = voice_channel_data.astype(np.float32) / 32768.0
            
            return waveform
    
    def get_raw_microphone_data(self, mic_channel=1):
        """
        Get raw microphone data from channels 1-4
        Args:
            mic_channel: Microphone channel (1-4) to extract
        Returns:
            Normalized waveform from the specified microphone channel
        """
        if mic_channel < 1 or mic_channel > 4:
            raise ValueError("Microphone channel must be between 1 and 4")
            
        with self.lock:
            if not self.audio_data:
                return None
            
            # Get the latest audio data
            latest_data = self.audio_data[-1]
            audio_array = np.frombuffer(latest_data, dtype='int16')
            
            # Extract the specified microphone channel (channels 1-4 are raw mic data)
            mic_channel_data = audio_array[mic_channel::self.channels]
            
            # Normalize to float32 [-1, 1]
            waveform = mic_channel_data.astype(np.float32) / 32768.0
            
            return waveform
    
    def clear_audio_buffer(self):
        """Clear the audio buffer to free memory"""
        with self.lock:
            self.audio_data = []
            self.recording_frames = []
    
    def save_to_wav(self, filename, channels=6, sample_rate=16000):
        """Save recorded audio to WAV file"""
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)  # 16-bit audio
            wav_file.setframerate(sample_rate)
            
            for frame in self.recording_frames:
                wav_file.writeframes(frame)
        
        return filename
    
    def save_voice_channel_wav(self, filename, sample_rate=16000):
        """Save only the voice channel (channel 0 - processed audio for ASR) as mono WAV file"""
        # Extract only the voice channel from each frame
        voice_frames = []
        for frame in self.recording_frames:
            # Convert frame to numpy array
            audio_array = np.frombuffer(frame, dtype='int16')
            
            # Extract only channel 0 (voice channel)
            if self.channels > 1:
                voice_channel_data = audio_array[self.voice_channel::self.channels]
            else:
                voice_channel_data = audio_array
            
            # Convert back to bytes
            voice_frame = voice_channel_data.tobytes()
            voice_frames.append(voice_frame)
        
        # Save mono WAV file
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit audio
            wav_file.setframerate(sample_rate)
            
            for frame in voice_frames:
                wav_file.writeframes(frame)
        
        return filename
    
    def _monitor(self):
        """Monitor audio levels from voice channel only"""
        while not self.stop_event.is_set():
            time.sleep(0.1)  # Check every 100ms
            
            with self.lock:
                if self.audio_data:
                    # Process the latest audio data
                    latest_data = self.audio_data[-1]
                    audio_array = np.frombuffer(latest_data, dtype='int16')
                    
                    # Calculate RMS level for voice channel only
                    if self.channels > 1:
                        voice_channel_data = audio_array[self.voice_channel::self.channels]
                    else:
                        voice_channel_data = audio_array
                    
                    if len(voice_channel_data) > 0:
                        rms = np.sqrt(np.mean(voice_channel_data.astype(np.float32)**2))
                        
                        # Keep only last few chunks to avoid memory issues
                        if len(self.audio_data) > 10:
                            self.audio_data = self.audio_data[-10:]
