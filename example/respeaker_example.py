#!/usr/bin/env python3
"""
Example script demonstrating ReSpeakerMic usage with 6_channels_firmware.bin
Voice channel (channel 0) will be automatically extracted from the 6-channel input
"""

from respeaker_mic import ReSpeakerMic
import time

def main():
    print("ReSpeaker Microphone Example - 6 Channels Firmware")
    print("=" * 50)
    print("This example works with 6_channels_firmware.bin")
    print("Voice channel (channel 0) will be extracted automatically")
    print()
    
    # Initialize ReSpeakerMic for 6-channel firmware
    print("Initializing ReSpeaker microphone...")
    mic = ReSpeakerMic(rate=16000, frames_size=1024)
    
    try:
        # Start recording
        print("\nStarting recording...")
        mic.start()
        
        # Record for 5 seconds
        print("Recording for 5 seconds...")
        time.sleep(5)
        
        # Stop recording
        print("Stopping recording...")
        mic.stop_recording()
        
        # Save recordings
        print("\nSaving recordings...")
        
        # Save full 6-channel recording
        full_filename = mic.save_to_wav("full_6channel_recording.wav")
        print(f"Saved full 6-channel recording: {full_filename}")
        
        # Save voice channel only (mono) - this is channel 0 from the 6 channels
        voice_filename = mic.save_voice_channel_wav("voice_channel_mono.wav")
        print(f"Saved voice channel (mono): {voice_filename}")
        
        print("\nRecording completed successfully!")
        print(f"Voice channel extracted from {mic.channels}-channel input")
        
    except KeyboardInterrupt:
        print("\nRecording interrupted by user")
    except Exception as e:
        print(f"Error during recording: {e}")
    finally:
        # Clean up
        mic.stop()
        print("Microphone stopped and cleaned up")

if __name__ == "__main__":
    main()
