import numpy as np
import torch
import nemo.collections.asr as nemo_asr
import torchaudio
import threading
import queue
import time
import subprocess
import ffmpeg
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig

class RTSPAudioProcessor:
    def __init__(self, rtsp_url, sample_rate=16000):
        self.rtsp_url = rtsp_url
        self.sample_rate = sample_rate
        self.audio_queue = queue.Queue()
        self.chunk_duration = 5  # seconds

        MODEL_DIR = "models"
        # Load the ASR model
        self.model = nemo_asr.models.ASRModel.restore_from(f"{MODEL_DIR}/vietnamese_asr_medium.nemo")
        self.model.sample_rate = sample_rate

        # Configure decoding
        # decoding_cfg = CTCDecodingConfig()
        
        # decoding_cfg.strategy = "flashlight"
        # decoding_cfg.beam.search_type = "flashlight"
        # decoding_cfg.beam.kenlm_path = f'{MODEL_DIR}/interpolated_lm_vi.bin'
        # decoding_cfg.beam.flashlight_cfg.lexicon_path = f'{MODEL_DIR}/interpolated_lm_vi.lexicon'
        # decoding_cfg.beam.beam_size = 32
        # decoding_cfg.beam.beam_alpha = 0.2
        # decoding_cfg.beam.beam_beta = 0.2
        # decoding_cfg.beam.flashlight_cfg.beam_size_token = 32
        # decoding_cfg.beam.flashlight_cfg.beam_threshold = 25.0
        
        # self.model.change_decoding_strategy(decoding_cfg)
        
        # Initialize stream process
        self.process = None
        self.is_running = False
        
    def start(self):
        """Start processing the RTSP stream"""
        self.is_running = True
        
        # Start the processing threads
        self.audio_thread = threading.Thread(target=self._process_audio)
        self.transcription_thread = threading.Thread(target=self._transcribe_audio)
        
        self.audio_thread.start()
        self.transcription_thread.start()
        
    def stop(self):
        """Stop processing the RTSP stream"""
        self.is_running = False
        
        # Stop ffmpeg process if running
        if self.process:
            try:
                self.process.kill()
            except:
                pass
        
        # Signal the transcription thread to exit
        self.audio_queue.put(None)
        
        # Wait for threads to finish
        if hasattr(self, 'audio_thread'):
            self.audio_thread.join()
        if hasattr(self, 'transcription_thread'):
            self.transcription_thread.join()
            
    def _process_audio(self):
        """Extract audio from the RTSP stream using ffmpeg"""
        try:
            # FFmpeg command to read RTSP stream and output raw audio
            process = (
                ffmpeg
                .input(self.rtsp_url)
                .output('pipe:', 
                       format='f32le',
                       acodec='pcm_f32le',
                       ac=1,
                       ar=str(self.sample_rate),
                       **{
                           'filter:a': f'aresample=async=1:first_pts=0:min_comp=0.001:min_hard_comp=0.100000:osr={self.sample_rate}'
                       })
                .overwrite_output()
                .run_async(pipe_stdout=True, pipe_stderr=True)
            )
            self.process = process
            
            # Calculate buffer size for chunk_duration seconds of audio
            buffer_size = int(self.sample_rate * self.chunk_duration * 4)  # 4 bytes per float32
            
            while self.is_running:
                # Read chunk_duration seconds of audio
                in_bytes = process.stdout.read(buffer_size)
                if not in_bytes:
                    print("End of stream or error")
                    break
                    
                # Convert bytes to numpy array
                audio_chunk = np.frombuffer(in_bytes, dtype=np.float32)
                if len(audio_chunk) == 0:
                    continue
                
                # Normalize audio if needed
                if np.abs(audio_chunk).max() > 1.0:
                    audio_chunk = audio_chunk / np.abs(audio_chunk).max()
                
                # Put in queue for transcription
                self.audio_queue.put(audio_chunk)
                
        except ffmpeg.Error as e:
            print("FFmpeg error:", e.stderr.decode())
        except Exception as e:
            print("Error processing audio:", str(e))
        finally:
            if self.process:
                try:
                    self.process.kill()
                except:
                    pass
            
    def _transcribe_audio(self):
        """Transcribe audio chunks"""
        while self.is_running:
            # Get audio data from queue
            audio_data = self.audio_queue.get()
            if audio_data is None:  # Exit signal
                break
                
            # Convert to tensor
            waveform = torch.tensor(audio_data).unsqueeze(0)
            
            # Transcribe
            with torch.no_grad():
                transcription = self.model.transcribe([waveform.squeeze(0).numpy()])
                print("Transcription:", transcription)
            
            self.audio_queue.task_done()

def main():
    # Replace with your RTSP stream URL
    rtsp_url = "rtsp://100.125.246.48:5554/live/merged"
    
    processor = RTSPAudioProcessor(rtsp_url)
    try:
        processor.start()
        
        # Keep the main thread running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        processor.stop()

if __name__ == "__main__":
    main() 