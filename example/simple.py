import torchaudio
import nemo.collections.asr as nemo_asr


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

# Specify the path to the .wav file
file_path = "your_file.wav"
transcribes = model.transcribe(audio=[file_path], batch_size=16)
print(transcribes)