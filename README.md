# The best Vietnamese speech recognition using Conforme-CTC 2024

[![PWC](https://avatars.githubusercontent.com/u/1728152?s=48&v=4)](https://github.com/NVIDIA/NeMo)

# Training data

I collect data from many different sources for training. The training data contains over 10k hours of speech data from the sources below

- Common Voice dataset
- (AN4) database audio files
- Vietnamese Speech recognition
- Youtube public dataset
- Data collecting from public space

# Models setup

Tokenizer SentencePieceTokenizer initialized with 128 tokens

121 M Total params

| Name              | Type                              | Params
|---|---|--- |
0 | preprocessor      | AudioToMelSpectrogramPreprocessor | 0     
1 | encoder           | ConformerEncoder                  | 121 M 
2 | decoder           | ConvASRDecoder                    | 66.2 K
3 | loss              | CTCLoss                           | 0     
4 | spec_augmentation | SpectrogramAugmentation           | 0     
5 | wer               | WER                               | 0     


# Benchmark WER result


| | WER | CER |
|---|---|--- |
|without ngram LM| 10.71 | 12.21
|with ngram LM| 9.15 | 10.2

# How to use

Download model weight here

https://drive.google.com/drive/folders/1SVNibfeMshfVkmatIU90LYok_Mf0zMD0?usp=sharing

Install Nemo Frameworks

https://github.com/NVIDIA/NeMo

You can try demo in the example folder

I created a free-to-use API server to submit the inference data

```python
import subprocess

command = [
    "curl", "--location", "https://api.voicesplitter.com/api/v1/uploads",
    "--form", 'file=@"/path/to/your/wav_file.wav"'
]

result = subprocess.run(command, capture_output=True, text=True)
print(result.stdout.encode('utf-8').decode('unicode_escape'))
```

# Contact

thinhle.ict@gmail.com | [Thinh Le's LinkedIn Profile](https://www.linkedin.com/in/thinhleict/)