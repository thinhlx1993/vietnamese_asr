import subprocess

command = [
    "curl", "--location", "https://api.voicesplitter.com/api/v1/uploads",
    "--form", 'file=@"/path/to/your/wav_file.wav"'
]

result = subprocess.run(command, capture_output=True, text=True)
print(result.stdout.encode('utf-8').decode('unicode_escape'))