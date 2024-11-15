import requests
import os
import time

# Define the file path and URL
# wav file should be in wav format and bitrate is 16K
file_path = 'SpeechData.wav'
url = 'https://api.voicesplitter.com/api/v1/uploads'

# Get the filename from the file path
filename = os.path.basename(file_path)

# Open the file as a stream without reading it into memory
with open(file_path, 'rb') as file_stream:
    files = {
        'file': (filename, file_stream, 'audio/wav')
    }
    
    # Measure the start time
    start_time = time.time()
    
    # Send the POST request with streaming enabled
    response = requests.post(url, files=files, headers={'accept': 'application/json'}, timeout=2)
    
    # Print the results
    print("Status code:", response.status_code)
    print("Response text:", response.text)
    print("Total upload time:", time.time() - start_time, "seconds")