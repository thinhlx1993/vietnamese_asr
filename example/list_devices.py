
import pyaudio

# Initialize PyAudio instance
p = pyaudio.PyAudio()

# Get the number of available audio devices
device_count = p.get_device_count()

# Open output.txt file in write mode
with open("output.txt", "w") as file:
    file.write(f"Number of devices: {device_count}\n\n")

    # Loop through each device and write its info to the file
    for i in range(device_count):
        device_info = p.get_device_info_by_index(i)
        file.write(f"Device {i}:\n")
        file.write(f"  Name: {device_info['name']}\n")
        file.write(f"  Input Channels: {device_info['maxInputChannels']}\n")
        file.write(f"  Output Channels: {device_info['maxOutputChannels']}\n")
        file.write(f"  Sample Rate: {device_info['defaultSampleRate']}\n")
        file.write(f"  Is Input: {device_info['maxInputChannels'] > 0}\n")
        file.write(f"  Is Output: {device_info['maxOutputChannels'] > 0}\n")
        file.write("="*50 + "\n")

# Close the PyAudio instance
p.terminate()

print("Device list has been written to output.txt.")