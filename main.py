import wave
import numpy as np
import matplotlib.pyplot as plt


def read_wav(filename):
    file = wave.open(filename, "rb")
    channels = file.getnchannels()
    width = file.getsampwidth()
    rate = file.getframerate()
    count = file.getnframes()
    raw = file.readframes(count)
    file.close()

    # 16 bit samples (-32768 to 32767)
    samples = np.frombuffer(raw, dtype=np.int16)
    return samples, rate, channels, width


def plot_waveform(samples, rate, title):
    time = np.arange(len(samples)) / rate

    plt.figure(figsize=(12, 4))
    # a whole song is millions of points and the window was really slow,
    # every 10th sample is enough to see the shape
    plt.plot(time[::10], samples[::10], linewidth=0.5)
    plt.title("Waveform: " + title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()


print("Singing Pitch Analyzer")
print("----------------------")

filename = input("Which recording do you want to analyze? ")
path = "recordings/" + filename

samples, rate, channels, width = read_wav(path)
length = len(samples) / rate

print()
print("File:", path)
print("Channels:", channels)
print("Sample rate:", rate, "Hz")
print("Sample width:", width * 8, "bit")
print("Length:", round(length, 2), "seconds")

plot_waveform(samples, rate, filename)
