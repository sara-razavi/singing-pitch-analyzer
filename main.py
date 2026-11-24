import wave
import numpy as np
import matplotlib.pyplot as plt

# about 46 ms
FRAME_SIZE = 2048
HOP = 1024


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
    # int16 overflowed when squaring
    samples = samples / 32768
    return samples, rate, channels, width


def split_frames(samples):
    frames = []
    for start in range(0, len(samples) - FRAME_SIZE, HOP):
        frames.append(samples[start:start + FRAME_SIZE])
    return np.array(frames)


def get_volume(frames):
    return np.sqrt(np.mean(frames ** 2, axis=1))


def find_silent_parts(volume, rate, min_length=0.3):
    # about -30 dB
    limit = np.percentile(volume, 95) * 0.03
    parts = []
    start = None
    for i in range(len(volume)):
        t = i * HOP / rate
        if volume[i] < limit:
            if start == None:
                start = t
        elif start != None:
            if t - start >= min_length:
                parts.append((start, t))
            start = None
    return parts


def detect_pitch(frame, rate, low=70, high=1000):
    frame = frame - np.mean(frame)
    corr = np.correlate(frame, frame, mode="full")
    corr = corr[len(frame) - 1:]
    if corr[0] == 0:
        return 0
    corr = corr / corr[0]

    min_lag = int(rate / high)
    max_lag = int(rate / low)
    lag = min_lag + np.argmax(corr[min_lag:max_lag])

    # noise or breathing
    if corr[lag] < 0.5:
        return 0
    return rate / lag


def get_pitches(frames, volume, rate):
    limit = np.percentile(volume, 95) * 0.03
    pitches = []
    for i in range(len(frames)):
        if volume[i] < limit:
            pitches.append(0)
        else:
            pitches.append(detect_pitch(frames[i], rate))
    return np.array(pitches)


def plot_waveform(samples, rate, silent_parts, title):
    time = np.arange(len(samples)) / rate

    plt.figure(figsize=(12, 4))
    # a whole song is millions of points and the window was really slow,
    # every 10th sample is enough to see the shape
    plt.plot(time[::10], samples[::10], linewidth=0.5)
    for part in silent_parts:
        plt.axvspan(part[0], part[1], color="gray", alpha=0.3)
    plt.title("Waveform: " + title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()


def plot_spectrogram(frames, rate, title):
    window = np.hanning(FRAME_SIZE)
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1))
    db = 20 * np.log10(spectrum + 0.000001)
    end = len(frames) * HOP / rate

    plt.figure(figsize=(12, 5))
    plt.imshow(db.T, origin="lower", aspect="auto", extent=[0, end, 0, rate / 2], cmap="magma", vmin=db.max() - 80)
    # voice is under 4000 Hz
    plt.ylim(0, 4000)
    plt.colorbar(label="dB")
    plt.title("Spectrogram: " + title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
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

frames = split_frames(samples)
volume = get_volume(frames)

silent_parts = find_silent_parts(volume, rate)
silent_time = 0
print()
print("Silent parts:")
for part in silent_parts:
    print(" ", round(part[0], 2), "to", round(part[1], 2), "s")
    silent_time += part[1] - part[0]
print("Silence:", round(silent_time, 2), "s (" + str(round(silent_time / length * 100, 1)) + "% of the recording)")

pitches = get_pitches(frames, volume, rate)
voiced = pitches[pitches > 0]
print()
print("Pitch found in", len(voiced), "of", len(frames), "frames")
print("Average pitch:", round(np.mean(voiced), 1), "Hz")
print("Lowest pitch:", round(np.min(voiced), 1), "Hz")
print("Highest pitch:", round(np.max(voiced), 1), "Hz")

plot_waveform(samples, rate, silent_parts, filename)
plot_spectrogram(frames, rate, filename)
