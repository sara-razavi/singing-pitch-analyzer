import wave
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# about 46 ms
FRAME_SIZE = 2048
HOP = 1024

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


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
    # on the edge, not a real peak
    if lag == min_lag or lag == max_lag - 1:
        return 0

    # noise or breathing
    if corr[lag] < 0.5:
        return 0

    # peak between two lags
    a = corr[lag - 1]
    b = corr[lag]
    c = corr[lag + 1]
    shift = 0
    if a - 2 * b + c != 0:
        shift = 0.5 * (a - c) / (a - 2 * b + c)
    return rate / (lag + shift)


def get_pitches(frames, volume, rate):
    limit = np.percentile(volume, 95) * 0.03
    pitches = []
    for i in range(len(frames)):
        if volume[i] < limit:
            pitches.append(0)
        else:
            pitches.append(detect_pitch(frames[i], rate))
    return np.array(pitches)


def smooth_pitch(pitches, size=5):
    smooth = np.zeros(len(pitches))
    half = size // 2
    for i in range(len(pitches)):
        if pitches[i] == 0:
            continue
        around = pitches[max(0, i - half):i + half + 1]
        around = around[around > 0]
        smooth[i] = np.median(around)
    return smooth


def freq_to_note(freq):
    # A4 = 69
    midi = 69 + 12 * np.log2(freq / 440)
    nearest = int(round(midi))
    cents = int(round((midi - nearest) * 100))
    octave = int(midi / 12) - 1
    return NOTE_NAMES[nearest % 12] + str(octave), cents


def get_notes(pitches, rate, min_length=0.1):
    parts = []
    current = None
    start = 0
    for i in range(len(pitches)):
        if pitches[i] > 0:
            name = freq_to_note(pitches[i])[0]
        else:
            name = None

        if name != current:
            if current != None:
                parts.append([current, start, i])
            current = name
            start = i

    # last note
    if current != None:
        parts.append([current, start, len(pitches)])

    # slides between notes
    min_frames = min_length * rate / HOP
    long_parts = []
    for p in parts:
        if p[2] - p[1] >= min_frames:
            long_parts.append(p)

    # vibrato can split one note
    joined = []
    for p in long_parts:
        if len(joined) > 0 and joined[-1][0] == p[0] and p[1] - joined[-1][2] < 0.2 * rate / HOP:
            joined[-1][2] = p[2]
        else:
            joined.append(p)

    rows = []
    for p in joined:
        rows.append(note_row(pitches, p[1], p[2], rate))
    return pd.DataFrame(rows, columns=["note", "start", "end", "length", "pitch", "cents"])


def note_row(pitches, start, end, rate):
    part = pitches[start:end]
    pitch = np.median(part[part > 0])
    name, cents = freq_to_note(pitch)
    start_time = start * HOP / rate
    end_time = end * HOP / rate
    return [name, round(start_time, 2), round(end_time, 2), round(end_time - start_time, 2), round(pitch, 1), cents]


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


def plot_pitch(pitches, smooth, rate, title):
    times = np.arange(len(pitches)) * HOP / rate
    # nan so no line there
    midi = np.full(len(smooth), np.nan)
    voiced = smooth > 0
    midi[voiced] = 69 + 12 * np.log2(smooth[voiced] / 440)
    raw = np.full(len(pitches), np.nan)
    raw[pitches > 0] = 69 + 12 * np.log2(pitches[pitches > 0] / 440)

    plt.figure(figsize=(12, 5))
    plt.plot(times, raw, color="lightgray", linewidth=1, label="before smoothing")
    plt.plot(times, midi, linewidth=1.5, label="smooth")

    # octave jumps made the axis too big
    low = int(np.nanpercentile(midi, 2)) - 1
    high = int(np.nanpercentile(midi, 98)) + 2
    ticks = list(range(low, high))
    labels = []
    for t in ticks:
        labels.append(NOTE_NAMES[t % 12] + str(t // 12 - 1))
    plt.yticks(ticks, labels)
    plt.ylim(low, high - 1)
    plt.grid(axis="y", alpha=0.4)
    plt.legend()
    plt.title("Pitch over time: " + title)
    plt.xlabel("Time (s)")
    plt.ylabel("Note")
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
smooth = smooth_pitch(pitches)
voiced = smooth[smooth > 0]
print()
print("Pitch found in", len(voiced), "of", len(frames), "frames")
print("Average pitch:", round(np.mean(voiced), 1), "Hz")
print("Lowest pitch:", round(np.min(voiced), 1), "Hz")
print("Highest pitch:", round(np.max(voiced), 1), "Hz")

notes = get_notes(smooth, rate)
print()
print("Notes:")
print(notes.to_string(index=False))

plot_waveform(samples, rate, silent_parts, filename)
plot_spectrogram(frames, rate, filename)
plot_pitch(pitches, smooth, rate, filename)
