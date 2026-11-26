import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def to_midi(freq):
    return 69 + 12 * np.log2(freq / 440)


def to_freq(midi):
    return 440 * 2 ** ((midi - 69) / 12)


def stable_midi(pitches):
    # slides between notes made extra clusters
    voiced = pitches > 0
    midi = np.zeros(len(pitches))
    midi[voiced] = to_midi(pitches[voiced])

    result = []
    for i in range(1, len(pitches) - 1):
        if voiced[i - 1] and voiced[i] and voiced[i + 1]:
            if abs(midi[i] - midi[i - 1]) < 0.3 and abs(midi[i + 1] - midi[i]) < 0.3:
                result.append(midi[i])
    return np.array(result)


def find_real_notes(pitches):
    midi = stable_midi(pitches)
    x = midi.reshape(-1, 1)

    best_k = 2
    best_score = -1
    for k in range(2, 13):
        model = KMeans(n_clusters=k, n_init=10, random_state=0)
        labels = model.fit_predict(x)
        # too slow with all the frames
        score = silhouette_score(x, labels, sample_size=2000, random_state=0)
        print("k =", k, "->", round(score, 3))
        if score > best_score:
            best_k = k
            best_score = score
    print("Best k:", best_k)

    model = KMeans(n_clusters=best_k, n_init=10, random_state=0)
    labels = model.fit_predict(x)
    centers = model.cluster_centers_[:, 0]

    # no piano, so my A is not exactly 440
    tuning = np.median(centers - np.round(centers))

    rows = []
    for c in range(best_k):
        center = centers[c] - tuning
        nearest = int(round(center))
        cents = int(round((center - nearest) * 100))
        if cents < -25:
            kind = "koron?"
        elif cents > 25:
            kind = "sori?"
        else:
            kind = "normal"
        percent = np.sum(labels == c) / len(labels) * 100
        rows.append([NAMES[nearest % 12] + str(nearest // 12 - 1), round(to_freq(centers[c]), 1), cents, kind, round(percent, 1)])

    table = pd.DataFrame(rows, columns=["note", "pitch", "cents", "kind", "percent"])
    table = table[table["percent"] >= 3]
    table = table.sort_values("pitch").reset_index(drop=True)
    return table, midi, tuning


def plot_real_notes(midi, table, tuning, title):
    plt.figure(figsize=(12, 5))
    low = int(np.floor(midi.min())) - 1
    high = int(np.ceil(midi.max())) + 1
    plt.hist(midi, bins=np.arange(low, high, 0.1), color="lightblue")

    ticks = []
    labels = []
    for n in range(low, high + 1):
        plt.axvline(n + tuning, color="gray", linestyle="--", linewidth=0.8)
        ticks.append(n + tuning)
        labels.append(NAMES[n % 12] + str(n // 12 - 1))
    for p in table["pitch"]:
        plt.axvline(to_midi(p), color="red", linewidth=2)

    plt.xticks(ticks, labels)
    plt.title("Notes I really sing (red) and the 12 notes (gray): " + title)
    plt.xlabel("Note")
    plt.ylabel("Frames")
    plt.show()
