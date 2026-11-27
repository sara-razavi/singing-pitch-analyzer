# Singing Pitch Analyzer

A program that looks at my singing recordings and shows which notes I sing and how in tune they are. You give it a wav file and it shows the waveform, the spectrogram and the pitch over time, finds the silent parts and lists every note with how many cents it is from the nearest note. It can also find the notes I really sing, including the ones between the 12 normal notes (koron and sori in Persian music), and compare two recordings.

I wanted to sing more professionally, and the idea for this project came from a university I'm applying to.

## How it works

- The audio is split into frames of 2048 samples (about 46 ms) and the quiet frames count as silence
- The pitch of every frame is found with autocorrelation, then smoothed with a median filter
- Every pitch is turned into a note name and its distance in cents from A = 440 Hz
- The summary shows the range, the most sung note and how far the notes are from the 12 notes on average, and everything is saved as csv in `results/`
- To find the notes I really sing, KMeans groups the stable frames and the number of groups is chosen by the silhouette score. I sing without a piano, so my own tuning is measured first and a note more than 25 cents away from the 12 notes is marked as koron or sori
- Compare shows the notes and tuning of two recordings and draws how far each one is from the 12 notes

## An example

On my recording of Gharibe (27 seconds), it found 5 notes: C4, D#4, F4, G4 and an A#4 that is 40 cents higher than normal, so it's probably a sori. My tuning was 4 cents above A = 440.

## Samples

`samples/` has seven of my own recordings:

- Gharibe, Mordab and Del az man, three Persian songs with only my voice
- Never enough, Frozen, I wish you the best and Read all about it, four English songs with music behind my voice, so you can see how the results get worse. In Read all about it, for example, it finds notes as low as D2, which come from the music and not from me

Copy them to a `recordings/` folder to try them.

## Things to know

- It reads 16 bit wav files only. Other files can be converted with ffmpeg:
  `ffmpeg -i song.ogg -ac 1 -sample_fmt s16 song.wav`
- It works best without music in the background, because the music makes the pitch jump around
- The koron/sori result is a guess (that's why it has a "?"). A note sung a bit out of tune can look the same

## Tech

Python, NumPy, pandas, matplotlib, scikit-learn

## How to run

```bash
git clone https://github.com/sara-razavi/singing-pitch-analyzer.git
cd singing-pitch-analyzer
pip install numpy pandas matplotlib scikit-learn
python main.py
```

Put wav files in a `recordings/` folder next to `main.py`.

## Author

**Sara Razavi**: [GitHub](https://github.com/sara-razavi)
