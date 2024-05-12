from librosa import load, stft
from numpy import argmax, abs

bpm = 138
audio_file = "audio.wav"
interval = (60 / bpm / 4) * 1000
y, sr = load(audio_file)
hop_length = int(sr * interval / 1000)
window_size = hop_length * 2
freqs = [sr * argmax(abs(stft(y[i:i + window_size], n_fft=window_size))) / (len(stft(y[i:i + window_size], n_fft=window_size)) * 2) for i in range(0, len(y), hop_length)]

out = f"{interval}\n{','.join(map(str, freqs))}\n"

print(f"{len(freqs)} notes ({round(len(out.encode('utf-8')) / 1000, 2)}kB)")

with open("out.audio", "w") as file:
  file.write(out)
