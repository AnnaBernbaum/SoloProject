import wave
import matplotlib.pyplot as plt
import numpy as np

fs = 44100

DPA_breathing1 = wave.open('WAV_files/DPA_tests1/DPA_breathing1.wav')
print(DPA_breathing1)

#Extract Raw Audio from Wav File
signal = DPA_breathing1.readframes(-1)
signal = np.fromstring(signal, 'Int16')
#load in data

eh = range(0, (len(signal)-1))
print(eh)
# Basic Plotting
time_values = range(0, (len(signal)-1))/44100 #0:length(signal)-1)/44100
print(time_values)
plt.figure(1)
plt.title('Audio Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude Normalised')
plt.plot(signal)
plt.show()

# plot spectrograms
# remove outliers
# remove points either side of outliers
# interpolate gaps
# compute spectral estimate
# plot RMS