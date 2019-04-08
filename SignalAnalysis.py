import wave
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os as os

plt.close('all')  # Close all open figures
fs = 44100  # Sampling frequency


def import_data(filepath):
    """Import a signal from a .wav file as an array"""

    data = wave.open(filepath)
    # Extract Raw Audio from Wav File
    signal = data.readframes(-1)
    signal = np.fromstring(signal, 'Int16')
    name = os.path.basename(filepath)
    return signal, name

def basic_plot(signal, name):
    time_values = np.arange(0,(len(signal))/fs, 1/fs)  # convert from smaple number to time
    plt.title(name)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude Normalised')
    plt.plot(time_values, signal)
    plt.show()

# Load in data to pandas dataframe
filepaths = ['WAV_files/DPA_tests1/DPA_breathing1.wav',
             'WAV_files/DPA_tests1/DPA_breathing2.wav',
             'WAV_files/DPA_tests1/DPA_breathing3.wav',
             'WAV_files/DPA_tests1/DPA_forced_breathing1.wav',
             'WAV_files/DPA_tests1/DPA_forced_breathing2.wav',
             'WAV_files/DPA_tests1/DPA_forced_breathing3.wav',
             'WAV_files/DPA_tests1/DPA_cough1.wav',
             'WAV_files/DPA_tests1/DPA_cough2.wav',
             'WAV_files/DPA_tests1/DPA_cough3.wav',
             'WAV_files/DPA_tests1/DPA_wheeze1.wav',
             'WAV_files/DPA_tests1/DPA_wheeze2.wav',
             'WAV_files/DPA_tests1/DPA_wheeze3.wav',
             'WAV_files/DPA_stethescope_tests1/DPA_stethescope_breathing1.wav',
             'WAV_files/DPA_stethescope_tests1/DPA_stethescope_breathing2.wav',
             'WAV_files/DPA_stethescope_tests1/DPA_stethescope_breathing3.wav',
             'WAV_files/DPA_stethescope_tests1/DPA_stethescope_forced_breathing1.wav',
             'WAV_files/DPA_stethescope_tests1/DPA_stethescope_forced_breathing2.wav',
             'WAV_files/DPA_stethescope_tests1/DPA_stethescope_forced_breathing3.wav',
             'WAV_files/DPA_stethescope_tests1/DPA_stethescope_cough1.wav',
             'WAV_files/DPA_stethescope_tests1/DPA_stethescope_wheeze1.wav',
             'WAV_files/Cducer_tests1/Cducer_breathing1.wav',
             'WAV_files/Cducer_tests1/Cducer_breathing2.wav',
             'WAV_files/Cducer_tests1/Cducer_breathing3.wav',
             'WAV_files/Cducer_tests1/Cducer_forced_breathing1.wav',
             'WAV_files/Cducer_tests1/Cducer_forced_breathing2.wav',
             'WAV_files/Cducer_tests1/Cducer_forced_breathing3.wav',
             'WAV_files/Cducer_tests1/Cducer_cough1.wav',
             'WAV_files/Cducer_tests1/Cducer_cough2.wav',
             'WAV_files/Cducer_tests1/Cducer_cough3.wav',
             ]

df = pd.DataFrame()  # create empty dataframe

# import data for every file of collected data
for file in filepaths:
    signal, name = import_data(file)
    df[name] = pd.Series(signal)  # append signal to new column in pandas df

# Basic Plotting
signal = (df['Cducer_cough2.wav'])  # Change this to the desired signal to plot
# basic_plot(signal, 'Cducer_cough2.wav')

# Remove outliers
window = int(0.2 * fs)  # window size in sample number
threshold = 6000

# https://ocefpaf.github.io/python4oceanographers/blog/2015/03/16/outlier_detection/
df['median'] = df['DPA_stethescope_forced_breathing1.wav'].rolling(window).median()
difference = np.abs(df['DPA_stethescope_forced_breathing1.wav'] - df['median'])
outlier_idx = difference > threshold

# Plot outliers
# time_values = np.arange(0, (len(df['DPA_stethescope_forced_breathing1.wav'])) / fs, 1 / fs)  # convert from sample number to time
# plt.figure(1)
# plt.title('Outlier Detection')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude Normalised')
# plt.plot(time_values, df['DPA_stethescope_forced_breathing1.wav'])
# plt.plot(time_values[outlier_idx], df['DPA_stethescope_forced_breathing1.wav'][outlier_idx], 'x')
# plt.show()

# remove points either side of outliers

# create duplicate column for editing
df['RemovedOutliers'] = df['DPA_stethescope_forced_breathing1.wav']

# test_data = df['DPA_stethescope_forced_breathing1.wav'].tolist()
# outliers = outlier_idx.tolist()

print(len(df['DPA_stethescope_forced_breathing1.wav']))
# print(len(test_data))
# print(len(outliers))

removal_length = 1000
# replacement_series = # series of NaNs length 2001

for i in np.arange(0, len(df['RemovedOutliers'])):
    if outlier_idx[i] == True:
        current = df['RemovedOutliers'][i]
        replace_range = df['RemovedOutliers'][i-removal_length : i+removal_length]
        # df['RemovedOutliers'].replace(current, np.nan)
        # print('outliers')
        if len(df['RemovedOutliers']) - i > len(replace_range): # if theres enough space
            # df['RemovedOutliers'].loc[i] = np.nan
            df['RemovedOutliers'].loc[i-removal_length:i+removal_length] = np.nan
    else:
        pass


basic_plot(df['RemovedOutliers'], 'ahaha cry')
basic_plot(df['DPA_stethescope_forced_breathing1.wav'], 'ahaha cry')


# interpolate gaps
df.interpolate(method = 'linear')

# compute spectral estimate
plt.specgram(df['RemovedOutliers'], Fs=fs)
plt.title('Spectrogram')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (kHz)')
plt.show()

# Plot spectrograms
# plot RMS