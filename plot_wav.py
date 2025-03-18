from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

import argparse


import os
import glob

import re
from collections import defaultdict


def get_wav_files(directory):
    """Retrieve all .wav files from a given directory and its subdirectories."""
    return glob.glob(os.path.join(directory, "**", "*.wav"), recursive=True)

def plot_channel_amp(wav_files):
        
    samplerate, data = wavfile.read(wav_files)

    _, num_channels = data.shape


    fig, axs = plt.subplots(num_channels, 1, figsize=(10, 3*num_channels))



    if num_channels == 1:
        axs = [axs]  # Ensure axs is iterable

    time = np.linspace(0, len(data) / samplerate, len(data))

    num_channels = 2

    for channel in range(0,num_channels):
        
        print(np.shape(data[:, channel]))
        axs[channel].plot(time, data[:,channel], label=f"Channel {channel}")
        axs[channel].set_title(f"Channel {channel} Amplitude")
        axs[channel].set_ylabel("Amplitude")
        axs[channel].set_xlabel("Time [s]")
        axs[channel].grid(True)

    plt.show()


def plot_channel_spect(wav_files):
    # Read WAV file
    samplerate, data = wavfile.read(wav_files)

    # Ensure data is 2D (single-channel audio needs reshaping)
    if data.ndim == 1:
        data = np.expand_dims(data, axis=1)  # Convert (N,) to (N,1)

    num_channels = data.shape[1]  # Extract number of channels


    print(num_channels)
    num_channels = 2


    # Create subplots
    fig, axs = plt.subplots(num_channels, 1, figsize=(10, 3 * num_channels))

    # Ensure axs is iterable for single-channel cases
    if num_channels == 1:
        axs = [axs]

    spectrograms, freqs, times = [], [], []

    for channel in range(num_channels):
        f, t, Sxx = signal.spectrogram(data[:, channel], samplerate)
        freqs.append(f)
        times.append(t)
        spectrograms.append(Sxx)

    # Compute global min/max for color scaling
    global_min = min(np.min(Sxx) for Sxx in spectrograms) - 1e-3
    global_max = max(np.max(Sxx) for Sxx in spectrograms) 
    norm = mcolors.LogNorm(vmin=global_min, vmax=global_max)

    # Plot each spectrogram
    for i in range(num_channels):
        pcm = axs[i].pcolormesh(times[i], freqs[i], spectrograms[i], shading="gouraud", norm=norm, cmap="magma")
        axs[i].set_ylabel("Frequency [Hz]")
        axs[i].set_xlabel("Time [s]")
        axs[i].set_title(f"Channel {i} Spectrogram")
        fig.colorbar(pcm, ax=axs[i])

    plt.show()

def plot_channel_fft(wav_file):
        # Read WAV file
    samplerate, data = wavfile.read(wav_file)

    # Ensure data is 2D (convert mono to stereo format)
    if data.ndim == 1:
        data = np.expand_dims(data, axis=1)  # Convert (N,) to (N,1)

    num_channels = data.shape[1]  # Extract number of channels

    # num_channels = 3
    # Create subplots
    fig, axs = plt.subplots(num_channels, 1, figsize=(12, 3 * num_channels))

    # Ensure axs is iterable for single-channel cases
    if num_channels == 1:
        axs = [axs]

    colors = ["red", "blue", "green", "purple", "orange", "brown", "pink", "cyan", "lime"]

    def compute_fft(channel_data, samplerate):
        # Normalize to mean amplitude
        mean_amplitude = np.mean(np.abs(channel_data))  # Compute mean amplitude
        if mean_amplitude == 0:
            mean_amplitude = 1e-10  # Avoid division by zero
        normalized_data = channel_data / mean_amplitude  # Normalize

        # Compute FFT
        fft_values = np.abs(np.fft.rfft(normalized_data))  # Compute magnitude spectrum
        fft_values[0] = 1e-10  # Remove DC offset
        fft_values /= np.max(fft_values)  # Normalize
        fft_values = 20 * np.log10(fft_values + 1e-10)  # Convert to dB

        freqs = np.fft.rfftfreq(len(normalized_data), 1 / samplerate)
        return freqs[1:], fft_values[1:]  # Exclude DC component

    # Process each channel
    for i in range(num_channels):
        freqs, fft_values = compute_fft(data[:, i], samplerate)

        axs[i].plot(freqs, fft_values, color=colors[i % len(colors)], label=f"Channel {i}")
        axs[i].set_xscale("log")
        axs[i].set_ylabel("Magnitude [dB]")
        axs[i].set_xlabel("Frequency [Hz]")
        axs[i].set_title(f"Channel {i} FFT (Normalized to Mean Amplitude)")
        axs[i].grid(True, which="both", linestyle="--", linewidth=0.5)
        axs[i].legend()

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Change plotting outputs")
    parser.add_argument("--plot_outputs", action="store_true", help="Serial port to read from (default: /dev/ttyACM0)")
    parser.add_argument("--audio_dir", type=str, help="Add audio directory for specific file" )
    args = parser.parse_args()

    print(args.audio_dir)
    path = args.audio_dir

    # plot_channel_amp(path)
    plot_channel_fft(path)






    

    # date_keys = listed_data.keys()

    # for key, value in listed_data.items():
    #     # breakpoint()

    #     if args.plot_outputs:
    #         plot_channel_amp(listed_data[key])
    #         plot_channel_spect(listed_data[key])
    #         plot_channel_fft(listed_data[key])
        
    #     if args.plot_spatial:
    #         plot_spatial_sound(listed_data[key])

        


        


