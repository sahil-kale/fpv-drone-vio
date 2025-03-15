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

AUDIO_DIR = r'C:\Users\aksha\Desktop\University\4B\MTE 546\Project\mte-546-proj\logs\wav_outputs'

SPEED_OF_SOUND = 343.0  

# Define microphone positions (x, y) in meters
MIC_POSITIONS = {
    0: (0, 0),   # Mic 0 at origin
    1: (0.15, 0),  # Mic 1 at (20 cm, 0)
    2: (0.3, 0) # Mic 2 at (10 cm, 20 cm)
}



def get_wav_files(directory):
    """Retrieve all .wav files from a given directory and its subdirectories."""
    return glob.glob(os.path.join(directory, "**", "*.wav"), recursive=True)

def group_wav_files(wav_files):
    """Group wav files by date-time combo, keeping only _ch0, _ch1, and _ch2 files."""
    grouped_files = defaultdict(list)
    pattern = re.compile(r"AUDIO_RECORD_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_ch([0-3])\.wav")

    for file in wav_files:
        filename = os.path.basename(file)
        match = pattern.match(filename)
        if match:
            datetime_part, channel = match.groups()
            if channel in {"0", "1", "2"}:  # Only keep ch0, ch1, ch2
                grouped_files[datetime_part].append(file)

    return grouped_files

def plot_channel_amp(wav_files):
    num_channels = len(wav_files)
    fig, axs = plt.subplots(num_channels, 1, figsize=(10, 3 * num_channels))

    if num_channels == 1:
        axs = [axs]  # Ensure axs is iterable

    samplerate, data_list = wavfile.read(wav_files[0]), []
    time = None

    for i, file in enumerate(wav_files):
        samplerate, data = wavfile.read(file)
        data_list.append(data)

        if time is None:
            time = np.linspace(0, len(data) / samplerate, len(data))

        axs[i].plot(time, data, label=f"Channel {i}")
        axs[i].set_title(f"Channel {i} Amplitude")
        axs[i].set_ylabel("Amplitude")
        axs[i].set_xlabel("Time [s]")
        axs[i].legend()
        axs[i].grid(True)

    plt.tight_layout()

def plot_channel_spect(wav_files):
    num_channels = len(wav_files)
    fig, axs = plt.subplots(num_channels, 1, figsize=(10, 3 * num_channels))

    if num_channels == 1:
        axs = [axs]  # Ensure axs is iterable

    spectrograms, freqs, times = [], [], []
    samplerate, _ = wavfile.read(wav_files[0])

    for i, file in enumerate(wav_files):
        samplerate, data = wavfile.read(file)
        f, t, Sxx = signal.spectrogram(data, samplerate)

        freqs.append(f)
        times.append(t)
        spectrograms.append(Sxx)

    # Compute global min/max for consistent color scaling
    global_min = min(Sxx.min() for Sxx in spectrograms) + 1e-5
    global_max = max(Sxx.max() for Sxx in spectrograms) + 1e3

    norm = mcolors.LogNorm(vmin=global_min, vmax=global_max)

    for i in range(num_channels):
        pcm = axs[i].pcolormesh(times[i], freqs[i], spectrograms[i], shading="gouraud", norm=norm, cmap="magma")
        axs[i].set_ylabel("Frequency [Hz]")
        axs[i].set_xlabel("Time [s]")
        axs[i].set_title(f"Channel {i} Spectrogram")
        fig.colorbar(pcm, ax=axs[i])

    plt.tight_layout()

def plot_channel_fft(wav_files):
    num_channels = len(wav_files)
    fig, axs = plt.subplots(num_channels, 1, figsize=(12, 3 * num_channels))

    if num_channels == 1:
        axs = [axs]  # Ensure axs is iterable

    colors = ["red", "blue", "green", "purple", "orange", "brown", "pink", "cyan", "lime"]

    def compute_fft(channel, samplerate):
        fft_values = np.abs(np.fft.rfft(channel))  # Compute magnitude
        fft_values[0] = 1e-10  # Remove DC offset
        fft_values /= np.max(fft_values)  # Normalize
        fft_values = 20 * np.log10(fft_values + 1e-10)  # Convert to dB
        freqs = np.fft.rfftfreq(len(channel), 1 / samplerate)
        return freqs[1:], fft_values[1:]

    for i, file in enumerate(wav_files):
        samplerate, data = wavfile.read(file)
        freqs, fft_values = compute_fft(data, samplerate)

        axs[i].plot(freqs, fft_values, color=colors[i % len(colors)], label=f"Channel {i}")
        axs[i].set_xscale("log")
        axs[i].set_ylabel("Magnitude [dB]")
        axs[i].set_xlabel("Frequency [Hz]")
        axs[i].set_title(f"Channel {i} FFT")
        axs[i].grid(True, which="both", linestyle="--", linewidth=0.5)
        axs[i].legend()

    plt.tight_layout()


def compute_tdoa(wav_files, samplerate):
    """Compute time differences of arrival (TDOA) between microphone pairs using GCC-PHAT."""
    num_channels = len(wav_files)
    delays = np.zeros((num_channels, num_channels))  # TDOA matrix
    signals = []
    
    for file in wav_files:
        _, data = wavfile.read(file)
        signals.append(data)
    
    for i in range(num_channels):
        for j in range(i + 1, num_channels):
            cross_corr = signal.correlate(signals[i], signals[j], mode='full')
            lag = np.argmax(np.abs(cross_corr)) - (len(signals[i]) - 1)
            delays[i, j] = lag / samplerate  # Convert lag to time (s)
    
    return delays

def estimate_sound_source(tdoa):
    """Estimate sound source position using multilateration."""
    mic_coords = np.array([MIC_POSITIONS[i] for i in range(len(MIC_POSITIONS))])
    estimated_positions = []
    
    for i in range(1, len(mic_coords)):
        d = SPEED_OF_SOUND * tdoa[0, i]  # Distance difference
        x1, y1 = mic_coords[0]
        x2, y2 = mic_coords[i]
        midpoint = [(x1 + x2) / 2, (y1 + y2) / 2]
        estimated_positions.append((midpoint[0] + d, midpoint[1] + d))
    
    return np.mean(estimated_positions, axis=0)

def plot_spatial_sound(wav_files):
    samplerate, _ = wavfile.read(wav_files[0])
    tdoa_matrix = compute_tdoa(wav_files, samplerate)
    sound_position = estimate_sound_source(tdoa_matrix)

    print(*sound_position)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot microphone positions
    for i, (x, y) in MIC_POSITIONS.items():
        ax.scatter(x, y, color='blue', label=f'Mic {i}' if i == 0 else "")
    
    # Plot estimated sound source position
    ax.scatter(*sound_position, color='red', marker='x', s=100, label='Estimated Source')
    
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('Sound Source Localization')
    ax.legend()
    ax.grid()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Change plotting outputs")
    parser.add_argument("--plot_outputs", action="store_true", help="Serial port to read from (default: /dev/ttyACM0)")
    parser.add_argument("--plot_spatial", action="store_true", help="Serial port to read from (default: /dev/ttyACM0)")
    
    args = parser.parse_args()
    

    wav_file_list = get_wav_files(AUDIO_DIR)
    listed_data = group_wav_files(wav_file_list)
    date_keys = listed_data.keys()

    for key, value in listed_data.items():
        # breakpoint()

        if args.plot_outputs:
            plot_channel_amp(listed_data[key])
            plot_channel_spect(listed_data[key])
            plot_channel_fft(listed_data[key])
        
        if args.plot_spatial:
            plot_spatial_sound(listed_data[key])

        

        plt.show()


