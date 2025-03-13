from constants import *
import matplotlib.pyplot as plt
import numpy as np
import argparse
import wave
import os

def plot_single_mic(time_vec, pcm_data, channel):
    """Plot single microphone data for a specific channel."""
    plt.figure(figsize=(12, 6))
    plt.plot(time_vec, pcm_data[:, channel], label=f"PCM Data (Channel {channel})", color='blue')
    plt.title(f"Single Microphone Data - Channel {channel}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.legend()
    plt.show()

def plot_all_mics(time_vec, pcm_data):
    """Plot all microphone channels as subplots."""
    num_channels = pcm_data.shape[1]
    fig, axes = plt.subplots(num_channels, 1, figsize=(12, 6 * num_channels), sharex=True)
    fig.suptitle("All Microphone Channels")
    
    for channel in range(num_channels):
        axes[channel].plot(time_vec, pcm_data[:, channel], label=f"Channel {channel}")
        axes[channel].set_ylabel("Amplitude")
        axes[channel].grid()
        axes[channel].legend()
    
    axes[-1].set_xlabel("Time (s)")
    plt.show()

def plot_combined(time_vec, pcm_data):
    """Plot all microphone channels in a single figure."""
    plt.figure(figsize=(12, 6))
    for channel in range(pcm_data.shape[1]):
        plt.plot(time_vec, pcm_data[:, channel], label=f"Channel {channel}")
    plt.title("All Microphone Channels Combined")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.legend()
    plt.show()

def get_multi_mic_data(filename):
    """Read and decode multi-channel microphone data from a raw file."""
    with open(filename, "rb") as f:
        raw_data = f.read()
        pcm_data = np.frombuffer(raw_data, dtype=np.int16)
        # trim the data to the nearest multiple of NUM_CHANNELS
        num_samples = len(pcm_data) // NUM_CHANNELS * NUM_CHANNELS
        pcm_data = pcm_data[:num_samples].reshape(-1, NUM_CHANNELS)

        sample_rate = SAMPLE_RATE_PCM_HZ
        time_vec = np.arange(pcm_data.shape[0]) / sample_rate
        return time_vec, pcm_data, sample_rate

def save_as_wav(filename, pcm_data, sample_rate):
    """Save PCM data as NUM_CHANNELS separate WAV files, one per channel."""
    
    # Get the directory of the input file
    file_dir = os.path.dirname(os.path.abspath(filename))
    
    # Create a "wav_outputs" directory in the same location as the input file
    output_dir = os.path.join(file_dir, "wav_outputs")
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
    
    base_filename = os.path.basename(filename).replace(".raw", "")
    
    for channel in range(NUM_CHANNELS):
        wav_filename = os.path.join(output_dir, f"{base_filename}_ch{channel}.wav")
        
        with wave.open(wav_filename, "w") as wf:
            wf.setnchannels(1)  # Mono per channel
            wf.setsampwidth(2)  # 16-bit PCM
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data[:, channel].tobytes())
        
        print(f"Saved WAV file: {wav_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot multi-channel microphone data from a raw file.")
    parser.add_argument("filename", type=str, help="Path to the raw file")
    parser.add_argument("--plot", type=int, default=None, help="Plot data for a specific channel (0-7)")
    parser.add_argument("--plot-all", action="store_true", help="Plot all channels as subplots")
    parser.add_argument("--plot-combined", action="store_true", help="Plot all channels in one figure")
    parser.add_argument("--generate-wav", action="store_true", help="Save each channel as a separate WAV file")
    args = parser.parse_args()
    filename = args.filename
    if not filename.endswith(".raw"):
        raise ValueError("Filename must end with .raw")

    time_vec, pcm_data, sample_rate = get_multi_mic_data(filename)
    if args.plot is not None:
        if 0 <= args.plot < NUM_CHANNELS:
            plot_single_mic(time_vec, pcm_data, args.plot)
        else:
            raise ValueError("Channel must be between 0 and 7")
    if args.plot_all:
        plot_all_mics(time_vec, pcm_data)
    if args.plot_combined:
        plot_combined(time_vec, pcm_data)
    if args.generate_wav:
        save_as_wav(filename, pcm_data, sample_rate)