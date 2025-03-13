from constants import *
import matplotlib.pyplot as plt
import numpy as np
import argparse

def plot_single_mic(time_vec, pcm_data):
    """Plot single microphone data."""
    plt.figure(figsize=(12, 6))
    plt.plot(time_vec, pcm_data, label="PCM Data", color='blue')
    plt.title("Single Microphone Data")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.legend()
    plt.show()

def get_single_mic_data(filename):
    """Read and decode single microphone data from a raw file."""
    with open(filename, "rb") as f:
        raw_data = f.read()
        pcm_data = np.frombuffer(raw_data, dtype=np.int16)
        sample_rate = SAMPLE_RATE_PCM_HZ
        time_vec = np.arange(len(pcm_data)) / sample_rate
        return time_vec, pcm_data
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot single microphone data from a raw file.")
    parser.add_argument("filename", type=str, help="Path to the raw file")
    args = parser.parse_args()
    filename = args.filename
    if not filename.endswith(".raw"):
        raise ValueError("Filename must end with .raw")

    time_vec, pcm_data = get_single_mic_data(filename)
    plot_single_mic(time_vec, pcm_data)
