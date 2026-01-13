import numpy as np

def create_histogram(array, num_bins=256):
    counts, bin_edges = np.histogram(array, bins=num_bins)
    x = (bin_edges[:-1] + bin_edges[1:]) / 2  # bin centers
    y = counts
    return x, y

def find_peak_and_drops(x, y, drop_threshold=0.10):
    peak_idx = np.argmax(y)

    # Find left drop
    left_idx = 0
    for i in range(0, peak_idx):
        if y[i] < y[peak_idx] * drop_threshold:
            left_idx = i
            break

    # Find right drop
    right_idx = peak_idx
    for i in range(peak_idx, len(y)):
        if y[i] < y[peak_idx] * drop_threshold:
            right_idx = i
            break

    return peak_idx, left_idx, right_idx