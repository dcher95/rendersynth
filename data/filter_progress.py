import os
import time
from collections import Counter

def get_file_counts(base_dir):
    file_counts = []
    for bbox_id in os.listdir(base_dir):
        bbox_path = os.path.join(base_dir, bbox_id)
        if os.path.isdir(bbox_path):
            num_files = sum(1 for f in os.listdir(bbox_path) if f.endswith(".geojson"))
            file_counts.append(num_files)
    return file_counts

def display_progress(file_counts, last_progress_info=None, elapsed=None, bar_width=40):
    total_folders = len(file_counts)
    count_distribution = Counter(file_counts)
    max_files = max(count_distribution.keys())
    folders_with_max = count_distribution[max_files]
    progress = folders_with_max / total_folders

    # Clear screen
    os.system('clear' if os.name == 'posix' else 'cls')

    print("File Count Distribution (%):")
    for num_files, count in sorted(count_distribution.items()):
        percentage = 100 * count / total_folders
        print(f"{num_files} files: {count} folders ({percentage:.2f}%)")

    filled_len = int(bar_width * progress)
    bar = "█" * filled_len + "-" * (bar_width - filled_len)
    print(f"\nProgress to all folders having {max_files} files:")
    print(f"[{bar}] {progress * 100:.2f}%")

    eta_text = "ETA: estimating..."
    if last_progress_info:
        delta_folders = folders_with_max - last_progress_info["folders_with_max"]
        delta_time = elapsed
        if delta_folders > 0:
            folders_remaining = total_folders - folders_with_max
            rate = delta_folders / delta_time  # folders per second
            eta_seconds = folders_remaining / rate
            eta_text = f"ETA: {format_time(eta_seconds)}"
        else:
            eta_text = "ETA: stalled (no progress)"

    print(eta_text)
    return {"folders_with_max": folders_with_max}

def format_time(seconds):
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {sec}s"
    elif minutes > 0:
        return f"{minutes}m {sec}s"
    else:
        return f"{sec}s"

if __name__ == "__main__":
    base_dir = "/Users/dsc/Desktop/repos/universe7/sandbox/vector_synth/data/clipped_bbox_geojson"
    interval = 5  # seconds
    last_info = None
    last_time = time.time()

    try:
        while True:
            start_time = time.time()
            file_counts = get_file_counts(base_dir)
            elapsed = start_time - last_time
            last_info = display_progress(file_counts, last_info, elapsed)
            last_time = start_time
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped.")
