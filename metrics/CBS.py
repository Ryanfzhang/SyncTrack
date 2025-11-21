import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import random
import csv
import json
import librosa
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
import warnings
import scipy

def load_beats_from_json(json_path):
    beats_list = []
    names = []
    with open(json_path, "r") as f:
        data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        for item in data:
            beats_list.append(np.array(item["beats"]))
            names.append(item["name"])
    return beats_list, names

def plot_multi_track_beats(track_beats, track_names=None, save_path="multi_track_beats_binary.png", resolution=0.01, window_edges=None, max_length=None):
    stem_names = ["bass", "drum", "guitar", "piano"]
    # Automatically infer plot length if it is not provided explicitly.
    if max_length is None:
        inferred = 0.0
        for beats in track_beats:
            if len(beats) > 0:
                inferred = max(inferred, float(np.max(beats)))
        # Add a small margin so the final beat is fully rendered.
        max_length = max(0.0, inferred + 1.0)
    time_points = np.arange(0, max_length, resolution)
    plt.figure(figsize=(16, 6))
    for idx, beats in enumerate(track_beats):
        binary_seq = np.zeros_like(time_points)
        beat_indices = np.searchsorted(time_points, beats)
        beat_indices = beat_indices[beat_indices < len(binary_seq)]
        binary_seq[beat_indices] = 1
        label = stem_names[idx] if idx < len(stem_names) else f"stem_{idx}"
        plt.plot(time_points, binary_seq + idx, label=label)
    plt.yticks(range(len(track_beats)), [stem_names[i] if i < len(stem_names) else f"stem_{i}" for i in range(len(track_beats))])
    plt.xlabel("Time (s)")
    plt.title("Multi-track Beat Binary Time Series")
    plt.legend(loc='upper right')
    if window_edges is not None:
        for edge in window_edges:
            plt.axvline(x=edge, color='grey', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Multi-track beat binary timeline saved to: {save_path}")

def estimate_beats(audio_file, sr=22050, beat_type="madmom", madmom_fps=150, madmom_transition_lambda=100):
    if beat_type == "madmom":
        downbeat_proc = RNNDownBeatProcessor()
        downbeat_activation = downbeat_proc(audio_file)
        dbn_downbeat = DBNDownBeatTrackingProcessor(
            beats_per_bar=[3, 4], min_bpm=30, max_bpm=300,
            fps=madmom_fps, transition_lambda=madmom_transition_lambda)
        downbeats = dbn_downbeat(downbeat_activation)
        beat_times = downbeats[:, 0]
    else:
        y, sr_ = librosa.load(audio_file, sr=sr)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr_)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr_)
    return beat_times



def compute_alignment_metric(
    track_files=None,
    track_beats=None,
    window_size=0.07,
    overlap=0.0,
    sr=22050,
    plot=True,
    fig_path="multi_track_beats.png",
    track_names=None,
    max_length=None,
    beat_type="madmom",
    madmom_fps=150,
    madmom_transition_lambda=100
):
    # Gather beats, track names, and maximum length.
    if track_beats is None:
        assert track_files is not None
        track_beats = [estimate_beats(f, sr, beat_type, madmom_fps, madmom_transition_lambda) for f in track_files]
        track_lengths = []
        for f in track_files:
            y, _ = librosa.load(f, sr=sr)
            track_lengths.append(len(y) / sr)
        max_length_local = max(track_lengths)
        track_names = track_files
    else:
        max_length_local = max_length
        if track_names is None:
            track_names = [f"track_{i}" for i in range(len(track_beats))]
        # When max_length is missing, infer it from beats in beats_json mode.
        if max_length is None:
            inferred = 0.0
            for beats in track_beats:
                if len(beats) > 0:
                    inferred = max(inferred, float(np.max(beats)))
            # Add one more window to guarantee coverage after the last beat.
            max_length_local = inferred + window_size

    N = len(track_beats)
    valid_track_count = sum(len(beats) > 0 for beats in track_beats)

    # Return NaN when there is at most one valid track.
    if valid_track_count <= 1:
        return {
            'window_count': np.nan,
            'valid_window_count': np.nan,
            'mean_beat_ratio': np.nan,
            'window_size(ms)': np.nan,
            'overlap(ms)': np.nan,
            'p_per_window': np.nan,
            'valid_track_count': valid_track_count,
        }

    # 4. Build sliding windows.
    step = window_size - overlap
    T = int(np.ceil((max_length_local - window_size) / step)) + 1
    windows = [(i * step, i * step + window_size) for i in range(T)]
    window_edges = [w[0] for w in windows]

    # 5. Count whether each track has a beat per window.
    b = np.zeros((T, N), dtype=int)
    for j, beats in enumerate(track_beats):
        if len(beats) == 0:
            continue
        for i, (start, end) in enumerate(windows):
            if np.any((beats >= start) & (beats < end)):
                b[i, j] = 1

    denominator = valid_track_count
    p = np.sum(b, axis=1) / denominator  # Coverage ratio per window.
    valid_window_mask = np.sum(b, axis=1) >= 1
    total_valid_windows = np.sum(valid_window_mask)
    mean_beat_ratio = float(np.sum(p[valid_window_mask]) / (total_valid_windows + 1e-10)) if total_valid_windows > 0 else 0

    if plot:
        plot_multi_track_beats(track_beats, track_names, save_path=fig_path, window_edges=window_edges, max_length=max_length_local)

    return {
        'window_count': T,
        'valid_window_count': int(total_valid_windows),
        'mean_beat_ratio': mean_beat_ratio,
        'window_size(ms)': window_size * 1000,
        'overlap(ms)': overlap * 1000,
        'p_per_window': p.tolist(),
        'valid_track_count': valid_track_count,
    }

def print_valid_track_count_distribution(results):
    from collections import Counter
    valid_track_counts = [r['valid_track_count'] for r in results if r['valid_track_count'] is not None]
    dist = Counter(valid_track_counts)
    print("\nDistribution of valid track counts:")
    for k in sorted(dist.keys()):
        print(f"  valid_track_count = {k}: {dist[k]} samples")
    return dist


def collect_tracks_single(folder, filename, stems_list):
    """
    Given a root directory and filename, gather the corresponding file under each stem.
    """
    return [os.path.join(folder, stem, filename) for stem in stems_list]

def collect_tracks_folder(folder, stems_list):
    """
    Traverse the folder to find all matching file groups.
    Returns: list of lists, each inner list is one set of tracks.
    """
    stem_files = {}
    # Collect all filenames under each stem.
    for stem in stems_list:
        stem_path = os.path.join(folder, stem)
        if not os.path.exists(stem_path):
            print(f"Stem directory missing: {stem_path}")
            continue
        files = set(f for f in os.listdir(stem_path) if f.endswith('.wav'))
        stem_files[stem] = files
        print(stem, len(files))
    # Intersect filenames that appear in every stem directory.
    common_files = set.intersection(*(stem_files[stem] for stem in stems_list if stem in stem_files))
    print(f"Found {len(common_files)} multi-track groups")
    # Build track groups by filename.
    all_tracks = []
    for filename in sorted(common_files):
        track_files = [os.path.join(folder, stem, filename) for stem in stems_list]
        all_tracks.append(track_files)
    print(f"Collected {len(all_tracks)} track groups")
    return all_tracks

def collect_tracks_beats(beats_json_dir, stems_list):
    """
    Load beat JSON files and return track_beats groups and track_names.
    track_names is shared across stems, so we keep the first set only.
    """
    all_stem_beats = []
    track_names = None
    for stem in stems_list:
        json_path = os.path.join(beats_json_dir, f"{stem}.json")
        if not os.path.exists(json_path):
            print(f"Missing JSON: {json_path}")
            return [], []
        beats_list, names = load_beats_from_json(json_path)
        all_stem_beats.append(beats_list)
        track_names = names 
    n_tracks = len(all_stem_beats[0])

    for stem_beats in all_stem_beats:
        assert len(stem_beats) == n_tracks, "Mismatch in number of stems"
    track_beats_groups = []
    for idx in range(n_tracks):
        track_beats = [all_stem_beats[s][idx] for s in range(len(stems_list))]
        track_beats_groups.append(track_beats)
    return track_beats_groups, track_names

def analyze_folder_by_json(beats_json_dir, stems_list, window_size=0.07, overlap=0.0, sr=22050, plot_folder=None, example_count=10, max_length=None, beat_type="madmom", madmom_fps=150, madmom_transition_lambda=100):
    track_beats_groups, track_names = collect_tracks_beats(beats_json_dir, stems_list)
    results = []
    example_results = []
    example_indices = set(random.sample(range(len(track_beats_groups)), min(example_count, len(track_beats_groups))))
    if plot_folder and not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    for idx, (track_beats, track_name) in enumerate(zip(track_beats_groups, track_names)):
        plot = idx in example_indices
        fig_path = os.path.join(plot_folder, f"{track_name}.png") if (plot and plot_folder) else None
        res = compute_alignment_metric(
            track_files=None,
            track_beats=track_beats,
            window_size=window_size,
            overlap=overlap,
            sr=sr,
            plot=plot,
            fig_path=fig_path,
            track_names=stems_list,
            max_length=max_length,
            beat_type=beat_type,
            madmom_fps=madmom_fps,
            madmom_transition_lambda=madmom_transition_lambda
        )
        results.append(res)
        if plot:
            example_results.append({
                'track_name': track_name,
                'window_count': res['window_count'],
                'valid_window_count': res['valid_window_count'],
                'mean_beat_ratio': res['mean_beat_ratio'],
                'valid_track_count': res['valid_track_count'],
            })
    print(f"Analyzed {len(results)} track groups")
    # Save selected examples.
    if plot_folder is not None and len(example_results) > 0:
        example_csv_path = os.path.join(plot_folder, "examples.csv")
        with open(example_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['track_name', 'window_count', 'valid_window_count', 'mean_beat_ratio', 'valid_track_count'])
            writer.writeheader()
            for row in example_results:
                writer.writerow(row)
        print(f"Saved metrics for {len(example_results)} examples to {example_csv_path}")
    # Save all results.
    if plot_folder is not None and len(results) > 0:
        all_csv_path = os.path.join(plot_folder, "all_results.csv")
        with open(all_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['track_name', 'window_count', 'valid_window_count', 'mean_beat_ratio', 'valid_track_count'])
            writer.writeheader()
            for idx, res in enumerate(results):
                writer.writerow({
                    'track_name': track_names[idx],
                    'window_count': res['window_count'],
                    'valid_window_count': res['valid_window_count'],
                    'mean_beat_ratio': res['mean_beat_ratio'],
                    'valid_track_count': res['valid_track_count'],
                })
        print(f"Saved metrics for all samples to {all_csv_path}")
    # Print aggregated statistics.
    avg_mean_beat_ratio = np.nanmean([r['mean_beat_ratio'] for r in results])
    avg_valid_window_count = np.nanmean([r['valid_window_count'] for r in results])
    avg_valid_track_count = np.nanmean([r['valid_track_count'] for r in results])
    print(f"\nAverage mean_beat_ratio across tracks: {avg_mean_beat_ratio:.4f}")
    print(f"Average valid_window_count across tracks: {avg_valid_window_count:.2f}")
    print(f"Average number of valid tracks: {avg_valid_track_count:.2f}")
    print_valid_track_count_distribution(results)
    return results

def analyze_folder_by_audio(
    folder, stems_list, window_size=0.07, overlap=0.0, sr=22050, num_workers=4,
    plot_folder=None, example_count=10, beat_type="madmom", madmom_fps=150, madmom_transition_lambda=100
):
    all_tracks = collect_tracks_folder(folder, stems_list)
    results = []
    example_results = []
    example_indices = set(random.sample(range(len(all_tracks)), min(example_count, len(all_tracks))))
    if plot_folder and not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    future_to_info = {}
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for idx, track_files in enumerate(all_tracks):
            plot = idx in example_indices
            fig_path = os.path.join(plot_folder, f"{os.path.basename(track_files[0])}.png") if (plot and plot_folder) else None
            future = executor.submit(
                compute_alignment_metric,
                track_files,
                None,
                window_size,
                overlap,
                sr,
                plot,
                fig_path,
                stems_list,
                None,  # Automatically infer audio length.
                beat_type,
                madmom_fps,
                madmom_transition_lambda
            )
            future_to_info[future] = (idx, track_files)
        for f in tqdm(as_completed(list(future_to_info.keys())), total=len(future_to_info), desc="Processing progress"):
            res = f.result()
            results.append(res)
            idx, track_files = future_to_info[f]
            if idx in example_indices and plot_folder is not None:
                wav_name = os.path.basename(track_files[0])
                example_results.append({
                    'track_files': '|'.join(track_files),
                    'window_count': res['window_count'],
                    'valid_window_count': res['valid_window_count'],
                    'mean_beat_ratio': res['mean_beat_ratio'],
                    'valid_track_count': res['valid_track_count'],
                })
    print(f"Analyzed {len(results)} track groups")
    # Save selected examples.
    if plot_folder is not None and len(example_results) > 0:
        example_csv_path = os.path.join(plot_folder, "examples.csv")
        with open(example_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['track_files', 'window_count', 'valid_window_count', 'mean_beat_ratio', 'valid_track_count'])
            writer.writeheader()
            for row in example_results:
                writer.writerow(row)
        print(f"Saved metrics for {len(example_results)} examples to {example_csv_path}")
    # Save all results.
    if plot_folder is not None and len(results) > 0:
        all_csv_path = os.path.join(plot_folder, "all_results.csv")
        with open(all_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['track_files', 'window_count', 'valid_window_count', 'mean_beat_ratio', 'valid_track_count'])
            writer.writeheader()
            for idx, res in enumerate(results):
                writer.writerow({
                    'track_files': '|'.join(all_tracks[idx]),
                    'window_count': res['window_count'],
                    'valid_window_count': res['valid_window_count'],
                    'mean_beat_ratio': res['mean_beat_ratio'],
                    'valid_track_count': res['valid_track_count'],
                })
        print(f"Saved metrics for all samples to {all_csv_path}")
    # Print aggregated statistics.
    avg_mean_beat_ratio = np.nanmean([r['mean_beat_ratio'] for r in results])
    avg_valid_window_count = np.nanmean([r['valid_window_count'] for r in results])
    avg_valid_track_count = np.nanmean([r['valid_track_count'] for r in results])
    print(f"\nAverage mean_beat_ratio across tracks: {avg_mean_beat_ratio:.4f}")
    print(f"Average valid_window_count across tracks: {avg_valid_window_count:.2f}")
    print(f"Average number of valid tracks: {avg_valid_track_count:.2f}")
    print_valid_track_count_distribution(results)
    return results

def analyze_single_track(
    folder, filename, stems_list, window_size=0.07, overlap=0.0, sr=22050, plot_folder=None,
    beat_type="madmom", madmom_fps=150, madmom_transition_lambda=100
):
    track_files = collect_tracks_single(folder, filename, stems_list)
    print(f"Analyzing tracks: {track_files}")
    if plot_folder and not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    fig_path = os.path.join(plot_folder, f"{filename}.png") if plot_folder else None
    result = compute_alignment_metric(
        track_files=track_files,
        track_beats=None,
        window_size=window_size,
        overlap=overlap,
        sr=sr,
        plot=True,
        fig_path=fig_path,
        track_names=stems_list,
        max_length=None, # If given original audio files, deduct max_length automatically
        beat_type=beat_type,
        madmom_fps=madmom_fps,
        madmom_transition_lambda=madmom_transition_lambda
    )
    print(f"Total window count: {result['window_count']}")
    print(f"Windows with at least one beat: {result['valid_window_count']}")
    print(f"Mean beat ratio: {result['mean_beat_ratio']:.4f}")
    print(f"Window length (ms): {result['window_size(ms)']}")
    print(f"Window overlap (ms): {result['overlap(ms)']}")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', required=False, help='Root directory that contains stem subfolders')
    parser.add_argument('--beats_json_dir', required=False, help='Directory with beat JSON files for each stem (preferred when available)')
    parser.add_argument('--track', help='Single filename to analyze (e.g., Track02098_from_40.wav); analyze all when omitted')
    parser.add_argument('--window_size', type=float, default=0.15, help='Window length in seconds')  
    parser.add_argument('--overlap', type=float, default=0.0, help='Window overlap in seconds (e.g., 0.035)')
    parser.add_argument('--sr', type=int, default=22050, help='Sampling rate')
    parser.add_argument('--stems', nargs='+', default=["stem_0", "stem_1", "stem_2", "stem_3"], help='List of stem subfolder names')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel processes')
    parser.add_argument('--plot_folder', default="DemoPage/Ours/CSV_015", help='Output folder for example plots')
    parser.add_argument('--example_count', type=int, default=1, help='Number of examples to save as plots')
    parser.add_argument('--max_length', type=float, default=None, help='Max duration in beats_json mode (seconds); inferred when omitted')
    parser.add_argument('--beat_type', type=str, default='madmom', choices=['madmom', 'librosa'], help='Beat detection backend')
    parser.add_argument('--madmom_fps', type=int, default=150, help='madmom fps parameter')
    parser.add_argument('--madmom_transition_lambda', type=int, default=100, help='madmom transition_lambda parameter')
    
    args = parser.parse_args()

    if args.beats_json_dir and os.path.exists(args.beats_json_dir):
        print(f"Analyzing beat JSON directory: {args.beats_json_dir}")
        analyze_folder_by_json(
            beats_json_dir=args.beats_json_dir,
            stems_list=args.stems,
            window_size=args.window_size,
            overlap=args.overlap,
            sr=args.sr,
            plot_folder=args.plot_folder,
            example_count=args.example_count,
            max_length=args.max_length,
            beat_type=args.beat_type,
            madmom_fps=args.madmom_fps,
            madmom_transition_lambda=args.madmom_transition_lambda
        )
    elif args.folder and os.path.exists(args.folder):
        if args.track:
            analyze_single_track(
                folder=args.folder,
                filename=args.track,
                stems_list=args.stems,
                window_size=args.window_size,
                overlap=args.overlap,
                sr=args.sr,
                plot_folder=args.plot_folder,
                beat_type=args.beat_type,
                madmom_fps=args.madmom_fps,
                madmom_transition_lambda=args.madmom_transition_lambda
            )
        else:
            analyze_folder_by_audio(
                folder=args.folder,
                stems_list=args.stems,
                window_size=args.window_size,
                overlap=args.overlap,
                sr=args.sr,
                num_workers=args.num_workers,
                plot_folder=args.plot_folder,
                example_count=args.example_count,
                beat_type=args.beat_type,
                madmom_fps=args.madmom_fps,
                madmom_transition_lambda=args.madmom_transition_lambda
            )
    else:
        print("Please specify --beats_json_dir or --folder")