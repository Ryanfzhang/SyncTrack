from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
import librosa
import numpy as np
import os
import json
from tqdm import tqdm

def extract_beats_from_wav(audio_file, sr=22050, beat_type="madmom", madmom_fps=200, madmom_transition_lambda=100):
    if beat_type == "madmom":
        downbeat_proc = RNNDownBeatProcessor()
        downbeat_activation = downbeat_proc(audio_file)
        dbn_downbeat = DBNDownBeatTrackingProcessor(
            beats_per_bar=[3, 4], min_bpm=30, max_bpm=300,
            fps=madmom_fps, transition_lambda=madmom_transition_lambda)
        downbeats = dbn_downbeat(downbeat_activation)
        beats = downbeats[:, 0]
    else:
        y, sr_ = librosa.load(audio_file, sr=sr)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr_)
        beats = librosa.frames_to_time(beat_frames, sr=sr_)
    return np.array(beats)

def estimate_beat_intervals(audio_file, sr=22050, beat_type="madmom", madmom_fps=200, madmom_transition_lambda=100):
    beat_times = extract_beats_from_wav(audio_file, sr=sr, beat_type=beat_type, madmom_fps=madmom_fps, madmom_transition_lambda=madmom_transition_lambda)
    if len(beat_times) < 2:
        return [], beat_times.tolist()
    beat_intervals = np.diff(beat_times)
    return beat_intervals.tolist(), beat_times.tolist()

def estimate_beat_intervals_from_times(beat_times):
    beat_times = np.asarray(beat_times)
    if len(beat_times) < 2:
        return []
    beat_intervals = np.diff(beat_times)
    return beat_intervals.tolist()

def process_stem_folder(folder_path, sr=22050, beat_type="madmom", madmom_fps=200, madmom_transition_lambda=100, stems=('stem_0', 'stem_1', 'stem_2', 'stem_3')):
    """
    Batch process folders structured as folder/stem_x/*.wav.
    - Compute statistics for each wav.
    - Aggregate per stem.
    - Then average across stems.
    """
    stem_results = {}
    stem_avg_beat_counts = []
    stem_avg_interval_means = []
    stem_avg_interval_stds = []

    for stem in tqdm(stems):
        stem_path = os.path.join(folder_path, stem)
        if not os.path.exists(stem_path) or not os.path.isdir(stem_path):
            print(f"{stem_path} does not exist or is not a directory, skipping.")
            continue
        beat_counts = []
        interval_means = []
        interval_stds = []
        file_count = 0

        for fname in tqdm(os.listdir(stem_path)):
            if fname.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                file_path = os.path.join(stem_path, fname)
                intervals, beats = estimate_beat_intervals(
                    file_path, sr=sr, beat_type=beat_type,
                    madmom_fps=madmom_fps, madmom_transition_lambda=madmom_transition_lambda)
                if beats is not None and len(beats) > 0:
                    beat_counts.append(len(beats))
                if intervals is not None and len(intervals) > 0:
                    interval_means.append(np.mean(intervals))
                    interval_stds.append(np.std(intervals))
                file_count += 1

        avg_beat_count = np.mean(beat_counts) if beat_counts else None
        avg_interval_mean = np.mean(interval_means) if interval_means else None
        avg_interval_std = np.mean(interval_stds) if interval_stds else None

        print(f"[{stem}] Processed {file_count} files")
        print(f"[{stem}] Avg beat count: {avg_beat_count}")
        print(f"[{stem}] Avg interval mean: {avg_interval_mean}")
        print(f"[{stem}] Avg interval std: {avg_interval_std}")

        stem_results[stem] = {
            'file_count': file_count,
            'avg_beat_count': avg_beat_count,
            'avg_interval_mean': avg_interval_mean,
            'avg_interval_std': avg_interval_std
        }
        # Record the per-stem averages for later aggregation.
        if avg_beat_count is not None:
            stem_avg_beat_counts.append(avg_beat_count)
        if avg_interval_mean is not None:
            stem_avg_interval_means.append(avg_interval_mean)
        if avg_interval_std is not None:
            stem_avg_interval_stds.append(avg_interval_std)

    # Average across stems.
    overall_avg_beat_count = np.mean(stem_avg_beat_counts) if stem_avg_beat_counts else None
    overall_avg_interval_mean = np.mean(stem_avg_interval_means) if stem_avg_interval_means else None
    overall_avg_interval_std = np.mean(stem_avg_interval_stds) if stem_avg_interval_stds else None

    print(f"[ALL STEMS] Avg of stem avg beat count: {overall_avg_beat_count}")
    print(f"[ALL STEMS] Avg of stem avg interval mean: {overall_avg_interval_mean}")
    print(f"[ALL STEMS] Avg of stem avg interval std: {overall_avg_interval_std}")

    return {
        'stem_results': stem_results,
        'overall_avg_beat_count': overall_avg_beat_count,
        'overall_avg_interval_mean': overall_avg_interval_mean,
        'overall_avg_interval_std': overall_avg_interval_std
    }

def process_beats_json_stem_folder(folder_path, stems=('stem_0.json','stem_1.json','stem_2.json','stem_3.json')):
    """
    Process folders containing files such as folder/stem_x.json.
    - Compute stats per track.
    - Aggregate per stem.
    - Then average across stems.
    """
    stem_results = {}
    stem_avg_beat_counts = []
    stem_avg_interval_means = []
    stem_avg_interval_stds = []

    for stem_json in stems:
        stem_path = os.path.join(folder_path, stem_json)
        if not os.path.exists(stem_path):
            print(f"{stem_path} doesn't exist, skipping.")
            continue

        with open(stem_path, "r") as f:
            data = json.load(f)
            if isinstance(data, dict):  # single track json
                data = [data]
        
        beat_counts = []
        interval_means = []
        interval_stds = []
        file_count = 0

        for track_obj in data:
            beats = track_obj.get("beats", [])
            intervals = estimate_beat_intervals_from_times(beats)
            if beats and len(beats) > 0:
                beat_counts.append(len(beats))
            if intervals and len(intervals) > 0:
                interval_means.append(np.mean(intervals))
                interval_stds.append(np.std(intervals))
            file_count += 1

        avg_beat_count = np.mean(beat_counts) if beat_counts else None
        avg_interval_mean = np.mean(interval_means) if interval_means else None
        avg_interval_std = np.mean(interval_stds) if interval_stds else None

        print(f"[{stem_json}] Processed {file_count} tracks")
        print(f"[{stem_json}] Avg beat count: {avg_beat_count}")
        print(f"[{stem_json}] Avg interval mean: {avg_interval_mean}")
        print(f"[{stem_json}] Avg interval std: {avg_interval_std}")

        stem_results[stem_json] = {
            'file_count': file_count,
            'avg_beat_count': avg_beat_count,
            'avg_interval_mean': avg_interval_mean,
            'avg_interval_std': avg_interval_std
        }
        if avg_beat_count is not None:
            stem_avg_beat_counts.append(avg_beat_count)
        if avg_interval_mean is not None:
            stem_avg_interval_means.append(avg_interval_mean)
        if avg_interval_std is not None:
            stem_avg_interval_stds.append(avg_interval_std)

    overall_avg_beat_count = np.mean(stem_avg_beat_counts) if stem_avg_beat_counts else None
    overall_avg_interval_mean = np.mean(stem_avg_interval_means) if stem_avg_interval_means else None
    overall_avg_interval_std = np.mean(stem_avg_interval_stds) if stem_avg_interval_stds else None

    print(f"[ALL STEMS] Avg of stem avg beat count: {overall_avg_beat_count}")
    print(f"[ALL STEMS] Avg of stem avg interval mean: {overall_avg_interval_mean}")
    print(f"[ALL STEMS] Avg of stem avg interval std: {overall_avg_interval_std}")

    return {
        'stem_results': stem_results,
        'overall_avg_beat_count': overall_avg_beat_count,
        'overall_avg_interval_mean': overall_avg_interval_mean,
        'overall_avg_interval_std': overall_avg_interval_std
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', type=str, help='Path to a single audio file')
    parser.add_argument('--folder', type=str, help='Root directory that contains stem subfolders')
    parser.add_argument('--beats_json_folder', type=str, help='Directory containing beat JSON files such as stem_0.json')
    parser.add_argument('--beat_type', type=str, default='librosa', choices=['madmom', 'librosa'], help='Beat detection backend: madmom or librosa')
    parser.add_argument('--madmom_fps', type=int, default=150, help='madmom fps parameter')
    parser.add_argument('--madmom_transition_lambda', type=int, default=100, help='madmom transition_lambda parameter')
    parser.add_argument('--sr', type=int, default=22050, help='Audio sample rate (used by librosa)')
    args = parser.parse_args()
    
    if args.audio:
        intervals, beats = estimate_beat_intervals(
            args.audio, sr=args.sr, beat_type=args.beat_type,
            madmom_fps=args.madmom_fps, madmom_transition_lambda=args.madmom_transition_lambda)
        print(f"Beat count: {len(beats)}")
        print(f"Mean interval: {np.mean(intervals) if intervals else 'N/A'}")
        print(f"Std interval: {np.std(intervals) if intervals else 'N/A'}")
    elif args.folder:
        process_stem_folder(
            args.folder, sr=args.sr, beat_type=args.beat_type,
            madmom_fps=args.madmom_fps, madmom_transition_lambda=args.madmom_transition_lambda)
    elif args.beats_json_folder:
        process_beats_json_stem_folder(args.beats_json_folder)
    else:
        print("Please specify --audio, --folder, or --beats_json_folder")