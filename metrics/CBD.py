import os
import json
import numpy as np
from madmom.evaluation.beats import (
    find_closest_matches,
    calc_absolute_errors
)
import librosa
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
import json
from tqdm import tqdm

def extract_beats_from_wavs(wav_files, sr=22050, beat_type="madmom", madmom_fps=200, madmom_transition_lambda=100):
    """
    Extract beat timestamps from all wav files.
    Supports madmom or librosa beat detection.
    """
    beats_list = []
    for wav in wav_files:
        if beat_type == "madmom":
            downbeat_proc = RNNDownBeatProcessor()
            downbeat_activation = downbeat_proc(wav)
            dbn_downbeat = DBNDownBeatTrackingProcessor(
                beats_per_bar=[3, 4], min_bpm=30, max_bpm=300,
                fps=madmom_fps, transition_lambda=madmom_transition_lambda)
            downbeats = dbn_downbeat(downbeat_activation)
            beats = downbeats[:, 0]
        elif beat_type == "librosa":
            y, sr_ = librosa.load(wav, sr=sr)
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr_)
            beats = librosa.frames_to_time(beats, sr=sr_)
        else:
            raise ValueError(f"Unsupported beat_type: {beat_type}")
        beats_list.append(np.array(beats))
    return beats_list

def compute_relative_beat_error(ref_beats, tgt_beats, pair_window_ratio=0.5):
    """
    Compute normalized beat error between reference and target track.
    """
    N = len(ref_beats)
    ref_beats = np.asarray(ref_beats)
    tgt_beats = np.asarray(tgt_beats)
    if N < 2 or len(tgt_beats) == 0:
        return []
    matches = find_closest_matches(ref_beats, tgt_beats)
    matched_beats = tgt_beats[matches]
    errors = calc_absolute_errors(ref_beats, tgt_beats, matches)
    intervals = np.diff(ref_beats)
    intervals_prev = np.concatenate([[intervals[0]], intervals])   # I_{n-1}
    intervals_next = np.concatenate([intervals, [intervals[-1]]]) # I_n

    left = ref_beats - intervals_prev * pair_window_ratio
    right = ref_beats + intervals_next * pair_window_ratio
    in_window = (matched_beats >= left) & (matched_beats < right)
    denominator = np.where(matched_beats >= ref_beats,
                           intervals_next * pair_window_ratio,
                           intervals_prev * pair_window_ratio)
    norm_errors = np.full(N, np.nan)
    norm_errors[in_window] = errors[in_window] / denominator[in_window]
    results = []
    for idx in range(N):
        results.append({
            'idx': idx,
            'ref': ref_beats[idx],
            'paired': bool(in_window[idx]),
            'error': errors[idx] if in_window[idx] else np.nan,
            'norm_error': norm_errors[idx]
        })
    return results

def summarize_error(results):
    """
    Summarize normalized error metrics.
    """
    paired_errors = [r['norm_error'] for r in results if r['paired']]
    avg_error = np.nanmean(paired_errors) if paired_errors else np.nan
    avg_std_error = np.nanstd(paired_errors) if paired_errors else np.nan
    avg_median_error = np.nanmedian(paired_errors) if paired_errors else np.nan
    avg_unpaired_ratio = 1.0 - len(paired_errors) / len(results) if results else 1.0
    return {
        'avg_error': avg_error,
        'avg_std_error': avg_std_error,
        'avg_median_error': avg_median_error,
        'avg_unpaired_ratio': avg_unpaired_ratio,
        'paired_count': len(paired_errors),
        'sample_count': len(results)
    }

def multi_track_consistency(beats_list, reference_type='all', pair_window_ratio=0.5):
    """
    Multi-track beat consistency metric.
    beats_list: list, beat timestamps for each track.
    reference_type: 'all' (each track as reference) or specified index.
    pair_window_ratio: proportion of beat interval as matching window.
    Returns statistics for each reference track.
    """
    N = len(beats_list)
    # Validate reference_type
    if reference_type == 'all':
        ref_indices = range(N)
    else:
        if not isinstance(reference_type, int) or reference_type < 0 or reference_type >= N:
            raise ValueError(f"reference_type must be 'all' or an integer in [0, {N-1}] (got {reference_type})")
        ref_indices = [reference_type]
    stats = []
    for i in ref_indices:
        # print(f"Processing reference track {i}...")
        ref_beats = beats_list[i]
        all_results = []
        for j in range(N):
            # print(f"Comparing with target track {j}...")
            if i == j: continue
            tgt_beats = beats_list[j]
            results = compute_relative_beat_error(ref_beats, tgt_beats, pair_window_ratio)
            # print(len(results), "results for track", j)
            all_results.extend(results)
            # print(len(results), "results for track", j, "with reference track", i)
        stat = summarize_error(all_results)
        stat['reference_track'] = i
        stats.append(stat)
        # print(stats)
    avg_std_error = np.nanmean([s['avg_std_error'] for s in stats])
    avg_error = np.nanmean([s['avg_error'] for s in stats])
    avg_median_error = np.nanmean([s['avg_median_error'] for s in stats])
    avg_unpaired_ratio = np.nanmean([s['avg_unpaired_ratio'] for s in stats])
    # print(avg_std_error, avg_error, avg_median_error, avg_unpaired_ratio)
    return stats, {
        'avg_std_error': avg_std_error,
        'avg_error': avg_error,
        'avg_median_error': avg_median_error,
        'avg_unpaired_ratio': avg_unpaired_ratio,
        'sample_count': len(stats)
    }

def load_beats_jsons(beats_json_folder, stems=('stem_0', 'stem_1', 'stem_2', 'stem_3')):
    """
    Load beats json files, return beat groups and track names.
    """
    stem_beats = []
    track_names = None
    for stem in stems:
        json_path = os.path.join(beats_json_folder, f"{stem}.json")
        with open(json_path, "r") as f:
            data = json.load(f)
            if isinstance(data, dict):
                data = [data]
            stem_beats.append([np.array(item['beats']) for item in data])
            if track_names is None:
                track_names = [item['name'] for item in data]
    num_tracks = len(stem_beats[0])
    beats_groups = []
    for i in range(num_tracks):
        beats_groups.append([stem_beats[s][i] for s in range(len(stems))])
    return beats_groups, track_names

def collect_wav_groups(folder, stems=('stem_0', 'stem_1', 'stem_2', 'stem_3')):
    """
    Traverse all stems in folder, return paired wav file lists.
    Returns wav_groups and track_names.
    """
    stem_files = []
    for stem in stems:
        stem_path = os.path.join(folder, stem)
        wavs = sorted([f for f in os.listdir(stem_path) if f.endswith('.wav')])
        stem_files.append(wavs)
    common_names = set(stem_files[0])
    for files in stem_files:
        common_names &= set(files)
    common_names = sorted(list(common_names))
    wav_groups = []
    for name in common_names:
        wav_groups.append([os.path.join(folder, stem, name) for stem in stems])
    return wav_groups, [os.path.splitext(n)[0] for n in common_names]

def batch_consistency_eval(beats_groups, reference_type='all', pair_window_ratio=0.5):
    """
    Batch evaluation: input multiple beats_list groups, output statistics and overall averages.
    """
    all_stats = []
    for beats_list in tqdm(beats_groups):
        stats, overall = multi_track_consistency(
            beats_list, reference_type=reference_type, pair_window_ratio=pair_window_ratio)
        all_stats.append({'stats': stats, 'overall': overall})
    all_avg_std_error = [x['overall']['avg_std_error'] for x in all_stats]
    all_avg_error = [x['overall']['avg_error'] for x in all_stats]
    all_avg_median_error = [x['overall']['avg_median_error'] for x in all_stats]
    all_avg_unpaired_ratio = [x['overall']['avg_unpaired_ratio'] for x in all_stats]
    summary = {
        'avg_std_error': np.nanmean(all_avg_std_error),
        'avg_error': np.nanmean(all_avg_error),
        'avg_median_error': np.nanmean(all_avg_median_error),
        'avg_unpaired_ratio': np.nanmean(all_avg_unpaired_ratio),
        'sample_count': len(all_stats)
    }
    return all_stats, summary

def parse_reference_type(arg_value, num_tracks=None):
    """
    Parse reference_type argument, allowing either 'all' or integer index.
    If num_tracks is given, also check range.
    """
    if str(arg_value).lower() == "all":
        return "all"
    try:
        idx = int(arg_value)
        if num_tracks is not None:
            if idx < 0 or idx >= num_tracks:
                raise ValueError(f"reference_type index {idx} out of range for {num_tracks} tracks")
        return idx
    except Exception:
        raise ValueError("reference_type must be 'all' or a valid track index integer")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--beats_json_folder', type=str, default=None, help='Path to beats json folder')
    parser.add_argument('--folder', type=str, default=None, help='Path to wav folder')
    parser.add_argument('--reference_type', type=str, default="all", help='"all" or index of reference track, eg "0", "1", etc.')
    parser.add_argument('--pair_window_ratio', type=float, default=0.5, help='Matching window ratio')
    parser.add_argument('--single_files', nargs='+', default=None, help='List of 4 wav files')
    parser.add_argument('--beat_type', type=str, default="madmom", choices=["madmom", "librosa"], help='Beat extraction method: madmom or librosa')
    parser.add_argument('--madmom_fps', type=int, default=150, help='madmom fps')
    parser.add_argument('--madmom_transition_lambda', type=float, default=100, help='madmom transition lambda')
    args = parser.parse_args()

    stems = ('stem_0', 'stem_1', 'stem_2', 'stem_3')
    if args.single_files is not None and len(args.single_files) == 4:
        wav_files = args.single_files
        beats_list = extract_beats_from_wavs(
            wav_files,
            beat_type=args.beat_type,
            madmom_fps=args.madmom_fps,
            madmom_transition_lambda=args.madmom_transition_lambda
        )
        reference_type = parse_reference_type(args.reference_type, num_tracks=len(beats_list))
        stats, overall = multi_track_consistency(
            beats_list, reference_type=reference_type, pair_window_ratio=args.pair_window_ratio)
        print("Single track consistency evaluation:")
        for s in stats:
            print(f"Ref track {s['reference_track']}: avg_error={s['avg_error']:.4f}, std_error={s['avg_std_error']:.4f}, median_error={s['avg_median_error']:.4f}, unpaired_ratio={s['avg_unpaired_ratio']:.4f}")
        print("Summary:")
        for k, v in overall.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    elif args.beats_json_folder is not None and os.path.exists(args.beats_json_folder):
        beats_groups, track_names = load_beats_jsons(args.beats_json_folder, stems)
        reference_type = parse_reference_type(args.reference_type, num_tracks=len(stems))
        all_stats, summary = batch_consistency_eval(beats_groups, reference_type=reference_type, pair_window_ratio=args.pair_window_ratio)
        print("Batch beats JSON consistency evaluation:")
        print(f"Total tracks: {len(beats_groups)}")
        print("Summary:")
        for k, v in summary.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    elif args.folder is not None and os.path.exists(args.folder):
        wav_groups, track_names = collect_wav_groups(args.folder, stems)
        beats_groups = []
        print("Extracting beats for batch...")
        for wav_files in tqdm(wav_groups):
            print(wav_files)
            beats_list = extract_beats_from_wavs(
                wav_files,
                beat_type=args.beat_type,
                madmom_fps=args.madmom_fps,
                madmom_transition_lambda=args.madmom_transition_lambda
            )
            beats_groups.append(beats_list)
        reference_type = parse_reference_type(args.reference_type, num_tracks=len(stems))
        all_stats, summary = batch_consistency_eval(beats_groups, reference_type=reference_type, pair_window_ratio=args.pair_window_ratio)
        print("Batch wav consistency evaluation:")
        print(f"Total tracks: {len(beats_groups)}")
        print("Summary:")
        for k, v in summary.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    else:
        print("Please specify --single_files, --beats_json_folder or --folder")

    