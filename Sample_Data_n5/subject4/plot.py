import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

FILES = [
    "baseline/baseline.csv",
    "mat/mat.csv",
    "recovery/recovery.csv",
    "stroop/stroop.csv",
]

COLORS = {
    "baseline": "blue",
    "stroop": "yellow",
    "mat": "red",
    "recovery": "green",
}

def load_and_label(files):
    base = os.path.dirname(__file__)
    dfs = []
    for rel in files:
        path = os.path.join(base, rel)
        if not os.path.exists(path):
            print("Warning: file not found:", path)
            continue
        df = pd.read_csv(path)
        # ensure columns exist
        if "timestamp" not in df.columns or "ecg" not in df.columns:
            raise ValueError(f"{path} must contain 'timestamp' and 'ecg' columns")
        # label session by folder name (e.g., baseline, stroop, ...)
        session = os.path.basename(os.path.dirname(path))
        df["session"] = session
        df["source_file"] = os.path.basename(path)
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError("No input CSVs loaded")
    return pd.concat(dfs, ignore_index=True)

def parse_timestamps(df):
    # convert timestamp strings like "11:31:16.375979" to datetime (today's date)
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%H:%M:%S.%f")
    except Exception:
        df["timestamp"] = pd.to_datetime(df["timestamp"], infer_datetime_format=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "ecg"]).sort_values("timestamp").reset_index(drop=True)
    return df

def segments_from_times(ts, mult=3.0):
    # ts: pandas Series of datetime (sorted). returns list of (start_idx,end_idx)
    dt = ts.diff().dt.total_seconds()
    median_dt = dt.iloc[1:].median() if len(ts) > 1 else 0
    if pd.isna(median_dt) or median_dt == 0:
        threshold = 0.5  # fallback
    else:
        threshold = mult * median_dt
    segments = []
    if len(ts) == 0:
        return segments
    start = 0
    for i in range(1, len(ts)):
        if dt.iloc[i] > threshold:
            segments.append((start, i - 1))
            start = i
    segments.append((start, len(ts) - 1))
    return segments

def plot_continuous(df):
    df = parse_timestamps(df)
    fig, ax = plt.subplots(figsize=(14,6))

    # ensure predictable order
    sessions = ["baseline", "stroop", "mat", "recovery"]
    for sess in sessions:
        sdf = df[df["session"] == sess].copy()
        if sdf.empty:
            continue
        sdf = sdf.sort_values("timestamp").reset_index(drop=True)
        segments = segments_from_times(sdf["timestamp"])
        color = COLORS.get(sess, None) or "C0"
        # solid segments
        for s,e in segments:
            ax.plot(sdf["timestamp"].iloc[s:e+1], sdf["ecg"].iloc[s:e+1], linestyle='-', color=color, label=sess if s==segments[0][0] else "")
        # dotted connectors across gaps
        for i in range(len(segments)-1):
            end_idx = segments[i][1]
            start_idx = segments[i+1][0]
            ax.plot(
                [sdf["timestamp"].iloc[end_idx], sdf["timestamp"].iloc[start_idx]],
                [sdf["ecg"].iloc[end_idx], sdf["ecg"].iloc[start_idx]],
                linestyle=':', color=color, linewidth=1
            )

    # formatting
    ax.set_xlabel("Time")
    ax.set_ylabel("ECG")
    ax.set_title("ECG traces (solid = contiguous, dotted = gap connector)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S.%f"))
    handles, labels = ax.get_legend_handles_labels()
    # dedupe labels in legend
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

def main():
    df = load_and_label(FILES)
    # now df always has "session" so the KeyError won't happen
    plot_continuous(df)

if __name__ == "__main__":
    main()
