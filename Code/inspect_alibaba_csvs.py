from pathlib import Path
import pandas as pd

DATA_DIR = Path(r"D:\other\ALIBABAQUATUM\Dataset")
OUT_MD = Path(r"D:\other\ALIBABAQUATUM\Dataset\README_DATASET_INSPECTION.md")

CSV_FILES = [
    "batch_instance.csv",
    "batch_task.csv",
    "container_meta.csv",
    "container_usage.csv",
    "machine_meta.csv",
    "machine_usage_bigger.csv",
]

# Alibaba v2018 likely headerless schemas
SCHEMA_HINTS = {
    "batch_instance.csv": [
        "instance_name", "task_name", "job_name", "task_type", "status",
        "start_time", "end_time", "machine_id", "seq_no", "total_seq_no",
        "cpu_avg", "cpu_max", "mem_avg", "mem_max"
    ],
    "batch_task.csv": [
        "task_name", "instance_num", "job_name", "task_type", "status",
        "start_time", "end_time", "plan_cpu", "plan_mem"
    ],
    "machine_meta.csv": [
        "machine_id", "time_stamp", "failure_domain_1", "failure_domain_2",
        "cpu_num", "mem_size", "status"
    ],
    "machine_usage_bigger.csv": [
        "machine_id", "time_stamp", "cpu_util_percent", "mem_util_percent",
        "mem_gps", "mkpi", "net_in", "net_out", "disk_io_percent"
    ],
    "container_meta.csv": [
        "container_id", "machine_id", "time_stamp", "app_du",
        "status", "cpu_request", "cpu_limit", "mem_size"
    ],
    "container_usage.csv": [
        "container_id", "machine_id", "time_stamp", "cpu_util_percent",
        "mem_util_percent", "cpi", "mem_gps", "mpki",
        "net_in", "net_out", "disk_io_percent"
    ],
}


def read_sample(path: Path, nrows: int = 10):
    """
    Reads a small sample safely.
    Tries normal header read first, then headerless read.
    """
    df_header = pd.read_csv(path, nrows=nrows, low_memory=False)

    # If pandas created numeric-like or unnamed columns, likely headerless or bad header
    bad_header = any(str(c).startswith("Unnamed") for c in df_header.columns)
    numeric_header = all(str(c).replace(".", "", 1).isdigit() for c in df_header.columns)

    if bad_header or numeric_header:
        df = pd.read_csv(path, nrows=nrows, header=None, low_memory=False)
    else:
        df = df_header

    return df


def read_headerless_with_schema(path: Path, fname: str, nrows: int = 10):
    df = pd.read_csv(path, nrows=nrows, header=None, low_memory=False)

    hints = SCHEMA_HINTS.get(fname)
    if hints and len(hints) == df.shape[1]:
        df.columns = hints
    else:
        df.columns = [f"col_{i}" for i in range(df.shape[1])]

    return df


def count_rows_fast(path: Path):
    """
    Counts rows without loading full file.
    For very large files this may take a little time, but uses low memory.
    """
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def md_table(df: pd.DataFrame):
    return df.to_markdown(index=False)


def main():
    lines = []
    lines.append("# Alibaba Cluster Trace Dataset Inspection")
    lines.append("")
    lines.append(f"Dataset folder: `{DATA_DIR}`")
    lines.append("")
    lines.append("This file reports file size, detected columns, and the first 10 rows of each CSV.")
    lines.append("")

    for fname in CSV_FILES:
        path = DATA_DIR / fname
        lines.append(f"## `{fname}`")
        lines.append("")

        if not path.exists():
            lines.append("**Status:** File not found.")
            lines.append("")
            continue

        size_mb = path.stat().st_size / (1024 ** 2)

        try:
            raw_sample = pd.read_csv(path, nrows=10, header=None, low_memory=False)
            n_cols = raw_sample.shape[1]
            row_count = count_rows_fast(path)

            df = read_headerless_with_schema(path, fname, nrows=10)

            lines.append(f"- **Size:** {size_mb:,.2f} MB")
            lines.append(f"- **Approx. rows including possible header:** {row_count:,}")
            lines.append(f"- **Columns:** {n_cols}")
            lines.append("")

            lines.append("### Column Names")
            lines.append("")
            for i, col in enumerate(df.columns):
                lines.append(f"{i+1}. `{col}`")
            lines.append("")

            lines.append("### First 10 Rows")
            lines.append("")
            lines.append(md_table(df))
            lines.append("")

        except Exception as e:
            lines.append(f"**Error reading file:** `{e}`")
            lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] README created at: {OUT_MD}")


if __name__ == "__main__":
    main()