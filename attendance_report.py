"""
============================================================
  Attendance Report Visualizer
  Run : python attendance_report.py
  Generates summary charts from attendance.csv
============================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path
import sys

ATTENDANCE_FILE = "attendance.csv"


def load_data(filepath: str) -> pd.DataFrame:
    """Load and parse the attendance CSV."""
    path = Path(filepath)
    if not path.exists():
        print(f"[ERROR] '{filepath}' not found. Run the attendance system first.")
        sys.exit(1)

    df = pd.read_csv(filepath, parse_dates=["Date"])
    if df.empty:
        print("[INFO] Attendance file is empty — no data to visualise.")
        sys.exit(0)

    # Combine Date + Time into a single datetime column
    df["DateTime"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"])
    df["Hour"]     = df["DateTime"].dt.hour
    df["DayName"]  = df["DateTime"].dt.day_name()
    return df


def report_daily_counts(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Bar chart: number of people present per day."""
    daily = df.groupby("Date")["Name"].nunique()
    dates = [d.strftime("%b %d") for d in daily.index]

    bars = ax.bar(dates, daily.values, color="#4CAF50", edgecolor="white", linewidth=0.8)
    ax.set_title("Daily Attendance Count", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("People Present", fontsize=10)
    ax.set_ylim(0, max(daily.values) + 2)
    ax.bar_label(bars, padding=3, fontsize=9, color="#333")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)


def report_per_person(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Horizontal bar chart: days each person attended."""
    per_person = df.groupby("Name")["Date"].nunique().sort_values()

    colors = plt.cm.Blues(
        [0.4 + 0.5 * (i / max(len(per_person) - 1, 1))
         for i in range(len(per_person))]
    )
    bars = ax.barh(per_person.index, per_person.values, color=colors, edgecolor="white")
    ax.set_title("Days Attended per Person", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Days", fontsize=10)
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)


def report_arrival_times(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Histogram of check-in hours across all records."""
    ax.hist(df["Hour"], bins=range(6, 22), color="#2196F3", edgecolor="white",
            linewidth=0.8, align="left")
    ax.set_title("Check-in Time Distribution", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Hour of Day (24 h)", fontsize=10)
    ax.set_ylabel("Check-ins", fontsize=10)
    ax.set_xticks(range(6, 22))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)


def report_heatmap(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Pivot heatmap: person × date presence."""
    pivot = df.groupby(["Name", "Date"])["Name"].count().unstack(fill_value=0)

    im = ax.imshow(pivot.values, cmap="YlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(
        [d.strftime("%b %d") for d in pivot.columns],
        rotation=45, ha="right", fontsize=8
    )
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title("Presence Heatmap (Name × Date)", fontsize=13, fontweight="bold", pad=10)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val  = pivot.values[i, j]
            text = "✓" if val > 0 else "–"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=10, color="#1a5276" if val > 0 else "#aaa")


def print_text_summary(df: pd.DataFrame) -> None:
    """Print a readable console summary."""
    today = datetime.now().strftime("%Y-%m-%d")
    print("\n" + "═"*52)
    print("  FULL ATTENDANCE REPORT")
    print("═"*52)

    # Overall stats
    total_records = len(df)
    unique_people  = df["Name"].nunique()
    unique_days    = df["Date"].nunique()
    print(f"  Total records   : {total_records}")
    print(f"  Unique people   : {unique_people}")
    print(f"  Days on record  : {unique_days}")

    # Per-person summary
    print(f"\n  {'Name':<18} {'Days Present':>12} {'First Seen':>12}")
    print(f"  {'─'*18} {'─'*12} {'─'*12}")
    for name, grp in df.groupby("Name"):
        days  = grp["Date"].nunique()
        first = grp["DateTime"].min().strftime("%b %d, %H:%M")
        print(f"  {name:<18} {days:>12} {first:>12}")

    # Today's attendees
    today_df = df[df["Date"].astype(str) == today]
    if not today_df.empty:
        print(f"\n  Today ({today}): {', '.join(today_df['Name'].tolist())}")
    else:
        print(f"\n  No entries for today ({today}).")

    print("═"*52 + "\n")


def main() -> None:
    print("\n[INFO] Loading attendance data …")
    df = load_data(ATTENDANCE_FILE)

    print_text_summary(df)

    # ── Dashboard layout ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "Facial Recognition Attendance — Dashboard",
        fontsize=16, fontweight="bold", y=1.01
    )
    fig.patch.set_facecolor("#f7f9fc")
    for ax in axes.flat:
        ax.set_facecolor("#ffffff")

    report_daily_counts(df,   axes[0, 0])
    report_per_person(df,     axes[0, 1])
    report_arrival_times(df,  axes[1, 0])
    report_heatmap(df,        axes[1, 1])

    plt.tight_layout(pad=2.5)
    out = "attendance_dashboard.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[INFO] Dashboard saved → {out}")
    plt.show()


if __name__ == "__main__":
    main()