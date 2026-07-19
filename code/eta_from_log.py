#!/usr/bin/env python3
"""Estimate training ETA from a DreamerV3/BTWM log file."""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import re
import statistics
import sys
from typing import Iterable


START_LINE_RE = re.compile(r"^\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")
RUN_STEPS_RE = re.compile(r"(?:^|\s)steps=(?P<steps>\d+)(?:\s|$)")
COMMAND_STEPS_RE = re.compile(r"--run\.steps\s+(?P<steps>\d+)")
AGENT_STEP_RE = re.compile(r"\[Agent Step (?P<step>[0-9_]+)\]")
FPS_POLICY_RE = re.compile(r"fps/policy (?P<fps>-?\d+(?:\.\d+)?)")
TIMESTAMP_RE = re.compile(r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate remaining training time from a log file.",
    )
    parser.add_argument("logfile", type=pathlib.Path, help="Path to the training log.")
    parser.add_argument(
        "--fps-window",
        type=int,
        default=10,
        help="Number of recent fps/policy values to average (default: 10).",
    )
    parser.add_argument(
        "--target-steps",
        type=int,
        default=None,
        help="Override total training steps if the log does not contain them.",
    )
    return parser.parse_args()


def parse_timestamp(text: str) -> dt.datetime:
    return dt.datetime.strptime(text, "%Y-%m-%d %H:%M:%S")


def load_lines(path: pathlib.Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise SystemExit(f"Failed to read log file: {exc}") from exc


def find_first_match(lines: Iterable[str], pattern: re.Pattern[str], group: str) -> str | None:
    for line in lines:
        match = pattern.search(line)
        if match:
            return match.group(group)
    return None


def find_last_agent_step(lines: Iterable[str]) -> int | None:
    latest = None
    for line in lines:
        match = AGENT_STEP_RE.search(line)
        if match:
            latest = int(match.group("step").replace("_", ""))
    return latest


def find_recent_fps(lines: Iterable[str], window: int) -> list[float]:
    fps_values: list[float] = []
    for line in lines:
        match = FPS_POLICY_RE.search(line)
        if match:
            fps_values.append(float(match.group("fps")))
    if window <= 0:
        return fps_values
    return fps_values[-window:]


def find_last_timestamp(lines: Iterable[str]) -> dt.datetime | None:
    latest = None
    for line in lines:
        for match in TIMESTAMP_RE.finditer(line):
            latest = parse_timestamp(match.group("ts"))
    return latest


def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    sign = "-" if total < 0 else ""
    total = abs(total)
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{sign}{days}d {hours}h {minutes}m {secs}s"
    if hours:
        return f"{sign}{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{sign}{minutes}m {secs}s"
    return f"{sign}{secs}s"


def main() -> int:
    args = parse_args()
    lines = load_lines(args.logfile)
    if not lines:
        raise SystemExit("Log file is empty.")

    start_ts_text = find_first_match(lines, START_LINE_RE, "ts")
    start_ts = parse_timestamp(start_ts_text) if start_ts_text else None

    run_steps_text = (
        find_first_match(lines, RUN_STEPS_RE, "steps")
        or find_first_match(lines, COMMAND_STEPS_RE, "steps")
    )
    target_steps = args.target_steps or (int(run_steps_text) if run_steps_text else None)
    latest_step = find_last_agent_step(lines)
    recent_fps = find_recent_fps(lines, args.fps_window)
    last_log_ts = find_last_timestamp(lines)

    if latest_step is None:
        raise SystemExit("Could not find any 'Agent Step' entries in the log.")
    if not recent_fps:
        raise SystemExit("Could not find any 'fps/policy' entries in the log.")
    if target_steps is None:
        raise SystemExit("Could not determine total steps. Pass --target-steps.")

    avg_fps = statistics.mean(recent_fps)
    remaining_steps = max(target_steps - latest_step, 0)
    remaining_seconds = remaining_steps / avg_fps if avg_fps > 0 else float("inf")
    eta_from_now = dt.datetime.now() + dt.timedelta(seconds=remaining_seconds)

    print(f"logfile: {args.logfile}")
    if start_ts:
        print(f"start_time: {start_ts:%Y-%m-%d %H:%M:%S}")
    if last_log_ts:
        print(f"last_log_timestamp: {last_log_ts:%Y-%m-%d %H:%M:%S}")
    print(f"target_steps: {target_steps:,}")
    print(f"current_step: {latest_step:,}")
    print(f"remaining_steps: {remaining_steps:,}")
    print(f"recent_fps_policy_avg({len(recent_fps)}): {avg_fps:.2f} step/s")
    print(f"eta_from_now: {format_duration(remaining_seconds)}")
    print(f"estimated_finish_time: {eta_from_now:%Y-%m-%d %H:%M:%S}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
