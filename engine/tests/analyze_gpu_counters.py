"""Summarize xctrace GPU counters during a target process's command-buffer span."""

import argparse
from array import array
from collections import Counter, defaultdict
import json
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


DEFAULT_COUNTERS = (
    "ALU Utilization",
    "Buffer Load Utilization",
    "Compute Occupancy",
    "GPU Read Bandwidth",
    "GPU Last Level Cache Utilization",
)


def table_xpath(run_number, schema):
    return (
        f'/trace-toc/run[@number="{run_number}"]'
        f'/data/table[@schema="{schema}"]'
    )


def export_table(trace, run_number, schema):
    command = [
        "xcrun",
        "xctrace",
        "export",
        "--input",
        str(trace),
        "--xpath",
        table_xpath(run_number, schema),
        "--quiet",
    ]
    return subprocess.run(command, check=True, capture_output=True).stdout


def resolve(cell, refs, convert):
    if "ref" in cell.attrib:
        return refs[cell.attrib["ref"]]
    value = convert(cell.text or "")
    if "id" in cell.attrib:
        refs[cell.attrib["id"]] = value
    return value


def counter_info(xml_data):
    root = ET.fromstring(xml_data)
    refs = {}
    by_name = {}
    for row in root.findall(".//row"):
        cells = list(row)
        if len(cells) < 11:
            continue
        values = []
        for cell in cells:
            if "ref" in cell.attrib:
                value = refs[cell.attrib["ref"]]
            elif "id" in cell.attrib:
                value = cell.text or ""
                refs[cell.attrib["id"]] = value
            else:
                value = ""
            values.append(value)
        name = values[2]
        if name:
            by_name[name] = {
                "id": int(values[1]),
                "type": values[7],
                "ring_buffer_count": int(values[8]),
                "requires_weighted_accumulation": values[9] == "1",
                "sample_interval": int(values[10]),
                "description": values[5],
            }
    return by_name


def target_command_window(xml_data):
    root = ET.fromstring(xml_data)
    numeric_refs = {}
    process_refs = {}
    pids = set()
    submissions = []

    for row in root.findall(".//row"):
        cells = list(row)
        if len(cells) < 8:
            continue
        resolved_numbers = {}
        for cell in row.iter():
            if cell.tag in {
                "start-time",
                "duration",
                "uint32",
                "uint64",
                "metal-command-buffer-id",
                "tid",
                "pid",
            } and ("id" in cell.attrib or "ref" in cell.attrib):
                resolved_numbers[id(cell)] = resolve(cell, numeric_refs, int)
        start = resolved_numbers[id(cells[0])]
        duration = resolved_numbers[id(cells[1])]
        encoders = resolved_numbers[id(cells[5])]

        process = cells[7]
        if "ref" in process.attrib:
            pid = process_refs[process.attrib["ref"]]
        else:
            pid_text = process.findtext("pid")
            pid = int(pid_text) if pid_text else None
            if "id" in process.attrib:
                process_refs[process.attrib["id"]] = pid
        if pid is not None:
            pids.add(pid)
        if pid is not None and encoders > 0:
            submissions.append((start, start + duration))

    if len(pids) != 1:
        raise ValueError(f"expected one target process in command-buffer table, found {sorted(pids)}")
    if not submissions:
        raise ValueError("no encoded command-buffer submissions found for the target process")
    return next(iter(pids)), min(start for start, _ in submissions), max(end for _, end in submissions), len(submissions)


def gpu_active_time(xml_data, start_ns, end_ns):
    root = ET.fromstring(xml_data)
    numeric_refs = {}
    text_refs = {}
    device_refs = {}
    totals = {"Active": 0, "Idle": 0, "Other": 0}

    for row in root.findall(".//row"):
        cells = list(row)
        if len(cells) < 7:
            continue
        start = resolve(cells[0], numeric_refs, int)
        duration = resolve(cells[1], numeric_refs, int)
        state = resolve(cells[2], text_refs, str)
        device = resolve(cells[6], device_refs, str)
        if device != "M1 Max":
            continue
        overlap = max(0, min(start + duration, end_ns) - max(start, start_ns))
        if state in ("Active", "Idle"):
            totals[state] += overlap
        else:
            totals["Other"] += overlap

    window_ns = end_ns - start_ns
    return {
        "active_ns": totals["Active"],
        "idle_ns": totals["Idle"],
        "other_ns": totals["Other"],
        "window_ns": window_ns,
        "active_percent": 100 * totals["Active"] / window_ns,
    }


def extend_to(values, index, fill):
    if index >= len(values):
        values.extend([fill] * (index + 1 - len(values)))


def counter_samples(trace, run_number, wanted_ids, weighted_ids, start_ns, end_ns):
    command = [
        "xcrun",
        "xctrace",
        "export",
        "--input",
        str(trace),
        "--xpath",
        table_xpath(run_number, "gpu-counter-value"),
        "--quiet",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    time_refs = {}
    uint32_values = array("I", [0])
    fixed_values = array("d", [0.0])
    samples = {counter_id: [] for counter_id in wanted_ids}
    stack = []

    try:
        for event, element in ET.iterparse(process.stdout, events=("start", "end")):
            if event == "start":
                stack.append(element)
                continue
            if element.tag == "row":
                cells = list(element)
                if len(cells) >= 3:
                    timestamp = resolve(cells[0], time_refs, int)
                    resolved_uint32 = {}
                    for cell in cells:
                        if cell.tag != "uint32":
                            continue
                        if "ref" in cell.attrib:
                            value_u32 = uint32_values[int(cell.attrib["ref"])]
                        else:
                            value_u32 = int(cell.text)
                        resolved_uint32[id(cell)] = value_u32
                        if "id" in cell.attrib:
                            value_id = int(cell.attrib["id"])
                            extend_to(uint32_values, value_id, 0)
                            uint32_values[value_id] = value_u32
                    counter_id = resolved_uint32[id(cells[1])]
                    value_cell = cells[2]
                    if "ref" in value_cell.attrib:
                        value = fixed_values[int(value_cell.attrib["ref"])]
                    else:
                        value = float(value_cell.text)
                        value_id = int(value_cell.attrib["id"])
                        extend_to(fixed_values, value_id, 0.0)
                        fixed_values[value_id] = value
                    if counter_id in wanted_ids and start_ns <= timestamp < end_ns:
                        if counter_id in weighted_ids:
                            samples[counter_id].append(
                                (
                                    timestamp,
                                    value,
                                    resolved_uint32[id(cells[4])],
                                    resolved_uint32[id(cells[5])],
                                )
                            )
                        else:
                            samples[counter_id].append(value)
                    stack[-2].remove(element)
            stack.pop()
    finally:
        if process.stdout:
            process.stdout.close()
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    return samples


def summarize_values(values):
    ordered = sorted(values)
    return {
        "samples": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p90": ordered[int(0.9 * (len(ordered) - 1))],
        "min": ordered[0],
        "max": ordered[-1],
        "zero_percent": 100 * sum(value == 0 for value in values) / len(values),
    }


def summarize(values, start_ns, end_ns):
    if not values or not isinstance(values[0], tuple):
        return summarize_values(values)

    by_sample = defaultdict(list)
    for timestamp, value, sample_index, ring_index in values:
        by_sample[sample_index].append((timestamp, value, ring_index))

    sample_points = []
    ring_counts = Counter()
    for records in by_sample.values():
        timestamp = round(statistics.mean(record[0] for record in records))
        value = statistics.mean(record[1] for record in records)
        ring_counts[len({record[2] for record in records})] += 1
        sample_points.append((timestamp, value))

    by_timestamp = defaultdict(list)
    for timestamp, value in sample_points:
        by_timestamp[timestamp].append(value)
    points = sorted((timestamp, statistics.mean(group)) for timestamp, group in by_timestamp.items())
    gaps = [right[0] - left[0] for left, right in zip(points, points[1:]) if right[0] > left[0]]
    last_sample_ns = int(statistics.median(gaps)) if gaps else 0
    weighted_total = 0.0
    coverage_ns = 0
    for index, (timestamp, value) in enumerate(points):
        next_timestamp = points[index + 1][0] if index + 1 < len(points) else timestamp + last_sample_ns
        left = max(start_ns, timestamp)
        right = min(end_ns, next_timestamp)
        if right > left:
            weighted_total += value * (right - left)
            coverage_ns += right - left

    group_values = [value for _, value in sample_points]
    return {
        "raw_sample_rows": summarize_values([value for _, value, _, _ in values]),
        "sample_index_groups": len(sample_points),
        "ring_count_distribution": dict(sorted(ring_counts.items())),
        "ring_averaged_sample_values": summarize_values(group_values),
        "timestamp_weighted_mean": weighted_total / coverage_ns if coverage_ns else None,
        "timestamp_weighted_coverage_seconds": coverage_ns / 1e9,
        "aggregation_note": (
            "Values averaged across ring-buffer indices per sample index, then weighted by observed timestamp gaps; "
            "this is a local estimate, not an Instruments-provided weighted aggregate."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--counter", action="append", dest="counters")
    args = parser.parse_args()

    names = args.counters or DEFAULT_COUNTERS
    info = counter_info(export_table(args.trace, args.run, "gpu-counter-info"))
    missing = [name for name in names if name not in info]
    if missing:
        parser.error(f"counters not available in this trace: {', '.join(missing)}")

    pid, start_ns, end_ns, submission_count = target_command_window(
        export_table(args.trace, args.run, "metal-application-command-buffer-submissions")
    )
    active = gpu_active_time(
        export_table(args.trace, args.run, "metal-gpu-state-intervals"), start_ns, end_ns
    )
    selected = {info[name]["id"]: name for name in names}
    samples = counter_samples(
        args.trace, args.run, set(selected), set(selected), start_ns, end_ns
    )
    empty = [selected[counter_id] for counter_id, values in samples.items() if not values]
    if empty:
        parser.error(f"no counter samples landed in the target window: {', '.join(empty)}")

    report = {
        "trace": str(args.trace),
        "target_pid": pid,
        "target_command_buffers": submission_count,
        "target_window_seconds": [start_ns / 1e9, end_ns / 1e9],
        "gpu_state_scope": "M1 Max GPU-wide; active-state overlap with target command-buffer span",
        "gpu_state": {
            key: value / 1e9 if key.endswith("_ns") else value
            for key, value in active.items()
        },
        "counter_scope": "M1 Max GPU-wide; samples within target command-buffer span",
        "counter_aggregation_note": (
            "Samples are averaged across ring-buffer indices per sample index and then weighted by observed "
            "timestamp gaps; these local estimates are not Instruments-provided weighted aggregates."
        ),
        "counters": {
            selected[counter_id]: {
                "metadata": info[selected[counter_id]],
                "statistics": summarize(values, start_ns, end_ns),
            }
            for counter_id, values in samples.items()
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
