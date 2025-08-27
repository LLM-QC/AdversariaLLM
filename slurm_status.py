#!/usr/bin/env python3
"""
Simple SLURM Job Status Visualizer for Multirun Directories
Shows a nice overview of job completion status with color coding.

Usage:
    python3 slurm_status.py           # Show overview of last 20 runs
    python3 slurm_status.py -n 50     # Show last 50 runs
    python3 slurm_status.py -d        # Show detailed view with individual job info
"""

import argparse
import os
import re
import subprocess
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path


# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'    # Success
    RED = '\033[91m'      # Failed
    YELLOW = '\033[93m'   # Running/Unknown
    BLUE = '\033[94m'     # Info
    PURPLE = '\033[95m'   # Header
    CYAN = '\033[96m'     # Timestamp
    WHITE = '\033[97m'    # Default
    BOLD = '\033[1m'      # Bold
    END = '\033[0m'       # Reset

@lru_cache(maxsize=None)
def get_jobs():
    """Get all jobs in the SLURM queue"""
    result = subprocess.run(['squeue', '-l', '-r'], capture_output=True, text=True, timeout=10)
    jobs = {}
    for line in result.stdout.splitlines()[1:]:
        id = line.split()[0]
        status = line.split()[4]
        jobs[id] = status
    return jobs

def check_job_status(job_dir):
    """Check the status of a single SLURM job"""
    # Look for log files with the correct naming pattern
    log_files = list(job_dir.glob("*_log.out"))
    job_id = job_dir.name

    if not log_files:
        # Check if job is still in SLURM queue
        try:
            jobs = get_jobs()
            if job_id in jobs and jobs[job_id] == "PENDING":
                return "PENDING", "Job is currently pending", None, None
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass  # squeue not available or timed out
        return "UNKNOWN", "No log file found", None, None

    log_out = log_files[0]
    log_err = log_out.with_suffix('.err')

    try:
        # Check the last few lines of the output log
        with open(log_out, 'r') as f:
            lines = f.readlines()

        with open(log_err, 'r') as f:
            err_lines = "\n".join(f.readlines()).lower()

        # Look for completion indicators in the last 10 lines
        last_lines = ''.join(lines[-10:]).lower()

        if any("Attack logged to" in line for line in lines):
            return "SUCCESS", "Completed successfully", None, None
        elif any("Skipping" in line and " because it already exists" in line for line in lines):
            return "SUCCESS", "Skipped, already exists", None, None
        # Check if log file was modified in the last 2 minutes (job is likely still running)
        if time.time() - os.path.getmtime(log_out) < 120:
            return "RUNNING", "Job is currently running", None, None
        # Check if job is still in SLURM queue
        try:
            jobs = get_jobs()
            if job_id in jobs and jobs[job_id] == "RUNNING":
                return "RUNNING", "Job is currently running", None, None
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass  # squeue not available or timed out
        if any(word in err_lines for word in ["error", "failed", "exception", "traceback"]):
            return "FAILED", "Error detected in logs", str(log_out.resolve()), str(log_err.resolve())
        elif any(word in last_lines for word in ["error", "failed", "exception", "traceback"]):
            return "FAILED", "Error detected in logs", str(log_out.resolve()), str(log_err.resolve())
        else:
            # Check the last few lines of the output log
            last_lines = ''.join(err_lines[-10:]).lower()
            if any(word in last_lines for word in ["error", "failed", "exception", "traceback"]):
                if "out of memory" in last_lines:
                    return "FAILED", "Out of memory", str(log_out.resolve()), str(log_err.resolve())
                if "cancelled at" in last_lines:
                    return "FAILED", "Cancelled", str(log_out.resolve()), str(log_err.resolve())
                return "FAILED", "Error detected in logs", str(log_out.resolve()), str(log_err.resolve())
        return "UNKNOWN", "Status unclear", str(log_out.resolve()), str(log_err.resolve())

    except Exception as e:
        return "ERROR", f"Could not read logs: {str(e)}", None, None

def get_run_timestamp(run_path):
    """Extract timestamp from run path for sorting"""
    try:
        date_str = run_path.parent.name  # YYYY-MM-DD
        time_str = run_path.name         # HH-MM-SS
        return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H-%M-%S")
    except:
        return datetime.min

def get_multirun_status(max_runs=20, detailed=False, failed=False):
    """Get status of all multirun jobs"""
    multirun_path = Path("multirun")

    if not multirun_path.exists():
        print(f"{Colors.RED}Error: multirun directory not found{Colors.END}")
        return

    # Find all .submitit directories
    submitit_dirs = list(multirun_path.glob("*/*/.submitit"))

    if not submitit_dirs:
        print(f"{Colors.YELLOW}No SLURM jobs found in multirun directory{Colors.END}")
        return

    # Sort by timestamp (most recent first)
    submitit_dirs.sort(key=lambda x: get_run_timestamp(x.parent), reverse=True)

    print(f"{Colors.PURPLE}{Colors.BOLD}🚀 SLURM Job Status Overview{Colors.END}")
    print(f"{Colors.PURPLE}{'='*80}{Colors.END}")

    total_runs = 0
    successful_runs = 0
    failed_runs = 0
    running_runs = 0
    pending_runs = 0

    for submitit_dir in submitit_dirs[:max_runs]:
        run_path = submitit_dir.parent
        run_name = f"{run_path.parent.name}/{run_path.name}"
        timestamp = get_run_timestamp(run_path)

        # Find all job directories
        job_dirs = [d for d in submitit_dir.iterdir() if d.is_dir() and re.match(r'\d+_\d+', d.name)]

        if not job_dirs:
            continue

        total_runs += 1

        # Check status of all jobs in this run
        job_statuses = []
        for job_dir in sorted(job_dirs):
            status, message, log_out_path, log_err_path = check_job_status(job_dir)
            job_statuses.append((job_dir.name, status, message, log_out_path, log_err_path))

        # Determine overall run status
        success_count = sum(1 for _, status, _, _, _ in job_statuses if status == "SUCCESS")
        failed_count = sum(1 for _, status, _, _, _ in job_statuses if status == "FAILED")
        running_count = sum(1 for _, status, _, _, _ in job_statuses if status == "RUNNING")
        pending_count = sum(1 for _, status, _, _, _ in job_statuses if status == "PENDING")
        unknown_count = sum(1 for _, status, _, _, _ in job_statuses if status not in ["SUCCESS", "FAILED", "RUNNING", "PENDING"])
        total_jobs = len(job_statuses)

        if success_count == total_jobs:
            run_status = "SUCCESS"
            status_icon = "✅"
            status_color = Colors.GREEN
            successful_runs += 1
        elif failed_count > 0:
            run_status = "FAILED"
            status_icon = "❌"
            status_color = Colors.RED
            failed_runs += 1
        elif running_count > 0:
            run_status = "RUNNING"
            status_icon = "🏃"
            status_color = Colors.YELLOW
            running_runs += 1
        elif pending_count > 0:
            run_status = "PENDING"
            status_icon = "⏳"
            status_color = Colors.YELLOW
            pending_runs += 1
        else:
            run_status = "UNKNOWN"
            status_icon = "⚠️ "
            status_color = Colors.YELLOW

        # Print run summary
        time_str = timestamp.strftime("%m-%d %H:%M") if timestamp != datetime.min else "unknown"
        print(f"{status_color}{status_icon} {run_name}{Colors.END} "
              f"{Colors.CYAN}[{time_str}]{Colors.END} "
              f"({total_jobs} jobs) "
              f"{Colors.GREEN}{success_count}✓{Colors.END} "
              f"{Colors.RED}{failed_count}✗{Colors.END} "
              f"{Colors.YELLOW}{running_count}🏃{Colors.END} "
              f"{Colors.YELLOW}{pending_count}⏳{Colors.END} "
              f"{Colors.YELLOW}{unknown_count}?{Colors.END}")

        # Show detailed job info if requested
        if detailed and (failed_count > 0 or running_count > 0 or unknown_count > 0):
            for job_name, status, message, log_out_path, log_err_path in job_statuses:
                if status != "SUCCESS":
                    color = Colors.RED if status == "FAILED" else Colors.YELLOW
                    if status == "FAILED" and log_err_path:
                        print(f"    {color}└─ {job_name}: {message} - {Colors.BLUE}{log_err_path}{Colors.END}")
                    else:
                        print(f"    {color}└─ {job_name}: {message}{Colors.END}")
        if not detailed and failed:
            for job_name, status, message, log_out_path, log_err_path in job_statuses:
                if status == "FAILED":
                    color = Colors.RED
                    if log_err_path:
                        print(f"    {color}└─ {job_name}: {message} - {Colors.BLUE}{log_err_path}{Colors.END}")
                    else:
                        print(f"    {color}└─ {job_name}: {message}{Colors.END}")

    # Print summary
    print(f"\n{Colors.PURPLE}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}📊 Summary:{Colors.END}")
    print(f"  Total runs: {total_runs}")
    print(f"  {Colors.GREEN}✅ Successful: {successful_runs}{Colors.END}")
    print(f"  {Colors.RED}❌ Failed: {failed_runs}{Colors.END}")
    print(f"  {Colors.YELLOW}🏃 Running: {running_runs}{Colors.END}")
    print(f"  {Colors.YELLOW}⚠️  Partial/Other: {total_runs - successful_runs - failed_runs - running_runs}{Colors.END}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="SLURM Job Status Visualizer")
    parser.add_argument("-n", "--num-runs", type=int, default=20,
                       help="Number of recent runs to show (default: 20)")
    parser.add_argument("-d", "--detailed", action="store_true",
                       help="Show detailed information for failed/unknown jobs")
    parser.add_argument("-f", "--failed", action="store_true",
                       help="Show only failed jobs")

    args = parser.parse_args()

    try:
        get_multirun_status(max_runs=args.num_runs, detailed=args.detailed, failed=args.failed)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Interrupted by user{Colors.END}")
    except Exception as e:
        print(f"{Colors.RED}Error: {str(e)}{Colors.END}")

if __name__ == "__main__":
    main()