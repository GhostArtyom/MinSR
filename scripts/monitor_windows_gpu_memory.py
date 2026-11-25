"""Record Windows GPU memory counters as JSON lines, including from WSL."""

import argparse
import shutil
import subprocess


_QUERY = r"""
& {
    param([int]$IntervalSeconds, [int]$Samples)
    $ErrorActionPreference = 'Stop'
    for ($i = 0; $Samples -eq 0 -or $i -lt $Samples; $i++) {
        $started = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
        $adapters = @(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory |
            ForEach-Object {
                @{
                    adapter = $_.Name
                    dedicated_bytes = [uint64]$_.DedicatedUsage
                    shared_bytes = [uint64]$_.SharedUsage
                }
            })

        $workerIds = @((Get-Process -Name vmwp -ErrorAction SilentlyContinue).Id)
        $vmWorkers = @(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory |
            ForEach-Object {
                if ($_.Name -match '^pid_(\d+)_' -and [int]$Matches[1] -in $workerIds) {
                    @{
                        instance = $_.Name
                        pid = [int]$Matches[1]
                        dedicated_bytes = [uint64]$_.DedicatedUsage
                        shared_bytes = [uint64]$_.SharedUsage
                    }
                }
            })

        @{
            started_unix_ms = $started
            finished_unix_ms = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
            adapters = $adapters
            vm_workers = $vmWorkers
        } | ConvertTo-Json -Depth 4 -Compress

        if ($Samples -eq 0 -or $i + 1 -lt $Samples) {
            Start-Sleep -Seconds $IntervalSeconds
        }
    }
}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval-seconds", type=int, default=5, help="Wait between samples.")
    parser.add_argument("--samples", type=int, default=0, help="Stop after this many samples (0 keeps sampling).")
    args = parser.parse_args()
    if args.interval_seconds < 1 or args.samples < 0:
        parser.error("--interval-seconds must be positive and --samples must be nonnegative")
    powershell = shutil.which("powershell.exe")
    if powershell is None:
        parser.error("Windows PowerShell is required in PATH")
    command = f"{_QUERY.rstrip()} {args.interval_seconds} {args.samples}"
    try:
        return subprocess.run([powershell, "-NoProfile", "-NonInteractive", "-Command", command]).returncode
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
