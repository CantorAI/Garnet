param(
    [Parameter(Mandatory=$true)][int]$NativeProcessId,
    [Parameter(Mandatory=$true)][string]$OutputPath,
    [int]$IntervalMs=1000
)
# Diagnostic run only: CIM polling can perturb timing. Keep raw adapter rows;
# LUID/phys identity must be resolved before claiming single-adapter totals.
$ErrorActionPreference='Stop'
while (Get-Process -Id $NativeProcessId -ErrorAction SilentlyContinue) {
    $sampleTime=[DateTime]::UtcNow.ToString('o')
    $rows=@(Get-CimInstance -Namespace root/cimv2 -ClassName Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory |
        Where-Object Name -Like "pid_${NativeProcessId}_*" |
        Select-Object Name,DedicatedUsage,SharedUsage,TotalCommitted)
    $sample=@{utc=$sampleTime;process_id=$NativeProcessId;rows=$rows;scope='WDDM per-process raw adapter counters; diagnostic sampler'}
    $sample | ConvertTo-Json -Depth 5 -Compress | Add-Content -LiteralPath $OutputPath
    Start-Sleep -Milliseconds $IntervalMs
}
@{utc=[DateTime]::UtcNow.ToString('o');process_id=$NativeProcessId;event='process_exited'} |
    ConvertTo-Json -Compress | Add-Content -LiteralPath $OutputPath
