param(
    [Parameter(Mandatory=$true)][string]$Executable,
    [Parameter(Mandatory=$true)][string]$WorkRoot
)
$ErrorActionPreference = 'Stop'
$process = New-Object System.Diagnostics.Process
$started = $false
$process.StartInfo.FileName = $Executable
$process.StartInfo.WorkingDirectory = Split-Path -Parent $Executable
$process.StartInfo.UseShellExecute = $false
$process.StartInfo.CreateNoWindow = $true
$process.StartInfo.RedirectStandardInput = $true
$process.StartInfo.RedirectStandardOutput = $true
$process.StartInfo.RedirectStandardError = $true
$process.StartInfo.EnvironmentVariables['GARNET_DATA_ROOT'] = Join-Path $WorkRoot ([Guid]::NewGuid().ToString('N'))
$process.StartInfo.EnvironmentVariables['GARNET_AUTO_INSTALL'] = '0'
function Request([string]$json) {
    $process.StandardInput.WriteLine($json)
    $process.StandardInput.Flush()
    $read = $process.StandardOutput.ReadLineAsync()
    if (-not $read.Wait(10000)) { throw 'Serving response timed out' }
    if ($null -eq $read.Result) { throw 'Serving exited before responding' }
    Write-Output ($read.Result | ConvertFrom-Json)
}
try {
    if (-not $process.Start()) { throw 'Cannot start serving process' }
    $started = $true
    $errors = $process.StandardError.ReadToEndAsync()
    $health = Request '{"path":"/v1/health"}'
    if ($health.status -ne 'ok' -or -not $health.ready) { throw 'Health check failed' }
    Start-Sleep -Milliseconds 250
    if ($process.HasExited) { throw 'Serving exited while input remained open' }
    $models = Request '{"path":"/v1/models"}'
    if ($models.schema_version -ne 1 -or $models.models.Count -ne 0) { throw 'Installed model listing failed' }
    $missing = Request '{"path":"/unknown"}'
    if ($missing.error_code -ne 'route_not_found') { throw 'Unknown route was accepted' }
    $health = Request '{"path":"/v1/health"}'
    if ($health.status -ne 'ok') { throw 'Process did not recover after invalid route' }
    $process.StandardInput.Close()
    if (-not $process.WaitForExit(10000)) { throw 'Serving did not stop on EOF' }
    if ($process.ExitCode -ne 0) { throw "Serving failed: $($errors.Result)" }
    Write-Output 'garnet-serving-process-lifecycle-passed'
} finally {
    if ($started -and -not $process.HasExited) {
        $process.Kill()
        $process.WaitForExit()
    }
    $process.Dispose()
}
