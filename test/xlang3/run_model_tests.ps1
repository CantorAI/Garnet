param(
    [Parameter(Mandatory=$true)][string]$RuntimePath,
    [Parameter(Mandatory=$true)][ValidateSet('tensorrt','openvino')][string]$Backend,
    [Parameter(Mandatory=$true)][string]$WorkRoot,
    [string]$TensorRTDirectory,
    [string]$CudaDirectory
)
$ErrorActionPreference = 'Stop'
if ($TensorRTDirectory) { $env:PATH = "$TensorRTDirectory;$env:PATH" }
if ($CudaDirectory) { $env:PATH = "$CudaDirectory;$env:PATH" }
$work = Join-Path $WorkRoot ([Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $work -Force | Out-Null
Set-Location (Split-Path -Parent $RuntimePath)
$models = Join-Path $PSScriptRoot 'models'
$cache = Join-Path $work 'engines'
& $RuntimePath (Join-Path $models 'run_models.py') $Backend $cache build
if ($LASTEXITCODE -ne 0) { throw "Model build/execution failed ($LASTEXITCODE); artifacts: $work" }
& $RuntimePath (Join-Path $models 'run_models.py') $Backend $cache reload
if ($LASTEXITCODE -ne 0) { throw "Model cache reload failed ($LASTEXITCODE); artifacts: $work" }
& $RuntimePath (Join-Path $models 'run_contracts.py') $Backend (Join-Path $work 'contracts')
if ($LASTEXITCODE -ne 0) { throw "Model contracts failed ($LASTEXITCODE); artifacts: $work" }
& $RuntimePath (Join-Path $models 'run_source_isolation.py') $Backend (Join-Path $work 'isolation')
if ($LASTEXITCODE -ne 0) { throw "Model isolation failed ($LASTEXITCODE); artifacts: $work" }
if ($Backend -eq 'tensorrt') {
    & $RuntimePath (Join-Path $models 'run_reuse.py') (Join-Path $work 'reuse')
    if ($LASTEXITCODE -ne 0) { throw "TensorRT reusable output test failed ($LASTEXITCODE); artifacts: $work" }
    & $RuntimePath (Join-Path $models 'run_concurrent.py') (Join-Path $work 'concurrent')
    if ($LASTEXITCODE -ne 0) { throw "TensorRT concurrent model test failed ($LASTEXITCODE); artifacts: $work" }
}
Write-Output "Garnet $Backend integration passed; artifacts: $work"
