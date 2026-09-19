<#
.SYNOPSIS
Sync this working tree to the Ubuntu VM and run the authoritative suites there.

.DESCRIPTION
Windows is the editing surface; the Ubuntu VM is the authoritative test environment, so a
green Windows run is not evidence on its own. This script packages exactly what the suites
import, copies it into an isolated /tmp directory on the VM, runs the unittest suite there,
and optionally runs the guarded real-Docker smoke test. Both the local bundle and the remote
copies are removed afterwards, so a failed run does not leave state behind.

The bundle must contain scripts, datasets and fixtures as well as src and tests: tests import
`scripts.*` and read the fixtures, and their absence shows up as import errors rather than as
test failures.

.PARAMETER DockerIntegration
Also run tests.test_docker_integration against the cached SWE-Gym base image.

.PARAMETER KeepBundle
Keep the local tar after the run for debugging; the remote copies are still removed.

.PARAMETER RemoteCommand
Run this shell command inside the extracted tree instead of the unittest suite, so a
rollout or a diagnostic uses this working copy rather than the VM's older checkout.
It is executed by `sh -c` from the extracted repository root, so keep it free of
single quotes and use `timeout` yourself for a step that can hang.

.EXAMPLE
pwsh -File scripts/sync_and_validate_vm.ps1

.EXAMPLE
pwsh -File scripts/sync_and_validate_vm.ps1 -DockerIntegration

.EXAMPLE
pwsh -File scripts/sync_and_validate_vm.ps1 -TimeoutSeconds 7200 `
  -RemoteCommand "PYTHONPATH=src python3 -m coding_agent_rl_lab.swe_gym_rollout --model '<id>' --context-window-tokens 32768 --task-set held-out"
#>
[CmdletBinding()]
param(
    [string]$VmUser = "wesz",
    [string]$VmHost = "192.168.137.130",
    [string]$IdentityFile = (Join-Path $HOME ".ssh\id_ed25519_codex_vm"),
    [string]$Python = "python3",
    [int]$TimeoutSeconds = 1800,
    [string]$DockerBaseImage = "xingyaoww/sweb.eval.x86_64.getmoto_s_moto-7365:latest",
    [switch]$DockerIntegration,
    [switch]$KeepBundle,
    [string]$RemoteCommand = ""
)

$ErrorActionPreference = "Stop"

foreach ($tool in "tar", "ssh", "scp") {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "$tool is required but was not found on PATH"
    }
}
if (-not (Test-Path -LiteralPath $IdentityFile)) {
    throw "SSH identity file not found: $IdentityFile"
}

$repository = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$bundlePaths = @("src", "tests", "scripts", "datasets", "fixtures", "pyproject.toml")
foreach ($relative in $bundlePaths) {
    if (-not (Test-Path -LiteralPath (Join-Path $repository $relative))) {
        throw "required bundle path is missing: $relative"
    }
}

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$bundleName = "coding-agent-rl-lab-$stamp.tar"
$localBundle = Join-Path ([System.IO.Path]::GetTempPath()) $bundleName
$remoteBundle = "/tmp/$bundleName"
$remoteRoot = "/tmp/coding-agent-rl-lab-$stamp"
$target = "$VmUser@$VmHost"
$sshOptions = @(
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=15",
    "-o", "StrictHostKeyChecking=accept-new",
    "-i", $IdentityFile
)

function Invoke-Remote {
    param([Parameter(Mandatory)][string]$Command)

    & ssh @sshOptions $target $Command
    if ($LASTEXITCODE -ne 0) {
        throw "remote command failed with exit code $LASTEXITCODE"
    }
}

try {
    Write-Host "[check] $target"
    Invoke-Remote "$Python -V && (docker --version || echo 'docker: not available')"

    Write-Host "[pack] $($bundlePaths -join ', ')"
    & tar -cf $localBundle --exclude=__pycache__ --exclude=.pytest_cache @bundlePaths
    if ($LASTEXITCODE -ne 0) {
        throw "tar failed with exit code $LASTEXITCODE"
    }
    Write-Host ("      {0:N0} bytes" -f (Get-Item -LiteralPath $localBundle).Length)

    Write-Host "[upload] $remoteBundle"
    & scp @sshOptions $localBundle "${target}:$remoteBundle"
    if ($LASTEXITCODE -ne 0) {
        throw "scp failed with exit code $LASTEXITCODE"
    }

    Write-Host "[unpack] $remoteRoot"
    Invoke-Remote "rm -rf $remoteRoot && mkdir -p $remoteRoot && tar -xf $remoteBundle -C $remoteRoot"

    if ($RemoteCommand) {
        $escaped = $RemoteCommand -replace "'", "'\''"
        Write-Host "[run] $RemoteCommand"
        Invoke-Remote "cd $remoteRoot && timeout $TimeoutSeconds sh -c '$escaped'"
    } else {
        Write-Host "[run] unittest suite"
        Invoke-Remote "cd $remoteRoot && PYTHONPATH=src timeout $TimeoutSeconds $Python -m unittest discover -s tests"
    }

    if ($DockerIntegration -and -not $RemoteCommand) {
        Write-Host "[run] real Docker integration smoke"
        Invoke-Remote (
            "cd $remoteRoot && RUN_DOCKER_INTEGRATION=1 " +
            "DOCKER_INTEGRATION_BASE_IMAGE=$DockerBaseImage PYTHONPATH=src " +
            "timeout $TimeoutSeconds $Python -m unittest tests.test_docker_integration -v"
        )
    } elseif ($DockerIntegration) {
        Write-Host "[run] Docker integration skipped while -RemoteCommand is set"
    } else {
        Write-Host "[run] Docker integration skipped; pass -DockerIntegration to run it"
    }
}
finally {
    Write-Host "[cleanup] $remoteRoot"
    & ssh @sshOptions $target "rm -rf $remoteRoot $remoteBundle" 2>$null | Out-Null
    if ($KeepBundle) {
        Write-Host "kept local bundle: $localBundle"
    } elseif (Test-Path -LiteralPath $localBundle) {
        Remove-Item -LiteralPath $localBundle -Force
    }
}
