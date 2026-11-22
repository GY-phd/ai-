# Run the CI checks on every supported interpreter, on this machine.
#
# Usage:
#   .\tools\run_matrix.ps1
#   .\tools\run_matrix.ps1 -WithTests
#   .\tools\run_matrix.ps1 -Env .venv-py310
#
# This is the Windows front end to tools/run_ci_matrix.py.  It sets up
# the environments the first time (they are created with ``uv``, and
# ignored by git), then runs the same checks the GitHub workflow runs
# against each interpreter in turn.
#
# The hosted workflow is still the authority on whether a commit is
# acceptable: it covers Linux as well as Windows, and it covers the
# runner image.  What this covers is the half a laptop can reproduce -
# the interpreter - which is the half that catches a dependency that
# does not build on the oldest supported Python.

[CmdletBinding()]
param(
    [switch]$WithTests,
    [string[]]$Env,
    [switch]$Setup
)

$ErrorActionPreference = "Stop"
try { [Console]::OutputEncoding = [Text.Encoding]::UTF8 } catch { }

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Push-Location $root
try {
    $primary = Join-Path $root ".venv\Scripts\python.exe"

    if (-not (Test-Path $primary)) {
        throw "the primary environment .venv is missing; create it with: uv venv --python 3.12 .venv"
    }

    # The matrix environments.  Each is named after the interpreter it
    # holds, and each is created the same way - with uv - so that they
    # are reproducible rather than a set of one-off virtualenvs.
    $targets = @(
        @{ Dir = ".venv-py310"; Version = "3.10" },
        @{ Dir = ".venv";      Version = "3.12" },
        @{ Dir = ".venv-py313"; Version = "3.13" },
        @{ Dir = ".venv-py314"; Version = "3.14" }
    )

    if ($Setup -or $Env) {
        if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
            throw "uv is not on the PATH; install it from https://docs.astral.sh/uv/"
        }
        $env:UV_LINK_MODE = "copy"
        foreach ($t in $targets) {
            $wanted = if ($Env) { $Env -contains $t.Dir } else { $true }
            if (-not $wanted) { continue }
            $py = Join-Path $root ($t.Dir + "\Scripts\python.exe")
            if (-not (Test-Path $py)) {
                Write-Host "creating $($t.Dir) (Python $($t.Version))" -ForegroundColor Cyan
                uv venv --python $t.Version $t.Dir | Out-Null
            }
            Write-Host "installing dependencies into $($t.Dir)" -ForegroundColor Cyan
            $installArgs = @(
                "pip", "install", "--python", $py,
                "-r", "requirements.txt",
                "-r", "requirements-dev.txt",
                "./contest_harness_sdk"
            )
            uv @installArgs | Out-Null
        }
    }

    $argv = @("tools/run_ci_matrix.py")
    if ($WithTests) { $argv += "--with-tests" }
    foreach ($e in $Env) { $argv += @("--env", $e) }

    & $primary @argv
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
