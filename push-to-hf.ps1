# Run openenv with UTF-8 so Rich/emoji output does not crash on Windows (charmap codec).
# Usage:
#   .\push-to-hf.ps1 push --repo-id YOUR_USER/YOUR_SPACE .
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $OpenEnvArgs
)

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $OutputEncoding = [Console]::OutputEncoding
} catch {}

if (-not $OpenEnvArgs -or $OpenEnvArgs.Count -eq 0) {
    Write-Host "Example: .\push-to-hf.ps1 push --repo-id nigmu/openenv-runtime ." -ForegroundColor Yellow
    exit 1
}

& openenv @OpenEnvArgs
