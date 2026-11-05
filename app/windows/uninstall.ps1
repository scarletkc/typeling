#Requires -RunAsAdministrator
# Removes the Typeling text service. Run from an elevated PowerShell.

$ErrorActionPreference = 'Stop'

$directory = Join-Path $env:ProgramFiles 'Typeling'
$target = Join-Path $directory 'typeling_tip.dll'

if (Test-Path $target) {
    regsvr32.exe /s /u $target | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "regsvr32 /u failed with exit code $LASTEXITCODE; nothing was deleted."
    }
}

# Applications that still have the DLL open block deleting it, which does no harm: it is no longer registered.
Remove-Item -Path $directory -Recurse -Force -ErrorAction SilentlyContinue
if (Test-Path $directory) {
    Write-Host "Unregistered. $directory is still in use: delete it after signing out and back in."
} else {
    Write-Host 'Uninstalled.'
}
