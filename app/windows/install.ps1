#Requires -RunAsAdministrator
# Installs the Typeling text service for all users of this machine. Run from an elevated PowerShell after building:
#     cargo build --release --manifest-path app\windows\tip\Cargo.toml
# The DLL goes to Program Files because it is loaded into every application, including elevated ones:
# it must not live anywhere a normal user can replace it. uninstall.ps1 reverses this.

$ErrorActionPreference = 'Stop'

$source = Join-Path $PSScriptRoot 'tip\target\release\typeling_tip.dll'
if (-not (Test-Path $source)) {
    throw "$source does not exist. Build it first: cargo build --release --manifest-path app\windows\tip\Cargo.toml"
}

$directory = Join-Path $env:ProgramFiles 'Typeling'
$target = Join-Path $directory 'typeling_tip.dll'
New-Item -ItemType Directory -Force -Path $directory | Out-Null

# Running applications keep the previous DLL open: it cannot be overwritten, but it can be renamed and deleted later.
Get-ChildItem -Path $directory -Filter '*.old' | Remove-Item -Force -ErrorAction SilentlyContinue
if (Test-Path $target) {
    Rename-Item -Path $target -NewName "typeling_tip.$(Get-Date -Format yyyyMMddHHmmss).old"
}
Copy-Item -Path $source -Destination $target

regsvr32.exe /s $target | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "regsvr32 failed with exit code $LASTEXITCODE."
}

Write-Host "Installed $target"
Write-Host 'Next: start the engine (python app\windows\engine.py), then press Win+Space and choose Typeling.'
Write-Host 'Applications that are already open pick it up after a restart.'
