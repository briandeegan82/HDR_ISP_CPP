# Create a temporary directory for downloads
$tempDir = "C:\temp"
if (-not (Test-Path $tempDir)) {
    New-Item -ItemType Directory -Path $tempDir
}

# CMake download URL (latest stable version)
$cmakeUrl = "https://github.com/Kitware/CMake/releases/download/v3.28.1/cmake-3.28.1-windows-x86_64.msi"
$cmakeInstaller = "$tempDir\cmake-installer.msi"

# Download CMake installer
Write-Host "Downloading CMake installer..."
Invoke-WebRequest -Uri $cmakeUrl -OutFile $cmakeInstaller

# Install CMake
Write-Host "Installing CMake..."
Start-Process msiexec.exe -ArgumentList "/i `"$cmakeInstaller`" /quiet /norestart" -Wait

# Add CMake to PATH if not already there
$cmakePath = "C:\Program Files\CMake\bin"
$currentPath = [Environment]::GetEnvironmentVariable("Path", "Machine")
if (-not $currentPath.Contains($cmakePath)) {
    [Environment]::SetEnvironmentVariable("Path", "$currentPath;$cmakePath", "Machine")
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
}

# Clean up
Remove-Item $cmakeInstaller

Write-Host "CMake installation complete!"
Write-Host "Please restart your PowerShell session for the PATH changes to take effect." 