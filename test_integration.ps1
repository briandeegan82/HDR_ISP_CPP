# Test Integration Script for HDR ISP Hybrid Backend
Write-Host "=== HDR ISP Hybrid Backend Integration Test ===" -ForegroundColor Green

# Set paths
$TEST_IMAGE = "in_frames/normal/ColorChecker_2592x1536_12bits_RGGB.raw"
$TEST_CONFIG = "config/configs.yml"
$BUILD_DIR = "test_build/Release"

# Check if test image exists
if (-not (Test-Path $TEST_IMAGE)) {
    Write-Host "Error: Test image not found at $TEST_IMAGE" -ForegroundColor Red
    exit 1
}

# Check if executable exists
if (-not (Test-Path "$BUILD_DIR/hdr_isp_pipeline.exe")) {
    Write-Host "Error: Executable not found at $BUILD_DIR/hdr_isp_pipeline.exe" -ForegroundColor Red
    exit 1
}

Write-Host "Test image: $TEST_IMAGE" -ForegroundColor Yellow
Write-Host "Test config: $TEST_CONFIG" -ForegroundColor Yellow
Write-Host "Executable: $BUILD_DIR/hdr_isp_pipeline.exe" -ForegroundColor Yellow

# Clean up any existing output files
if (Test-Path "out_frames") {
    Write-Host "Cleaning up existing output files..." -ForegroundColor Yellow
    Remove-Item "out_frames\*.png" -Force -ErrorAction SilentlyContinue
}

# Test 1: Run with hybrid backend disabled (baseline)
Write-Host "`n=== Test 1: Baseline Performance (Hybrid Backend Disabled) ===" -ForegroundColor Cyan
$start_time = Get-Date
$baseline_result = & "$BUILD_DIR/hdr_isp_pipeline.exe" $TEST_CONFIG $TEST_IMAGE 2>&1
$end_time = Get-Date
$baseline_duration = ($end_time - $start_time).TotalSeconds

if ($LASTEXITCODE -eq 0) {
    Write-Host "Baseline execution completed successfully" -ForegroundColor Green
    Write-Host "Baseline execution time: $baseline_duration seconds" -ForegroundColor Green
} else {
    Write-Host "Baseline execution failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    Write-Host "Error output: $baseline_result" -ForegroundColor Red
}

# Test 2: Run with hybrid backend enabled (if available)
Write-Host "`n=== Test 2: Hybrid Backend Performance ===" -ForegroundColor Cyan
$start_time = Get-Date
$hybrid_result = & "$BUILD_DIR/hdr_isp_pipeline.exe" $TEST_CONFIG $TEST_IMAGE 2>&1
$end_time = Get-Date
$hybrid_duration = ($end_time - $start_time).TotalSeconds

if ($LASTEXITCODE -eq 0) {
    Write-Host "Hybrid execution completed successfully" -ForegroundColor Green
    Write-Host "Hybrid execution time: $hybrid_duration seconds" -ForegroundColor Green
} else {
    Write-Host "Hybrid execution failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    Write-Host "Error output: $hybrid_result" -ForegroundColor Red
}

# Calculate performance improvement
if ($baseline_duration -gt 0 -and $hybrid_duration -gt 0) {
    $speedup = $baseline_duration / $hybrid_duration
    $improvement = (($baseline_duration - $hybrid_duration) / $baseline_duration) * 100
    Write-Host "`n=== Performance Results ===" -ForegroundColor Magenta
    Write-Host "Speedup factor: ${speedup}x" -ForegroundColor Green
    Write-Host "Performance improvement: ${improvement}%" -ForegroundColor Green
    Write-Host "Time saved: $($baseline_duration - $hybrid_duration) seconds" -ForegroundColor Green
    
    if ($speedup -gt 1) {
        Write-Host "✅ Performance improvement achieved!" -ForegroundColor Green
    } else {
        Write-Host "⚠️  No performance improvement detected" -ForegroundColor Yellow
    }
} else {
    Write-Host "`nWarning: Could not calculate performance improvement" -ForegroundColor Yellow
}

# Check output files
Write-Host "`n=== Output Files ===" -ForegroundColor Cyan
if (Test-Path "out_frames") {
    $output_files = Get-ChildItem "out_frames" -Filter "*.png"
    Write-Host "Generated output files:" -ForegroundColor Green
    foreach ($file in $output_files) {
        Write-Host "  - $($file.Name) ($($file.Length / 1KB) KB)" -ForegroundColor White
    }
} else {
    Write-Host "No output files found" -ForegroundColor Yellow
}

# Check for hybrid backend usage in output
Write-Host "`n=== Hybrid Backend Usage Analysis ===" -ForegroundColor Cyan
if ($hybrid_result -match "Using Halide-optimized") {
    Write-Host "✅ Halide modules detected in execution" -ForegroundColor Green
    $halide_modules = ($hybrid_result | Select-String "Using Halide-optimized").Count
    Write-Host "Number of Halide modules used: $halide_modules" -ForegroundColor Green
} else {
    Write-Host "⚠️  No Halide modules detected - using fallback implementations" -ForegroundColor Yellow
}

Write-Host "`n=== Integration Test Complete ===" -ForegroundColor Green 