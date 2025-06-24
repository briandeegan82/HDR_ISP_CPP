@echo off
cd /d C:\workspace\HDR_ISP_CPP\build\Release
hdr_isp_pipeline.exe "C:\workspace\HDR_ISP_CPP\in_frames\normal\ColorChecker_2592x1536_12bits_RGGB.raw" "C:\workspace\HDR_ISP_CPP\config\svs_cam.yml"
pause 