@echo off
setlocal

REM Optional GPU compute architecture argument (default: sm_89 for RTX 40-series)
REM Examples:
REM   build.bat sm_75   (RTX 20-series / Turing)
REM   build.bat sm_86   (RTX 30-series / Ampere)
REM   build.bat sm_89   (RTX 40-series / Ada Lovelace)
REM   build.bat sm_120  (RTX 50-series / Blackwell)
set "CUDA_ARCH=%~1"
if "%CUDA_ARCH%"=="" set "CUDA_ARCH=sm_89"

REM Auto-locate Visual Studio x64 environment (VS 2022 or VS 2019, BuildTools/Community/Pro/Enterprise)
set "VCVARS="
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" goto :vs_fallback
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
    if exist "%%i\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=%%i\VC\Auxiliary\Build\vcvars64.bat"
)
if defined VCVARS goto :vs_found

:vs_fallback
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if defined VCVARS goto :vs_found
if exist "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if defined VCVARS goto :vs_found
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if defined VCVARS goto :vs_found
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
if defined VCVARS goto :vs_found

echo [BUILD ERROR] Could not find Visual Studio C++ x64 Build Tools (vcvars64.bat).
echo Install Visual Studio 2019 or 2022 with the "Desktop development with C++" workload.
exit /b 1

:vs_found
call "%VCVARS%" >nul
if errorlevel 1 (
    echo [BUILD ERROR] Failed to load Visual Studio x64 environment from "%VCVARS%"
    exit /b 1
)

REM Auto-locate CUDA Toolkit (prefers CUDA_PATH env var, falls back to CUDA v12.8)
if "%CUDA_PATH%"=="" if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8" set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
if "%CUDA_PATH%"=="" (
    echo [BUILD ERROR] CUDA_PATH is not set. Install NVIDIA CUDA Toolkit 12.x and reopen your terminal.
    exit /b 1
)

if not exist build mkdir build

echo [1/2] Compiling CUDA kernels (%CUDA_ARCH%, CUDA=%CUDA_PATH%)...
nvcc -O3 -arch=%CUDA_ARCH% --use_fast_math -std=c++17 -Xcompiler "/O2 /MD /EHsc" -c src/cuda_kernels.cu -o build/cuda_kernels.obj
if errorlevel 1 exit /b 1

echo [2/2] Compiling C++ AVX2 engine and linking inkengine.exe...
cl /nologo /O2 /arch:AVX2 /fp:fast /std:c++17 /EHsc /MD /DNOMINMAX ^
   /I"%CUDA_PATH%\include" ^
   src/gguf.cpp src/cpu_kernels.cpp src/expert_cache.cpp src/moe_predictor.cpp ^
   src/paged_kv.cpp src/tokenizer.cpp src/vision.cpp src/model.cpp src/server.cpp src/main.cpp ^
   build/cuda_kernels.obj ^
   /Fe:build/inkengine.exe /Fo:build/ ^
   /link /LIBPATH:"%CUDA_PATH%\lib\x64" cudart.lib ws2_32.lib windowscodecs.lib ole32.lib
if errorlevel 1 exit /b 1

echo [SUCCESS] Built build\inkengine.exe (%CUDA_ARCH%)
