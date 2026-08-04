<# Build script for Windows (PowerShell) #>

param(
    [string]$BuildDir = "build",
    [string]$BuildType = "Release",
    [string]$InstallPrefix = "",
    [switch]$Tests = $false,
    [switch]$Static = $false,
    [switch]$NoValidation = $false,
    [string]$Generator = "Visual Studio 17 2022"
)

$cmakeArgs = @(
    "-DCMAKE_BUILD_TYPE=$BuildType",
    "-DVVM_BUILD_SHARED=" + (!$Static).ToString().ToUpper(),
    "-DVVM_BUILD_TESTS=" + $Tests.ToString().ToUpper(),
    "-DVVM_ENABLE_VALIDATION=" + (!$NoValidation).ToString().ToUpper()
)

if ($InstallPrefix) {
    $cmakeArgs += "-DCMAKE_INSTALL_PREFIX=$InstallPrefix"
}

Write-Host "Configuring Vulkan-Automaton-VM..."
cmake -B $BuildDir -G $Generator $cmakeArgs .

Write-Host "Building..."
cmake --build $BuildDir --config $BuildType

if ($Tests) {
    Write-Host "Running tests..."
    ctest --test-dir $BuildDir --output-on-failure -C $BuildType
}

Write-Host "Build complete. Install with: cmake --install $BuildDir --config $BuildType"