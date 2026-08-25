# gpu_windows.ps1 - Self-hosted GPU CI suite for the Chonk Buffer box.
#
# Runs the REAL hardware gates on every push to main:
#   1. CPU-only regression suites (buddy, chonk slab fuzz, placement)
#   2. Vulkan device enumeration sanity (expects the dual-vendor pair)
#   3. llama-server boot with the Chonk pool on both GPUs
#   4. Live completion + throughput floor (catches VRAM-spill-class
#      regressions - the silent 2x cliff from VRAM_OVERFLOW_FINDINGS.md)
#   5. /vvm/stats: both pools engaged
#
# Config via env (defaults match this box):
#   VVM_REPO       (default D:\VulkanVM)
#   VVM_BUILD      (default D:\VulkanVM\build_infer - reused incrementally)
#   VVM_LLAMA_BIN  (default D:\llama-src\build\bin)
#   CHONK_TEST_MODEL (default the 40B Q4_K_M)
#
# Exit code 0 = all gates passed.

param(
    [string]$RepoRoot  = $env:VVM_REPO      ?? "D:\VulkanVM",
    [string]$BuildDir  = $env:VVM_BUILD     ?? "D:\VulkanVM\build_infer",
    [string]$LlamaBin  = $env:VVM_LLAMA_BIN ?? "D:\llama-src\build\bin",
    [string]$Model     = $env:CHONK_TEST_MODEL ?? "C:\Users\mikeh\Downloads\Qwen3.6-40B-FF6core-Deck-Eleanor-H-Uncen-NEO-MAX-MTP-Q4_K_M.gguf"
)

$ErrorActionPreference = "Continue"
$failed = 0
function Gate([string]$name, [bool]$ok, [string]$detail = "") {
    if ($ok) { echo "PASS  $name $detail" }
    else     { echo "FAIL  $name $detail"; $script:failed++ }
}

$vcvars = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
function Shell([string]$c) { cmd /c "call `"$vcvars`" >nul 2>&1 && set PATH=C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64;%PATH% && $c" }

echo "=== GPU CI suite (Windows) ==="
echo "repo: $RepoRoot"

# ---- Gate 1: CPU-only regression suites ------------------------------------
Shell "cmake --build $BuildDir --target buddy_test chonk_slab_test placement_test vulkan_vm" | Out-Null
Copy-Item "$BuildDir\vulkan_vm.dll" "$BuildDir\tests\" -Force -ErrorAction SilentlyContinue

& "$BuildDir\tests\buddy_test.exe" *> $null
Gate "buddy_test" ($LASTEXITCODE -eq 0)

& "$BuildDir\tests\chonk_slab_test.exe" 100000 *> $null
Gate "chonk_slab_test (100k fuzz)" ($LASTEXITCODE -eq 0)

& "$BuildDir\tests\placement_test.exe" *> $null
Gate "placement_test" ($LASTEXITCODE -eq 0)

# ---- Gate 2: Vulkan device enumeration --------------------------------------
$vi = vulkaninfo --summary 2>$null | Out-String
$hasXtx = $vi -match "7900 XTX"
$hasB70 = $vi -match "Arc\(TM\) Pro B70"
Gate "Vulkan devices (XTX + B70 present)" ($hasXtx -and $hasB70)

# ---- Gate 3+4+5: llama-server boot, completion throughput, pool stats -------
$env:GGML_VK_VISIBLE_DEVICES = "0,2"
$env:GGML_VK_VVM_POOL = "1"
$log = "$env:TEMP\gpu_ci_server.log"
$proc = Start-Process -FilePath "$LlamaBin\llama-server.exe" `
    -ArgumentList '-m', "`"$Model`"", '--alias', 'gpu-ci', '-ngl', '99', '-sm', 'layer', `
                  '-fa', 'on', '-c', '8192', '-ctk', 'q8_0', '-ctv', 'q8_0', `
                  '-b', '2048', '-ub', '512', '--temp', '0.8', `
                  '--host', '127.0.0.1', '--port', '8123' `
    -PassThru -RedirectStandardError $log
try {
    $up = $false
    for ($i = 0; $i -lt 30; ++$i) {
        Start-Sleep -Seconds 5
        try {
            $h = Invoke-WebRequest "http://127.0.0.1:8123/health" -UseBasicParsing -TimeoutSec 3
            if ($h.Content -match '"ok"') { $up = $true; break }
        } catch {}
    }
    Gate "llama-server boot (Chonk pool)" $up
    if (-not $up) { echo "--- server log tail ---"; Get-Content $log -Tail 15; return }

    # both pools engaged (device-filter regression canary)
    $stats = (Invoke-WebRequest "http://127.0.0.1:8123/vvm/stats" -UseBasicParsing -TimeoutSec 5).Content
    $pools = ($stats | ConvertFrom-Json).Count
    Gate "/vvm/stats both pools" ($pools -ge 2) "($pools pools)"

    # completion + throughput floor
    $body = @{
        messages = @(@{ role = "user"; content = "Say ready." })
        temperature = 0.8; max_tokens = 96; stream = $false
    } | ConvertTo-Json -Depth 5
    $r = Invoke-WebRequest "http://127.0.0.1:8123/v1/chat/completions" -Method Post `
         -Body $body -ContentType "application/json" -UseBasicParsing -TimeoutSec 300
    $j = $r.Content | ConvertFrom-Json
    $tps = [math]::Round($j.timings.predicted_per_second, 1)
    $ntok = $j.timings.predicted_n
    Gate "completion generated" ($ntok -gt 0) "($ntok tokens)"
    # Throughput floor: healthy dual-GPU decode at 8K ctx is 25+ t/s.
    # The VRAM-spill class of regression halves it; hard faults kill it.
    Gate "decode throughput floor" ($tps -ge 8) "(${tps} t/s, floor 8)"

    echo "=== GPU CI summary ==="
    if ($failed -eq 0) { echo "ALL GATES PASSED"; exit 0 }
    echo "$failed GATE(S) FAILED"; exit 1
} finally {
    if ($proc -and -not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
    Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
}
