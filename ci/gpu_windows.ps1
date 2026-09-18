# gpu_windows.ps1 - Self-hosted GPU CI suite for the Chonk Buffer box.
#
# Runs the REAL hardware gates on every push to main:
#   1. CPU-only regression suites (buddy incl. concurrent fuzz, slab, placement)
#   2. Backend seam tests (Level Zero, P2P, shared arena, XN probe, retirement)
#   3. Storage + network suites (e2e self-contained, udp verbs)
#   4. Vulkan device enumeration sanity (expects the dual-vendor pair)
#   5. llama-server boot with the Chonk pool (4B smoke model, seconds not hours)
#   6. Live completion + throughput floor (catches spill-class regressions)
#   7. /vvm/stats: both pools engaged (device-filter regression canary)
#   8. vvm-info smoke (diagnostics bundle runs clean)
#
# Config via env (defaults match this box):
#   VVM_REPO       (default D:\VulkanVM)
#   VVM_BUILD      (default D:\VulkanVM\build_infer - reused incrementally)
#   VVM_LLAMA_BIN  (default D:\llama-src\build-both-win\bin)
#   CHONK_TEST_MODEL (default the 4B Q8 smoke model)
#
# Exit code 0 = all gates passed.

param(
    [string]$RepoRoot  = $env:VVM_REPO      ?? "D:\VulkanVM",
    [string]$BuildDir  = $env:VVM_BUILD     ?? "D:\VulkanVM\build_infer",
    [string]$LlamaBin  = $env:VVM_LLAMA_BIN ?? "D:\llama-src\build-both-win\bin",
    [string]$Model     = $env:CHONK_TEST_MODEL ?? "G:\New folder\models\loras\huihui-qwen3-4b-abliterated-v2-q8_0.gguf"
)

$ErrorActionPreference = "Continue"
$failed = 0
function Gate([string]$name, [bool]$ok, [string]$detail = "") {
    if ($ok) { echo "PASS  $name $detail" }
    else     { echo "FAIL  $name $detail"; $script:failed++ }
}

$vcvars = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
function Shell([string]$c) { cmd /c "call `"$vcvars`" >nul 2>&1 && $c" }

# Test binaries resolve their DLLs from the build tree (vulkan_vm.dll at the
# build root, vulkan_vm_network.dll under src\network) via PATH, not copies.
$env:PATH = "$BuildDir;$BuildDir\src\network;" + $env:PATH

echo "=== GPU CI suite (Windows) ==="
echo "repo: $RepoRoot"

# ---- Gate 1: CPU-only regression suites ------------------------------------
Shell "cmake --build $BuildDir --target buddy_test chonk_slab_test placement_test pool_test minimal_test vulkan_vm" | Out-Null
& "$BuildDir\tests\buddy_test.exe" *> $null
Gate "buddy_test (incl. concurrent fuzz)" ($LASTEXITCODE -eq 0)

& "$BuildDir\tests\chonk_slab_test.exe" 100000 *> $null
Gate "chonk_slab_test (100k fuzz)" ($LASTEXITCODE -eq 0)

& "$BuildDir\tests\placement_test.exe" *> $null
Gate "placement_test" ($LASTEXITCODE -eq 0)

& "$BuildDir\tests\pool_test.exe" *> $null
Gate "pool_test" ($LASTEXITCODE -eq 0)

# ---- Gate 2: backend + cross-vendor seams ----------------------------------
Shell "cmake --build $BuildDir --target l0_backend_test multi_gpu_test shared_arena_test p2p_xn_test retirement_test external_handle_test udp_verb_test storage_e2e_stream_test" | Out-Null

& "$BuildDir\tools\l0_backend_test.exe" *> $null
Gate "l0_backend_test (B70 Level Zero)" ($LASTEXITCODE -eq 0)

& "$BuildDir\tests\multi_gpu_test.exe" *> $null
Gate "multi_gpu_test (P2P)" ($LASTEXITCODE -eq 0)

& "$BuildDir\tests\shared_arena_test.exe" *> $null
Gate "shared_arena_test (host arena)" ($LASTEXITCODE -eq 0)

& "$BuildDir\tests\p2p_xn_test.exe" *> $null
Gate "p2p_xn_test (staged XN)" ($LASTEXITCODE -eq 0)

& "$BuildDir\tests\retirement_test.exe" *> $null
Gate "retirement_test (GPU reclamation)" ($LASTEXITCODE -eq 0)

& "$BuildDir\tests\external_handle_test.exe" *> $null
Gate "external_handle_test" ($LASTEXITCODE -eq 0)

& "$BuildDir\tests\udp_verb_test.exe" *> $null
Gate "udp_verb_test" ($LASTEXITCODE -eq 0)

& "$BuildDir\tests\storage_e2e_stream_test.exe" *> $null
Gate "storage_e2e_stream_test (self-contained)" ($LASTEXITCODE -eq 0)

# ---- Gate 3: Vulkan device enumeration --------------------------------------
$vi = vulkaninfo --summary 2>$null | Out-String
$hasXtx = $vi -match "7900 XTX"
$hasB70 = $vi -match "Arc\(TM\) Pro B70"
Gate "Vulkan devices (XTX + B70 present)" ($hasXtx -and $hasB70)

# ---- Gate 4: vvm-info smoke -------------------------------------------------
& "$BuildDir\tools\vvm_info.exe" *> $null
Gate "vvm_info" ($LASTEXITCODE -eq 0)

# ---- Gates 5+6+7: llama-server boot, completion, pool stats -----------------
# 4B Q8 smoke model: loads in seconds, splits across both dGPUs, exercises
# the pool on the decode path. Measured ~97-99 t/s warm on this box; the
# floor (50) catches 2x-class regressions with thermal headroom.
$env:GGML_VK_VVM_POOL = "1"
$log = "$env:TEMP\gpu_ci_server.log"
$proc = Start-Process -FilePath "$LlamaBin\llama-server.exe" `
    -ArgumentList '-m', "`"$Model`"", '-ngl', '99', '-c', '8192', `
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

    # completion + throughput floor (raw /completion: no template dependency)
    $body = @{
        prompt = "Question: What is 17 times 24?"
        n_predict = 64; temperature = 0; cache_prompt = $false
    } | ConvertTo-Json -Compress
    $r = Invoke-WebRequest "http://127.0.0.1:8123/completion" -Method Post `
         -Body $body -ContentType "application/json" -UseBasicParsing -TimeoutSec 300
    $j = $r.Content | ConvertFrom-Json
    $tps = [math]::Round($j.timings.predicted_per_second, 1)
    $ntok = $j.timings.predicted_n
    Gate "completion generated" ($ntok -ge 64) "($ntok tokens)"
    Gate "decode throughput floor" ($tps -ge 50) "(${tps} t/s, floor 50)"

    echo "=== GPU CI summary ==="
    if ($failed -eq 0) { echo "ALL GATES PASSED"; exit 0 }
    echo "$failed GATE(S) FAILED"; exit 1
} finally {
    if ($proc -and -not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
    Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
}
