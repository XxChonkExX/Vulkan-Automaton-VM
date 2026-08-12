#!/bin/bash
# ============================================================================
# VulkanVM AMD-Specific RDMA Path Test (Linux/Ubuntu)
#
# Exercises the vendor-specific GPU-direct RDMA path for AMD GPUs (0x1002):
#   DMA-BUF export (vkGetMemoryFdKHR) -> ibv_reg_dmabuf_mr (verbs) -> RDMA MR
#
# Topology:
#   SERVER: Evo-X2 (AMD Strix Halo 395 / Radeon 890M) - tensor_server_test
#   CLIENT: Ubuntu box here (AMD Radeon)                - tensor_client_test
#
# Since neither box has InfiniBand hardware, Soft-RoCE (rdma_rxe) is used as
# the RDMA device. Requires rdma-core + a kernel with CONFIG_RDMA_RXE (any
# stock Ubuntu 22.04+/24.04+ kernel). NOTE: WSL2 kernels do NOT include
# rdma_rxe - run this on a native Ubuntu install.
#
# Kernel notes (affects which AMD sub-path engages):
#   kernel >= 6.19  : rxe supports dma-buf MRs -> "AMD GPUDirect via
#                     ibv_reg_dmabuf_mr" (true zero-copy registration)
#   kernel <  6.19  : falls back to mmap of the dma-buf -> "DMA-BUF mmap
#                     fallback" (still RDMA, staged through CPU mapping).
#                     On Strix Halo / APU unified memory this works fine.
#
# Usage:
#   sudo ./test_amd_rdma.sh setup          # deps + Soft-RoCE, idempotent
#   sudo ./test_amd_rdma.sh build          # cmake configure + build tests
#   sudo ./test_amd_rdma.sh env            # print RDMA/GPU environment status
#         ./test_amd_rdma.sh server [opts] # run server (foreground)
#         ./test_amd_rdma.sh client [opts] # run client, auto-verify AMD path
#
# Options (server/client):
#   --iface <if>     Ethernet interface for the rxe link (default: default route)
#   --port <p>       TCP port (server listens; client connects) (default: 51000)
#   --local-port <p> Client local listen port (default: 51005)
#   --server-ip <ip> Server address for client (default: 192.168.0.117)
#   --size-mb <n>    Tensor size in MB (default: 16)
#   --rdma-nic <n>   RDMA device name (default: rxe0)
#   --announce-count <n> Server announces n times then exits (0 = infinite, default)
#   --build-dir <d>  Build directory (default: build_rdma)
#   --repo <dir>     Repo root (default: parent of script dir)
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(dirname "$SCRIPT_DIR")}"
BUILD_DIR="${BUILD_DIR:-$REPO/build_rdma}"
ROLE="${1:-help}"
shift || true

IFACE=""
PORT=51000
LOCAL_PORT=51005
SERVER_IP=192.168.0.117
SIZE_MB=16
RDMA_NIC=rxe0
ANNOUNCE_COUNT=0

log()  { echo -e "\033[1;32m[+] $*\033[0m"; }
warn() { echo -e "\033[1;33m[!] $*\033[0m"; }
err()  { echo -e "\033[1;31m[x] $*\033[0m" >&2; }

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --iface)      IFACE="$2"; shift 2 ;;
            --port)       PORT="$2"; shift 2 ;;
            --local-port) LOCAL_PORT="$2"; shift 2 ;;
            --server-ip)  SERVER_IP="$2"; shift 2 ;;
            --size-mb)    SIZE_MB="$2"; shift 2 ;;
            --rdma-nic)   RDMA_NIC="$2"; shift 2 ;;
            --announce-count) ANNOUNCE_COUNT="$2"; shift 2 ;;
            --build-dir)  BUILD_DIR="$2"; shift 2 ;;
            --repo)       REPO="$2"; shift 2 ;;
            *) warn "Unknown option: $1"; shift ;;
        esac
    done
}

require_root() {
    if [[ $EUID -ne 0 ]]; then
        err "Please run as root (sudo)."
        exit 1
    fi
}

detect_iface() {
    if [[ -n "$IFACE" ]]; then return; fi
    IFACE="$(ip route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="dev"){print $(i+1); exit}}')"
    if [[ -z "$IFACE" ]]; then
        err "Could not auto-detect network interface; pass --iface <name>"
        exit 1
    fi
    log "Using interface: $IFACE"
}

is_wsl() {
    [[ "$(uname -r)" == *microsoft* ]]
}

# ---------------------------------------------------------------------------
# setup: install packages, load rdma_rxe, create rxe link
# ---------------------------------------------------------------------------
do_setup() {
    require_root
    detect_iface

    if is_wsl; then
        warn "WSL2 detected: rdma_rxe is NOT available in WSL2 kernels."
        warn "Install native Ubuntu for the RDMA client. Continuing anyway..."
    fi

    log "Installing build + RDMA + Vulkan packages..."
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -y
    apt-get install -y \
        build-essential cmake ninja-build pkg-config git \
        libvulkan-dev vulkan-tools mesa-vulkan-drivers \
        rdma-core libibverbs-dev librdmacm-dev iproute2

    log "Loading rdma_rxe kernel module..."
    modprobe rdma_rxe 2>/dev/null || warn "modprobe rdma_rxe failed (already builtin?)"

    log "Creating Soft-RoCE link $RDMA_NIC on $IFACE..."
    if ! rdma link show "$RDMA_NIC" >/dev/null 2>&1; then
        rdma link add "$RDMA_NIC" type rxe netdev "$IFACE"
    fi

    log "Waiting for link to become ACTIVE..."
    for i in $(seq 1 15); do
        if [[ -f "/sys/class/infiniband/$RDMA_NIC/state" ]] && \
           [[ "$(cat "/sys/class/infiniband/$RDMA_NIC/state")" == "ACTIVE" ]]; then
            break
        fi
        sleep 1
    done

    do_env
}

# ---------------------------------------------------------------------------
# env: report RDMA + GPU status
# ---------------------------------------------------------------------------
do_env() {
    echo
    log "== Kernel =="
    uname -r
    if is_wsl; then warn "WSL2 detected - rdma_rxe will not work here"; fi

    kver="$(uname -r)"
    if [[ "$(printf '%s\n' "$kver" "6.19" | sort -V | head -1)" == "$kver" ]]; then
        warn "Kernel < 6.19: rxe dma-buf MR unsupported, AMD path will use mmap fallback"
    else
        log "Kernel >= 6.19: rxe dma-buf MR supported"
    fi

    echo
    log "== RDMA Devices =="
    ibv_devices 2>/dev/null || warn "no RDMA devices (rdma_rxe not loaded?)"
    echo
    echo "rdma link show:"
    rdma link show 2>/dev/null || true

    echo
    log "== GPU / Vulkan =="
    if command -v vulkaninfo >/dev/null 2>&1; then
        vulkaninfo --summary 2>/dev/null | grep -E "deviceName|driverName|deviceType" || \
            vulkaninfo 2>/dev/null | grep -E "deviceName" | head -4
    else
        warn "vulkan-tools not installed"
    fi

    echo
    log "== GPU vendor path (what the RDMA test expects) =="
    echo "  AMD (0x1002)  -> DMA-BUF export + ibv_reg_dmabuf_mr (mmap fallback < 6.19)"
    echo "  NVIDIA (0x10DE) -> vkGetMemoryRemoteAddressNV + peermem"
    echo "  Intel (0x8086) -> Level Zero / DMA-BUF"
}

# ---------------------------------------------------------------------------
# build: configure and build the test binaries
# ---------------------------------------------------------------------------
do_build() {
    require_root
    if [[ ! -d "$REPO/CMakeLists.txt" ]]; then
        err "Repo not found at $REPO (pass --repo <dir> or REPO=...)"
        exit 1
    fi
    log "Configuring build in $BUILD_DIR..."
    local cfg_out
    cfg_out="$(cmake -S "$REPO" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DVVM_BUILD_NETWORK=ON \
        -DVVM_BUILD_PYTORCH=OFF \
        -DVVM_BUILD_ONNX=OFF \
        -DVVM_USE_VOLK=ON 2>&1)"

    echo "$cfg_out" | grep -E "network:|verbs|RDMA" || true
    if ! grep -q "VVM_NETWORK_HAS_VERBS=1" "$BUILD_DIR"/CMakeFiles/*/flags.make 2>/dev/null \
        && ! grep -q "RDMA/verbs transport enabled" <<< "$cfg_out"; then
        warn "VERBS NOT DETECTED - RDMA path will be disabled. Check: dpkg -l | grep libibverbs-dev"
    else
        log "Verbs transport enabled - RDMA path will be compiled in"
    fi

    log "Building tensor_server_test / tensor_client_test..."
    cmake --build "$BUILD_DIR" -j"$(nproc)" --target tensor_server_test tensor_client_test

    ls -la "$BUILD_DIR/examples/tensor_server_test" "$BUILD_DIR/examples/tensor_client_test"
    echo
    log "Build complete. Binaries:"
    echo "  server: $BUILD_DIR/examples/tensor_server_test"
    echo "  client: $BUILD_DIR/examples/tensor_client_test"
}

# ---------------------------------------------------------------------------
# run: launch server or client, capture logs, verify AMD RDMA path
# ---------------------------------------------------------------------------
verify_log() {
    local logfile="$1"
    local rc=0

    echo
    log "== AMD RDMA path verification =="
    local markers=(
        "VerbsRdmaTransport initialized"
        "Vendor registration succeeded for 0x1002"
        "Vendor registration unavailable for 0x1002"
        "AMD GPUDirect via ibv_reg_dmabuf_mr"
        "AMD GPUDirect via DMA-BUF mmap fallback"
        "RDMA transport initialization failed"
        "migrateFromRemote: pulled"
        "Joined cluster via"
    )
    for m in "${markers[@]}"; do
        if grep -qF "$m" "$logfile"; then
            log "FOUND : $m"
        else
            echo "       (absent) $m"
        fi
    done

    echo
    log "== AMD-specific lines =="
    grep -aE "GPUDirect|Vendor registration|VerbsRdmaTransport|rdma" "$logfile" | \
        grep -avE "rdmaCapable|rdmaAddr" | sed 's/^/    /' | head -20 || true

    if grep -qF "RDMA transport initialization failed" "$logfile"; then
        err "FAIL: RDMA transport failed to initialize"
        rc=1
    fi
    if grep -qF "VerbsRdmaTransport initialized" "$logfile"; then
        if grep -qE "AMD GPUDirect via (ibv_reg_dmabuf_mr|DMA-BUF mmap fallback)" "$logfile"; then
            log "PASS: AMD vendor RDMA path engaged (GPU memory registered with verbs)"
        else
            warn "RDMA transport up, but no AMD GPU registration seen (import side only?)"
        fi
    fi
    if grep -qF "migrateFromRemote: pulled" "$logfile"; then
        log "PASS: tensor data transferred over the wire"
    fi
    return $rc
}

run_server() {
    parse_args "$@"
    local exe="$BUILD_DIR/examples/tensor_server_test"
    if [[ ! -x "$exe" ]]; then
        err "Server binary missing. Run: sudo ./test_amd_rdma.sh build"
        exit 1
    fi

    local logfile="/tmp/vvm_server_rdma.log"
    log "Starting server: port=$PORT size=${SIZE_MB}MB rdma-nic=$RDMA_NIC"
    log "Log: $logfile  (Ctrl+C to stop)"
    echo
    local extra=()
    if [[ "$ANNOUNCE_COUNT" -gt 0 ]]; then
        extra+=(--announce-count "$ANNOUNCE_COUNT")
        log "Server will announce $ANNOUNCE_COUNT times then exit"
    fi
    "$exe" --port "$PORT" --size-mb "$SIZE_MB" --rdma-nic "$RDMA_NIC" --verbose "${extra[@]}" 2>&1 | tee "$logfile"
    verify_log "$logfile"
}

run_client() {
    parse_args "$@"
    local exe="$BUILD_DIR/examples/tensor_client_test"
    if [[ ! -x "$exe" ]]; then
        err "Client binary missing. Run: sudo ./test_amd_rdma.sh build"
        exit 1
    fi

    local logfile="/tmp/vvm_client_rdma.log"
    log "Starting client: server=$SERVER_IP:$PORT local=$LOCAL_PORT rdma-nic=$RDMA_NIC"
    log "Log: $logfile"
    echo
    "$exe" --server "$SERVER_IP" --port "$PORT" --local-port "$LOCAL_PORT" \
           --rdma-nic "$RDMA_NIC" 2>&1 | tee "$logfile"
    local rc=${PIPESTATUS[0]}
    verify_log "$logfile"
    return $rc
}

# ---------------------------------------------------------------------------
case "$ROLE" in
    setup)  do_setup ;;
    env)    do_env ;;
    build)  do_build ;;
    server) run_server "$@" ;;
    client) run_client "$@" ;;
    help|-h|--help|"")
        sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'
        ;;
    *)
        err "Unknown role: $ROLE (setup|env|build|server|client)"
        exit 1
        ;;
esac