#!/usr/bin/env bash
# ============================================================================
# wsl2_rxe_kernel.sh - Build a custom WSL2 kernel with Soft-RoCE (rdma_rxe)
# ============================================================================
# Microsoft's stock WSL2 kernel does NOT include rdma_rxe. Building a custom
# kernel from https://github.com/microsoft/WSL2-Linux-Kernel with CONFIG_RDMA_RXE
# enables a full verbs/RDMA environment inside WSL2 (loopback or cross-machine
# to a native Linux host over the LAN).
#
#   - ibv_reg_dmabuf_mr on rxe is NOT in any released kernel (as of 2026):
#     rxe is a CPU memcpy driver, so the AMD GPU-direct path in WSL2 is
#     limited to the DMA-BUF mmap fallback, and GPU dma-buf export itself is
#     blocked by the WSL2 virtual GPU (wslgd) - see docs.
#   - The verbs control/data path over HOST memory works fine in WSL2 and can
#     be validated end-to-end (including against Evo-X2 over the LAN).
#
# Usage:
#   ./scripts/wsl2_rxe_kernel.sh prepare   # install kernel build deps (sudo)
#   ./scripts/wsl2_rxe_kernel.sh build     # clone + patch config + build kernel
#   ./scripts/wsl2_rxe_kernel.sh install   # copy bzImage + set kernel= in .wslconfig
#   wsl --shutdown                          # from Windows PowerShell, then reopen
#   ./scripts/wsl2_rxe_kernel.sh verify    # load rxe, create link, show devices
#   ./scripts/wsl2_rxe_kernel.sh loopback  # veth pair + ib_write_bw self-test
#
# Reference: https://ricli.tech/wsl-rxe/ and microsoft/WSL2-Linux-Kernel
# ============================================================================
set -u

GREEN=$'\033[1;32m'; YELLOW=$'\033[1;33m'; RED=$'\033[1;31m'; NC=$'\033[0m'
log()  { echo "${GREEN}[+]${NC} $*"; }
warn() { echo "${YELLOW}[!]${NC} $*"; }
err()  { echo "${RED}[-]${NC} $*"; }

# Override with WSL_KERNEL_TAG=<branch> WSL_KERNEL_SRC=<dir>
WSL_KERNEL_TAG="${WSL_KERNEL_TAG:-linux-msft-wsl-6.18.y}"
WSL_KERNEL_SRC="${WSL_KERNEL_SRC:-$HOME/wsl-kernel}"
KCONFIG="arch/x86/configs/config-wsl"

cmd_prepare() {
    log "Installing kernel build dependencies..."
    sudo apt-get update
    sudo apt-get install -y build-essential flex bison pahole libelf-dev bc \
        libncurses-dev pkg-config libssl-dev git
    log "Also installing RDMA userspace tooling for verification..."
    sudo apt-get install -y rdma-core perftest infiniband-diags ibverbs-utils
    log "Done. Run: ./scripts/wsl2_rxe_kernel.sh build"
}

cmd_build() {
    if [ ! -f "$WSL_KERNEL_SRC/arch/x86/configs/config-wsl" ]; then
        log "Cloning microsoft/WSL2-Linux-Kernel ($WSL_KERNEL_TAG)..."
        git clone --depth 1 --branch "$WSL_KERNEL_TAG" \
            https://github.com/microsoft/WSL2-Linux-Kernel.git "$WSL_KERNEL_SRC"
    else
        log "Kernel source already present at $WSL_KERNEL_SRC"
    fi

    cd "$WSL_KERNEL_SRC" || exit 1
    log "Enabling RDMA / Soft-RoCE config options..."
    # Stock WSL config: INFINIBAND=n. Enable the IB core, userspace verbs,
    # rdma-cm (librdmacm), and Soft-RoCE; add a distinctive localversion.
    ./scripts/config --file "$KCONFIG" \
        --enable INFINIBAND \
        --enable INFINIBAND_VIRT_DMA \
        --enable INFINIBAND_USER_ACCESS \
        --enable RDMA_CM \
        --enable RDMA_UCMA \
        --enable RDMA_RXE \
        --set-str LOCALVERSION "-wsl-rxe"

    log "Building kernel (this takes a while, use WSL_KERNEL_TAG to switch branch)..."
    make -j"$(nproc)" KCONFIG_CONFIG="$KCONFIG"

    if [ ! -f arch/x86/boot/bzImage ]; then
        err "Build failed: arch/x86/boot/bzImage not found"
        exit 1
    fi
    log "Kernel built: $WSL_KERNEL_SRC/arch/x86/boot/bzImage"
    log "Next: ./scripts/wsl2_rxe_kernel.sh install"
}

cmd_install() {
    local bz="$WSL_KERNEL_SRC/arch/x86/boot/bzImage"
    if [ ! -f "$bz" ]; then
        err "No kernel image found. Run: ./scripts/wsl2_rxe_kernel.sh build"
        exit 1
    fi

    local win_user win_home
    win_user=$(powershell.exe -NoProfile -Command '$env:USERNAME' 2>/dev/null | tr -d '\r')
    win_home=$(powershell.exe -NoProfile -Command '$env:USERPROFILE' 2>/dev/null | tr -d '\r')
    if [ -z "$win_user" ]; then
        err "Could not detect Windows user. Set WIN_USER=<name> and retry."
        exit 1
    fi

    local win_kernel_dir="$win_home\\.wsl-kernel"
    powershell.exe -NoProfile -Command "New-Item -ItemType Directory -Force -Path '$win_kernel_dir' | Out-Null" || true
    log "Copying bzImage to $win_kernel_dir\\bzImage ..."
    cp -f "$bz" "/mnt/c/Users/$win_user/.wsl-kernel/bzImage" || {
        err "Copy to /mnt/c/Users/$win_user/.wsl-kernel/ failed"
        exit 1
    }

    log "Updating .wslconfig (kernel= line)..."
    local cfg="/mnt/c/Users/$win_user/.wslconfig"
    # .wslconfig wants escaped backslashes (C:\\Users\\...\\bzImage)
    local wsl_path="${win_kernel_dir//\\/\\\\}\\bzImage"
    if [ -f "$cfg" ]; then
        if grep -qi '^kernel\s*=' "$cfg"; then
            perl -i -pe "s/^kernel\\s*=.*/kernel=$wsl_path/ if /^kernel\\s*=/" "$cfg"
        else
            if ! grep -q '^\[wsl2\]' "$cfg"; then
                printf '\n[wsl2]\n' >> "$cfg"
            fi
            printf 'kernel=%s\n' "$wsl_path" >> "$cfg"
        fi
    else
        printf '[wsl2]\nkernel=%s\n' "$wsl_path" > "$cfg"
    fi

    log "Done. From Windows PowerShell run:  wsl --shutdown"
    log "Then reopen this distro and run:  ./scripts/wsl2_rxe_kernel.sh verify"
}

cmd_verify() {
    log "Kernel: $(uname -r)"
    case "$(uname -r)" in
        *-wsl-rxe) ;;
        *) warn "Kernel does NOT have the -wsl-rxe localversion - did you install/restart?" ;;
    esac

    if ! lsmod 2>/dev/null | grep -q rdma_rxe; then
        log "Loading rdma_rxe module..."
        if ! sudo modprobe rdma_rxe 2>/dev/null; then
            err "modprobe rdma_rxe failed. If kernel build worked, try 'wsl --shutdown' and reopen."
            exit 1
        fi
    fi

    local iface
    iface=$(ip route get 1.1.1.1 2>/dev/null | grep -oP 'dev \K[^ ]+' | head -1)
    iface="${iface:-eth0}"
    if ! ls /sys/class/infiniband/ 2>/dev/null | grep -q rxe; then
        log "Creating rxe0 over $iface ..."
        sudo rdma link add rxe0 type rxe netdev "$iface" 2>/dev/null \
            || warn "rdma link add failed (already exists?)"
    fi

    log "ibv_devices:"
    ibv_devices 2>/dev/null || warn "no RDMA devices visible"
    log "rdma link show:"
    rdma link show 2>/dev/null
    log "GIDs (RoCEv2):"
    ibv_devinfo -v 2>/dev/null | grep -E 'GID\[|active_speed|state:' | head -8

    log "Verbs check: rdma_cm + librxe userspace present?"
    ls /usr/lib/*/libibverbs/librxe* 2>/dev/null || ls /usr/lib/x86_64-linux-gnu/libibverbs/librxe* 2>/dev/null \
        || warn "librxe provider not found (apt install rdma-core)"
    log "Next: ./scripts/wsl2_rxe_kernel.sh loopback  (or use scripts/test_amd_rdma.sh)"
}

cmd_loopback() {
    # Two rxe devices over a veth pair, then ib_write_bw across them.
    local a=192.168.96.110 b=192.168.96.111
    sudo ip link add veth0 type veth peer name veth1 2>/dev/null || warn "veth pair exists?"
    sudo ip addr add "$a/24" dev veth0 2>/dev/null
    sudo ip addr add "$b/24" dev veth1 2>/dev/null
    sudo ip link set veth0 up
    sudo ip link set veth1 up
    sudo sysctl -w net.ipv4.conf.veth0.accept_local=1 net.ipv4.conf.veth1.accept_local=1 >/dev/null

    sudo modprobe rdma_rxe 2>/dev/null
    sudo rdma link add rxe_0 type rxe netdev veth0 2>/dev/null || warn "rxe_0 exists?"
    sudo rdma link add rxe_1 type rxe netdev veth1 2>/dev/null || warn "rxe_1 exists?"

    log "Running ib_write_bw self-test (rxe_0 -> rxe_1, 16MiB x 5)..."
    ib_write_bw -d rxe_1 -R -q 8 -s 16 -n 5 >/tmp/ibw_server.log 2>&1 &
    local spid=$!
    sleep 2
    ib_write_bw -d rxe_0 -R -q 8 -s 16 -n 5 "$a" | grep -E '65536|16 MiB|BW'
    kill "$spid" 2>/dev/null

    log "Loopback test done. Cross-machine: create rxe0 over eth0, then"
    log "use scripts/test_amd_rdma.sh (server/client) against the X2."
}

case "${1:-}" in
    prepare) cmd_prepare ;;
    build)   cmd_build ;;
    install) cmd_install ;;
    verify)  cmd_verify ;;
    loopback) cmd_loopback ;;
    *)
        sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'
        exit 1
        ;;
esac
