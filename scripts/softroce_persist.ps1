<#
.SYNOPSIS
    SoftRoCE (rxe) persistence for WSL2 - creates rxe links on boot

.DESCRIPTION
    Creates SoftRoCE links on eth0 and lo interfaces and installs as a systemd service
    for automatic creation on WSL2 startup.

.USAGE
    .\softroce_persist.ps1 -Install    # Install as systemd service (run as admin)
    .\softroce_persist.ps1 -Create     # Create links now
    .\softroce_persist.ps1 -Status     # Show status
    .\softroce_persist.ps1 -Uninstall  # Remove service
#>

param(
    [Parameter(Mandatory=$true, ParameterSetName='Install')]
    [switch]$Install,
    
    [Parameter(Mandatory=$true, ParameterSetName='Create')]
    [switch]$Create,
    
    [Parameter(Mandatory=$true, ParameterSetName='Status')]
    [switch]$Status,
    
    [Parameter(Mandatory=$true, ParameterSetName='Uninstall')]
    [switch]$Uninstall
)

$ErrorActionPreference = "Stop"

$ServiceName = "vulkanvm-softroce"
$ServiceFile = "/etc/systemd/system/${ServiceName}.service"
$ScriptFile = "/usr/local/bin/vulkanvm-softroce-create"
$Interfaces = @("eth0", "lo")
$LogFile = "/var/log/vulkanvm-softroce.log"

function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$timestamp $Message" | Tee-Object -FilePath $LogFile -Append | Write-Host
}

function Test-Interface {
    param([string]$Interface)
    try {
        $result = wsl -e ip link show $Interface 2>$null
        return $result -match "UP|LOWER_UP"
    } catch {
        return $false
    }
}

function Create-Links {
    Write-Log "Creating SoftRoCE links on interfaces: $($Interfaces -join ', ')"
    
    foreach ($iface in $Interfaces) {
        if (-not (Test-Interface $iface)) {
            Write-Log "Interface $iface does not exist, skipping"
            continue
        }
        
        # Check if already linked
        $alreadyLinked = $false
        $existingLinks = wsl -e bash -c "ls /sys/class/infiniband/rxe*/netdev 2>/dev/null" 2>$null
        foreach ($link in $existingLinks) {
            $netdev = wsl -e cat $link 2>$null
            if ($netdev.Trim() -eq $iface) {
                Write-Log "Interface $iface already has an rxe link"
                $alreadyLinked = $true
                break
            }
        }
        
        if ($alreadyLinked) { continue }
        
        # Find next available index
        for ($i = 0; $i -le 15; $i++) {
            $exists = wsl -e test -e "/sys/class/infiniband/rxe$i" 2>$null
            if (-not $exists) {
                Write-Log "Creating rxe$i on $iface"
                wsl -e sudo rdma link add "rxe$i" type rxe netdev $iface 2>&1 | ForEach-Object { Write-Log $_ }
                break
            }
        }
    }
    
    # Wait for ACTIVE
    Write-Log "Waiting for links to become ACTIVE..."
    for ($attempt = 1; $attempt -le 10; $attempt++) {
        $activeCount = 0
        $totalCount = 0
        $states = wsl -e bash -c "cat /sys/class/infiniband/rxe*/state 2>/dev/null" 2>$null
        foreach ($state in $states) {
            $totalCount++
            if ($state.Trim() -eq "ACTIVE") { $activeCount++ }
        }
        
        if ($totalCount -gt 0 -and $activeCount -eq $totalCount) {
            Write-Log "All $totalCount SoftRoCE links are ACTIVE"
            break
        }
        
        Start-Sleep -Seconds 1
    }
    
    Show-Status
}

function Show-Status {
    Write-Host "`n=== SoftRoCE Status ==="
    Write-Host "ibv_devices:"
    wsl -e ibv_devices 2>&1 | ForEach-Object { Write-Host "  $_" }
    
    Write-Host "`nrdma link show:"
    wsl -e rdma link show 2>&1 | ForEach-Object { Write-Host "  $_" }
    
    Write-Host "`nInterfaces:"
    foreach ($iface in $Interfaces) {
        if (Test-Interface $iface) {
            Write-Host "  $iface: UP"
        } else {
            Write-Host "  $iface: NOT FOUND"
        }
    }
}

function Install-Service {
    Write-Log "Installing systemd service..."
    
    # Create the creation script
    $scriptContent = @'
#!/bin/bash
set -euo pipefail
INTERFACES=("eth0" "lo")
LOGFILE="/var/log/vulkanvm-softroce.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOGFILE"; }

for iface in "${INTERFACES[@]}"; do
    if ! ip link show "$iface" >/dev/null 2>&1; then continue; fi
    
    already_linked=false
    for existing in /sys/class/infiniband/rxe*/netdev; do
        if [[ -e "$existing" ]] && [[ "$(cat "$existing")" == "$iface" ]]; then
            already_linked=true; break
        fi
    done
    [[ "$already_linked" == "true" ]] && continue
    
    for i in {0..15}; do
        if [[ ! -e "/sys/class/infiniband/rxe$i" ]]; then
            rdma link add "rxe$i" type rxe netdev "$iface" 2>>"$LOGFILE"
            log "Created rxe$i on $iface"
            break
        fi
    done
done

for i in {1..10}; do
    active=0; total=0
    for link in /sys/class/infiniband/rxe*/state; do
        [[ -e "$link" ]] || continue
        total=$((total + 1))
        [[ "$(cat "$link")" == "ACTIVE" ]] && active=$((active + 1))
    done
    if [[ $total -gt 0 && $active -eq $total ]]; then
        log "All $total SoftRoCE links ACTIVE"; break
    fi
    sleep 1
done
'@
    
    $scriptContent | wsl -e bash -c "cat > $ScriptFile && chmod +x $ScriptFile"
    
    # Create systemd service
    $serviceContent = @"
[Unit]
Description=VulkanVM SoftRoCE (rxe) Link Creation
Documentation=https://github.com/XxChonkExX/Vulkan-Automaton-VM
After=network-online.target
Wants=network-online.target
DefaultDependencies=no

[Service]
Type=oneshot
ExecStart=$ScriptFile
RemainAfterExit=yes
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"@
    
    $serviceContent | wsl -e bash -c "cat > $ServiceFile"
    wsl -e sudo systemctl daemon-reload
    wsl -e sudo systemctl enable $ServiceName
    Write-Log "Service installed and enabled. Start with: systemctl start $ServiceName"
}

function Uninstall-Service {
    Write-Log "Uninstalling systemd service..."
    wsl -e sudo systemctl disable $ServiceName 2>$null
    wsl -e sudo systemctl stop $ServiceName 2>$null
    wsl -e sudo rm -f $ServiceFile $ScriptFile
    wsl -e sudo systemctl daemon-reload
    Write-Log "Service uninstalled"
}

# Main
if ($Install) { Install-Service }
elseif ($Create) { Create-Links }
elseif ($Status) { Show-Status }
elseif ($Uninstall) { Uninstall-Service }