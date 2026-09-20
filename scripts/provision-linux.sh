#!/usr/bin/env bash
set -euo pipefail

host="${1:?Usage: $0 HOST}"
pubkey="${LINUX_MCP_PUBLIC_KEY:-secrets/linux_ssh_key.pub}"

echo "Creating homelab-mcp on $host"

ssh "$host" "
    id homelab-mcp >/dev/null 2>&1 ||
    sudo useradd \
        --system \
        --no-create-home \
        --home-dir /nonexistent \
        --shell /bin/sh \
        --password '*NP*' \
        homelab-mcp

    sudo install \
        -d \
        -o root \
        -g root \
        -m 0755 \
        /etc/ssh/authorized_keys
"

{
    printf 'restrict '
    cat "$pubkey"
} | ssh "$host" '
    sudo tee /etc/ssh/authorized_keys/homelab-mcp >/dev/null
    sudo chown root:root /etc/ssh/authorized_keys/homelab-mcp
    sudo chmod 0644 /etc/ssh/authorized_keys/homelab-mcp
'

ssh "$host" 'sudo tee /etc/ssh/sshd_config.d/90-homelab-mcp.conf >/dev/null' <<'EOF'
Match User homelab-mcp
    AuthorizedKeysFile /etc/ssh/authorized_keys/%u

    AuthenticationMethods publickey
    PubkeyAuthentication yes
    PasswordAuthentication no
    KbdInteractiveAuthentication no

    PermitTTY no
    DisableForwarding yes
    PermitTunnel no
    PermitUserRC no
EOF

ssh "$host" '
    sudo sshd -t
    sudo systemctl restart ssh
'

echo "Provisioned $host"
