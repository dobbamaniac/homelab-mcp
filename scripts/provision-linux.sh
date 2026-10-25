#!/usr/bin/env bash
set -euo pipefail

fqdn="${1:?Usage: $0 HOST.EXAMPLE.TLD}"
host="${fqdn%%.*}"

pubkey="${LINUX_MCP_PUBLIC_KEY:-secrets/ssh_key.pub}"
known_hosts="${LINUX_MCP_KNOWN_HOSTS:-secrets/known_hosts}"

if [[ ! -r "$pubkey" ]]; then
    echo "Public key is not readable: $pubkey" >&2
    exit 1
fi

ssh "$fqdn" "
    id homelab-mcp >/dev/null 2>&1 ||
    sudo useradd \
        --system \
        --no-create-home \
        --home-dir /nonexistent \
        --shell /bin/sh \
        --password '*NP*' \
        homelab-mcp

    sudo usermod \
        --append \
        --groups systemd-journal \
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
} | ssh "$fqdn" '
    sudo tee /etc/ssh/authorized_keys/homelab-mcp >/dev/null
    sudo chown root:root /etc/ssh/authorized_keys/homelab-mcp
    sudo chmod 0644 /etc/ssh/authorized_keys/homelab-mcp
'

echo "Creating SSH rule for homelab-mcp on $fqdn"

ssh "$fqdn" 'sudo tee /etc/ssh/sshd_config.d/90-homelab-mcp.conf >/dev/null' <<'EOF'
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

echo "Restarting sshd"

ssh "$fqdn" '
    sudo sshd -t
    sudo systemctl restart ssh
'

ssh "$fqdn" true

echo "Registering SSH host key for $host"

host_key="$(
    ssh "$fqdn" \
        'sudo cat /etc/ssh/ssh_host_ed25519_key.pub'
)"

read -r key_type key_data _ <<< "$host_key"

if [[ "$key_type" != "ssh-ed25519" || -z "$key_data" ]]; then
    echo "Unable to read ed25519 host key from $fqdn" >&2
    exit 1
fi

mkdir -p "$(dirname "$known_hosts")"
touch "$known_hosts"

new_key="$host $key_type $key_data"
existing="$(
    ssh-keygen -F "$host" -f "$known_hosts" 2>/dev/null |
        grep -v '^#' |
        head -n 1 || true
)"

if [[ -n "$existing" && "$existing" != "$new_key" ]]; then
    echo "SSH host key for $host has changed; refusing to replace it" >&2
    exit 1
fi

if [[ -z "$existing" ]]; then
    printf '%s\n' "$new_key" >> "$known_hosts"
fi

echo "Provisioned $fqdn"
