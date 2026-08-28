# Security policy

## Supported versions

Until the first stable release, security fixes are made on the default branch.

## Reporting a vulnerability

Do not open a public issue containing credentials, exploit details, or a working
escape from the sandbox. Use the repository host's private security-advisory
channel after publication. Maintainers should acknowledge a report within seven
days and coordinate disclosure after a fix is available.

## Threat model and operating guidance

REDAgentBench intentionally executes adversarial prompts and agent-generated
commands. Run it in a disposable, access-controlled VM. Do not mount SSH keys,
cloud credentials, Docker configuration, a home directory, or production data
into an agent runtime.

Secure defaults include an internal per-run Docker network, authenticated local
model proxy, authenticated MCP control routes, random per-run OpenClaw gateway
tokens, and an opt-in diagnostic collector. Service fixtures use dedicated
Docker bridges with IP masquerading disabled: the runner can reach their
explicitly published ephemeral ports, while the fixtures cannot use the host as
a NAT gateway for external egress. `network_mode=host` disables the runtime
network boundary and is rejected unless `sandbox.allow_host_network=true`; use
that override only inside an already isolated VM.

Service fixtures use synthetic credentials and ephemeral state. Treat their
ports and generated artifacts as sensitive during a run, and do not expose the
host to an untrusted LAN. Stop orphaned Compose projects before reusing a host.
