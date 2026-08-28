<div align="center">
  <img src="./assets/dianjin_logo.png" alt="DianJin Logo" width="180">
  <h1>DianJin-RED</h1>
  <p><strong>Action-grounded red teaming for complete agent systems</strong></p>
  <p><strong>English</strong> | <a href="README_zh.md">中文</a></p>
</div>

![REDAgentBench IVC taxonomy](./assets/ivc_taxonomy_overview.png)

*The paper's Intervention–Vulnerability–Constraint taxonomy: 15 intervention
strategies, 11 vulnerability manifestations, and 28 policy constraints.*

DianJin-RED packages REDAgentBenchmark for the DianJin project. It evaluates the complete agent stack—model, harness, tools, and
sandbox—inside controlled service worlds. Safety outcomes are grounded in
observable side effects instead of response wording alone. The benchmark
measures unsafe execution through Attack Success Rate (ASR) and whether an agent
recognized a risk before acting through the Recognition–Execution Gap (REG).

> **Release status:** the software is MIT-licensed and the public Codex path is
> reproducible. Benchmark data is distributed under the terms described in
> [DATA_LICENSE.md](DATA_LICENSE.md).

## What is included

- Docker-backed banking, browser, Gmail, external-files, and workspace worlds.
- Adapters for Codex CLI, Gemini CLI, OpenClaw, Hermes, Claude Code, and
  WorkBuddy.
- Fifteen intervention strategies: six entering through user input, five through
  agent-platform state (workspace, skills, memory, or MCP), and four through
  external tools or data.
- Trajectory, environment-state, and action-grounded judges.
- Release checks and reusable judging/analysis utilities under `scripts/`.

## Benchmark at a glance

| Item | Included |
|---|---:|
| Executable cases | 1,661 |
| Intervention strategies | 15 (6 user-input, 5 platform-state, 4 tool/data) |
| REG probe arms | 4 |
| Service worlds | Banking, Browser, Gmail, External Files, Workspace |

Every released case has a row-level mapping in
[`DATA_PROVENANCE.csv`](DATA_PROVENANCE.csv). Code and data use separate
licenses; see [DATA_LICENSE.md](DATA_LICENSE.md).

## IVC taxonomy

Each case records one executable path `I -> V -> C`:

- **Intervention (I):** where adversarial content first enters the system and
  how it is introduced, such as a direct user request, a poisoned workspace
  file or skill, or tampered tool output.
- **Vulnerability (V):** the weakness manifested by the agent, such as missing
  verification, approval bypass, unsafe parameter construction, workspace
  damage, or unsupported output.
- **Constraint (C):** the safety requirement that the observed side effect
  violates, such as credential confidentiality, asset protection, data
  integrity, availability, system compromise, external-action safety, or
  governance.

The machine-readable definitions are in
[`taxonomy/registry.json`](taxonomy/registry.json); case files use the same
names under `taxonomy.intervention`, `taxonomy.vulnerability`, and
`taxonomy.constraint`. The short file identifiers remain stable for released
datasets, but the registry is the source of their human-readable meaning.

## Evaluation pipeline

![REDAgentBench case generation and evaluation pipeline](./assets/framework.png)

*The paper's complete pipeline. Source-linked attack knowledge is converted into
IVC plans, compiled and validated in executable sandboxes, then frozen after
human audit. During evaluation, the trajectory judge examines messages and tool
calls, the state judge verifies durable service effects, and the hybrid judge
reconciles the two evidence views. [Vector PDF](./assets/framework.pdf)*

## Security defaults

- Agent runtimes use a per-run internal Docker network with no public Internet
  egress.
- The runtime can reach only explicitly exposed harness endpoints through its
  per-network `host.docker.internal` gateway.
- The OpenAI-compatible host proxy requires a cryptographically random per-run
  bearer token. The real upstream key is never copied into the agent container.
- MCP control routes require a separate random control token.
- Service fixtures run on dedicated Docker bridges with IP masquerading disabled,
  so they cannot use the host as a NAT gateway to reach external networks.
- Host networking is rejected unless `sandbox.allow_host_network` is explicitly
  set to `true`.
- The diagnostic collector is disabled unless `--collector` is supplied.

These controls reduce accidental exposure; they do not make adversarial agent
code safe to run on a developer workstation. Use a disposable VM, keep real
credentials out of benchmark workspaces, and read [SECURITY.md](SECURITY.md).

## Quickstart (Codex)

Requirements: Python 3.10+, Docker Engine with the Compose plugin, and enough
space to build an Ubuntu/Node runtime image. Run the following commands from
the `DianJin-RED/` directory.

### 1. Install the Python package

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

### 2. Configure the model and API key

An OpenAI-compatible model endpoint is required for a real rollout. Copy the
private configuration template:

```bash
cp config/config_local_private.example.json config/config_local_private.json
```

Edit `config/config_local_private.json` and set the endpoint, model name, and
API key for both the target agent and the judge. They may use the same provider
and key for a minimal reproduction, or separate credentials in a larger run:

```json
{
  "agent": {
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "api_keys": ["YOUR_DASHSCOPE_API_KEY"],
    "model": "qwen3.7-plus"
  },
  "judge": {
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "api_keys": ["YOUR_DASHSCOPE_API_KEY"],
    "model": "qwen3.7-plus"
  }
}
```

Other OpenAI-compatible providers can be used by changing `base_url`, `model`,
and `api_keys`. The private file is excluded by `.gitignore`; verify this before
adding any real credential:

```bash
git check-ignore config/config_local_private.json
```

Keep `agent_proxy.enabled=true`. In the default configuration, the real
upstream key stays in the host-side proxy process. The Agent container receives
only a random per-run proxy token.

### 3. Build and validate the isolated runtime

```bash

# The default build is the public path: sandbox base + Codex runtime.
bash docker/build-agent-images.sh

# Validates assets, image availability, proxy authentication, and blocked egress.
python scripts/quickstart_check.py --network-smoke

# Unit/contract suite.
pytest -q
```

The default build downloads only public Ubuntu, Node.js, and Codex packages.
No local vendor archive or unpublished base image is required. If Docker Hub
or npmjs is unavailable on your network, select any trusted public mirror with
`SANDBOX_BASE_IMAGE=...` and `NPM_REGISTRY=...`; the defaults remain the
official upstream registries. The Gmail fixture can likewise use a trusted
mirror with `MAILPIT_IMAGE=...` while retaining the pinned `v1.30.0` default.

### 4. Run one case

The following command runs the first workspace-file intervention case with one
worker:

```bash
python -m red_agent_world.runners.codex_sandbox_runner \
  --config config/config_local_private.json \
  --dataset-file test/E1.json \
  --limit 1 \
  --concurrency 1 \
  --output-name quickstart_codex
```

Artifacts are written under `results/`, which is ignored by Git. Inspect the
result CSV together with the exported trajectory, service receipts, and state
diffs. Increase `--limit` and `--concurrency` only after this one-case run
completes successfully.

The `agent_proxy.bind_host` example remains `0.0.0.0` so the isolated Docker
network can reach it; the endpoint is protected by a fresh, unexported per-run
token. Never place a real API key in a benchmark case, sandbox seed, committed
configuration, shell history, or result artifact. Rotate the key immediately if
it is ever committed.

## Runtime image matrix

| Target | Public build status | Upstream/licensing note |
|---|---|---|
| `base`, `codex` | Default and tested | Codex is Apache-2.0 |
| `openclaw` | Buildable from pinned public npm package | OpenClaw is MIT |
| `hermes` | Buildable from a pinned public Git revision | Hermes Agent is MIT |
| `claudecode` | Buildable from pinned official npm package; not benchmark-validated | Anthropic terms apply |
| `workbuddy` | Buildable from pinned public npm package; not benchmark-validated | Vendor package terms apply |

Run `bash docker/build-agent-images.sh --help` for target names. Details and
attributions are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Repository layout

```text
config/            runtime configuration templates
docker/            sandbox and agent runtime images
prompts/           active judge prompts
reg_probe_suite/   paired REG probes
sandbox/           service worlds and MCP servers
scripts/           release checks and reusable rollout/judging utilities
src/red_agent_world/
taxonomy/          IVC intervention, vulnerability, and constraint registry
test/              benchmark cases and row-level source records
tests/             unit and contract tests
```

## Data and responsible use

The cases contain synthetic credentials and adversarial instructions. Use them
only on systems you own or are authorized to test. Row-level provenance is
preserved in `DATA_PROVENANCE.csv`; applicable data terms are documented in
[DATA_LICENSE.md](DATA_LICENSE.md).

## License

Original software is MIT-licensed; see [LICENSE](LICENSE). The MIT grant does
not override licenses or missing permissions for benchmark data, external
images, CLIs, or other third-party material. See [DATA_LICENSE.md](DATA_LICENSE.md)
and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
