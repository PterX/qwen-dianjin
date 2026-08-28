# Third-party notices

This file records direct third-party components used or downloaded by the
benchmark. Package managers and container images may have additional transitive
dependencies; users must also review the notices shipped by those components.

## Service-world components

| Component | Use in this repository | License | Source |
|---|---|---|---|
| Playwright Python | Browser automation in the repository-local browser service. | Apache-2.0 | https://github.com/microsoft/playwright-python |
| FastMCP | MCP server framework used by service implementations. | Apache-2.0 | https://github.com/PrefectHQ/fastmcp |
| Mailpit | Local synthetic email capture service. | MIT | https://github.com/axllent/mailpit |
| Blnk | Local double-entry ledger used by Banking cases. | Apache-2.0 | https://github.com/blnkfinance/blnk |

## Agent runtimes

Runtime packages are downloaded during an explicitly selected Docker build;
their binaries are not stored in this repository.

| Runtime | Default build | License/status | Source |
|---|---:|---|---|
| Codex CLI | yes | Apache-2.0 | https://github.com/openai/codex |
| OpenClaw | yes | MIT | https://github.com/openclaw/openclaw |
| Hermes Agent | optional | MIT | https://github.com/NousResearch/hermes-agent |
| Claude Code | optional, not benchmark-validated in this release | Anthropic terms; not relicensed by this project | https://docs.anthropic.com/en/docs/claude-code/getting-started |
| WorkBuddy / CodeBuddy Code | optional, not benchmark-validated in this release | Vendor package terms; not relicensed by this project | https://www.npmjs.com/package/@tencent-ai/codebuddy-code |

The Dockerfiles install pinned package versions from public upstream sources.
Building or using a runtime may require accepting its vendor terms and using
separately obtained credentials. The project MIT license does not grant rights
to those runtime packages or services.
