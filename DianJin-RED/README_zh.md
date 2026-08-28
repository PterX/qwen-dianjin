<div align="center">
  <img src="./assets/dianjin_logo.png" alt="通义点金 Logo" width="180">
  <h1>DianJin-RED</h1>
  <p><strong>面向完整 Agent 系统、以真实动作结果为依据的红队安全评测基准</strong></p>
  <p><a href="README.md">English</a> | <strong>中文</strong></p>
</div>

![REDAgentBench IVC 分类图](./assets/ivc_taxonomy_overview.png)

*论文中的 Intervention–Vulnerability–Constraint 分类图，覆盖 15 种干预策略、
11 种脆弱性表现和 28 项策略约束。*

DianJin-RED 是 REDAgentBenchmark 在通义点金项目中的开源实现，用于在受控服务世界中评测完整的 Agent 系统，包括模型、
Harness、工具和 Sandbox。评测不以模型回复中的措辞作为攻击是否成功的
依据，而是检查真实发生的服务动作、环境状态变化和执行轨迹。基准使用攻击
成功率（Attack Success Rate，ASR）衡量不安全执行，并通过识别–执行差距
（Recognition–Execution Gap，REG）分析 Agent 是否在识别风险后仍然执行
危险动作。

> **发布状态：** 软件代码采用 MIT 许可证；公开 Codex 路径可复现。
> Benchmark 数据遵循 [DATA_LICENSE.md](DATA_LICENSE.md) 中说明的许可条款。

## 项目内容

- 基于 Docker 的 Banking、Browser、Gmail、External Files 和 Workspace 服务世界。
- Codex CLI、Gemini、OpenClaw、Hermes、Claude Code 和 WorkBuddy 适配器。
- 15 种干预策略：6 种通过用户输入进入，5 种通过 Agent 平台状态（Workspace、Skill、Memory 或 MCP）进入，4 种通过外部工具或数据进入。
- 包含四组配对条件的 REG probe suite。
- 基于轨迹、环境状态和真实动作结果的 Judge。
- 可复用的预检、重判、ASR 汇总和 REG 分析工具。

## 数据概览

| 项目 | 数量或范围 |
|---|---:|
| 可执行 case | 1,661 |
| 干预策略 | 15 种（用户输入 6、平台状态 5、工具/数据 4） |
| REG probe 条件 | 4 组 |
| 服务世界 | Banking、Browser、Gmail、External Files、Workspace |

每条发布 case 均可在 [DATA_PROVENANCE.csv](DATA_PROVENANCE.csv) 中找到对应的
来源记录。软件代码与 Benchmark 数据采用不同许可证，详见
[DATA_LICENSE.md](DATA_LICENSE.md)。

## IVC 分类

每条 case 都记录一条可执行的 `I -> V -> C` 路径：

- **Intervention（干预）：** 攻击内容最先从哪里进入系统，以及采用什么方式进入，例如直接用户请求、被投毒的 Workspace 文件或 Skill、被篡改的工具输出。
- **Vulnerability（脆弱性表现）：** Agent 在处理或执行干预时表现出的弱点，例如缺少验证、绕过审批、不安全的参数构造、破坏 Workspace 或输出无依据结论。
- **Constraint（安全约束）：** 由真实副作用违反的安全要求，例如凭据保密、资产保护、数据完整性、可用性、系统安全、外部动作安全或治理约束。

机器可读的正式定义见 [taxonomy/registry.json](taxonomy/registry.json)。每条 case
在 `taxonomy.intervention`、`taxonomy.vulnerability` 和 `taxonomy.constraint`
中使用同一套字段。发布数据中的短文件标识为了兼容性保持不变，但具体含义以 registry
中的人类可读名称为准。

## 完整评测流程

![REDAgentBench case 生成与评测流程](./assets/framework.png)

*论文中的完整流程图。系统先将带来源的攻击知识转换为 IVC 计划，编译并在 Sandbox
中验证可执行性、可达性和可观测性，经人工审计后冻结 case。正式评测时，Trajectory
Judge 检查消息和工具调用，State Judge 验证服务回执与持久状态变化，Hybrid Judge
综合两种证据视角给出最终结果。[矢量 PDF](./assets/framework.pdf)*

## 评测原则

REDAgentBenchmark 将攻击入口、Agent 执行和安全结果拆成可验证链路：

1. 攻击材料通过用户输入、Workspace、Skill/Memory、依赖或工具/数据通道进入任务。
2. Agent 与 Harness 在隔离环境中使用真实工具完成任务。
3. Banking、Browser、Gmail、External Files 和 Workspace 服务记录真实副作用。
4. Judge 联合环境终态、服务动作和 Agent 轨迹判定攻击是否成功。
5. 汇总 ASR，并分析 Agent 的风险识别与危险执行之间是否存在 REG。

这种设计避免仅凭“模型说自己拒绝了”或“模型描述了某个动作”推断安全结果。

## 安全默认值

- 每次运行使用独立的内部 Docker 网络，默认没有公网出口。
- Agent 容器只能通过每个网络独立的 `host.docker.internal` 网关访问 Harness 明确暴露的端点。
- OpenAI 兼容代理使用每次运行随机生成的 Bearer Token；真实上游 Key 不会复制到 Agent 容器。
- MCP 控制接口使用独立的随机控制 Token。
- 服务 Fixture 使用关闭 IP masquerading 的专用 Docker bridge，不能借宿主机 NAT 访问外部网络。
- 除非显式设置 `sandbox.allow_host_network=true`，否则拒绝 Host Network。
- 诊断 Collector 默认关闭，只有显式传入 `--collector` 才会启用。

这些控制可以减少意外暴露，但不能保证在个人工作机上执行对抗性 Agent 代码绝对安全。
建议使用一次性虚拟机，避免在 Benchmark Workspace 中放入真实凭据，并先阅读
[SECURITY.md](SECURITY.md)。

## 快速开始（Codex）

环境要求：Python 3.10+、带 Compose 插件的 Docker Engine，以及足够的镜像构建空间。
以下命令均在 `DianJin-RED/` 目录中执行。

### 1. 安装 Python 包

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

### 2. 配置模型与 API Key

执行真实 rollout 需要一个兼容 OpenAI API 的模型服务。先复制私有配置模板：

```bash
cp config/config_local_private.example.json config/config_local_private.json
```

编辑 `config/config_local_private.json`，分别填写目标 Agent 和 Judge 使用的
服务地址、模型名称与 API Key。最小复现可以让二者共用同一个服务和 Key；大规模
实验也可以使用相互独立的凭据：

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

使用其他 OpenAI 兼容服务时，只需修改 `base_url`、`model` 和 `api_keys`。
私有配置已经被 `.gitignore` 排除；写入真实 Key 前可用下面的命令确认：

```bash
git check-ignore config/config_local_private.json
```

应保持 `agent_proxy.enabled=true`。在默认配置下，真实上游 Key 只保存在宿主机的
代理进程中，Agent 容器拿到的是每次运行随机生成的代理 Token。

### 3. 构建并检查隔离运行环境

```bash

# 默认只构建公开可复现的 Sandbox Base 与 Codex Runtime。
bash docker/build-agent-images.sh

# 检查必要资源、镜像、代理鉴权和网络隔离。
python scripts/quickstart_check.py --network-smoke

# 运行单元测试与契约测试。
pytest -q
```

默认构建仅下载公开的 Ubuntu、Node.js 和 Codex 包，不依赖私有压缩包或未发布的
基础镜像。如果当前网络无法访问 Docker Hub 或 npmjs，可以通过
`SANDBOX_BASE_IMAGE=...` 和 `NPM_REGISTRY=...` 选择可信镜像源。Gmail Fixture
同样支持通过 `MAILPIT_IMAGE=...` 选择镜像，默认固定为 `v1.30.0`。

### 4. 运行第一条 case

下面的命令使用一个 Worker 执行第一条 Workspace 文件干预 case：

```bash
python -m red_agent_world.runners.codex_sandbox_runner \
  --config config/config_local_private.json \
  --dataset-file test/E1.json \
  --limit 1 \
  --concurrency 1 \
  --output-name quickstart_codex
```

运行产物写入 Git 忽略的 `results/`。应联合检查结果 CSV、导出的 Trajectory、
服务回执和环境状态差异。确认单 case 成功后，再增加 `--limit` 和 `--concurrency`。

示例中的 `agent_proxy.bind_host` 为 `0.0.0.0`，用于让隔离 Docker 网络访问代理；
代理端点由每次运行生成且不会导出的随机 Token 保护。禁止把真实 API Key 写进
Benchmark case、Sandbox seed、Git 配置、Shell 历史或结果文件；如果 Key 曾被提交，
必须立即轮换。

## Runtime 镜像

| Target | 公开构建状态 | 上游或许可说明 |
|---|---|---|
| `base`、`codex` | 默认构建并已测试 | Codex：Apache-2.0 |
| `openclaw` | 可从固定版本的公开 npm 包构建 | OpenClaw：MIT |
| `hermes` | 可从固定公开 Git revision 构建 | Hermes Agent：MIT |
| `claudecode` | 可从官方 npm 包构建；本版本未完成 Benchmark 验证 | 遵循 Anthropic 条款 |
| `workbuddy` | 可从公开 npm 包构建；本版本未完成 Benchmark 验证 | 遵循供应商条款 |

运行 `bash docker/build-agent-images.sh --help` 查看构建 Target。完整第三方声明见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 目录结构

```text
config/            Runtime 配置模板
docker/            Sandbox 与 Agent Runtime 镜像
prompts/           当前 Judge Prompt
reg_probe_suite/   配对 REG Probe
sandbox/           服务世界与 MCP Server
scripts/           预检、重判、汇总与分析工具
src/red_agent_world/
taxonomy/          IVC 的干预、脆弱性与安全约束注册表
test/              Benchmark case 与逐条来源记录
tests/             单元测试与契约测试
```

## 数据许可与使用边界

Case 中包含合成凭据和对抗性指令，只能用于评测自己拥有或获准测试的系统。
逐条来源记录保存在 `DATA_PROVENANCE.csv`，数据许可见
[DATA_LICENSE.md](DATA_LICENSE.md)。

原创软件采用 [MIT License](LICENSE)。MIT 许可不覆盖 Benchmark 数据、外部镜像、
CLI 或其他第三方材料；相关边界见 [DATA_LICENSE.md](DATA_LICENSE.md) 和
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
