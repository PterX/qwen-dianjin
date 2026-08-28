from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = REPO_ROOT.parent
CONFIG_DIR = REPO_ROOT / "config"
SANDBOX_DIR = REPO_ROOT / "sandbox"
WORLDS_DIR = SANDBOX_DIR / "worlds"
SHARDS_DIR = REPO_ROOT / "shards"
RUNS_DIR = REPO_ROOT / "runs"
RESULTS_DIR = REPO_ROOT / "results"

BRAVEGUARD_ROOT = WORKSPACE_ROOT / "BraveGuard"
DEFAULT_PRIVATE_CONFIG = CONFIG_DIR / "config_local_private.json"
DEFAULT_OPENCLAW_CONFIG = CONFIG_DIR / "openclaw.json"
