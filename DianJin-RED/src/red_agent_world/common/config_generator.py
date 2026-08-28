"""Generate per-case OpenClaw config files from a local template."""

import json
import logging
import secrets
from pathlib import Path
from typing import Any, Dict


logger = logging.getLogger(__name__)


def openclaw_provider_profile(base_url: str, model: str) -> Dict[str, Any]:
    """Return light provider compatibility hints for OpenAI-compatible APIs."""
    haystack = "%s %s" % (base_url or "", model or "")
    haystack = haystack.lower()
    if any(marker in haystack for marker in ("dashscope", "aliyuncs", "qwen")):
        return {
            "family": "qwen",
            "max_tokens_field": "max_tokens",
            "context_window": 131072,
            "max_tokens": 8192,
            "reasoning": False,
        }
    if any(marker in haystack for marker in ("moonshot", "kimi")):
        return {
            "family": "kimi",
            "max_tokens_field": "max_tokens",
            "context_window": 131072,
            "max_tokens": 8192,
            "reasoning": False,
        }
    return {
        "family": "openai-compatible",
        "max_tokens_field": "max_completion_tokens",
        "context_window": 400000,
        "max_tokens": 128000,
        "reasoning": False,
    }


class ConfigGenerator:
    def __init__(self, template_path: str):
        self.template_path = Path(template_path)
        if not self.template_path.exists():
            raise FileNotFoundError(f"OpenClaw config template not found: {template_path}")
        self.template = json.loads(self.template_path.read_text(encoding="utf-8"))
        logger.info("loaded OpenClaw config template: %s", self.template_path)

    def generate_config(
        self,
        api_key: str,
        base_url: str,
        model: str,
        output_path: str,
        max_tokens: int | None = None,
    ) -> str:
        config = json.loads(json.dumps(self.template))
        profile = openclaw_provider_profile(base_url, model)
        effective_max_tokens = int(max_tokens or profile["max_tokens"])

        providers = config.setdefault("models", {}).setdefault("providers", {})
        provider_name = list(providers.keys())[0] if providers else "DefaultProvider"
        provider_config = providers.setdefault(provider_name, {})
        provider_config["baseUrl"] = base_url
        provider_config["apiKey"] = api_key
        provider_config["api"] = "openai-completions"
        provider_config["authHeader"] = True

        models = provider_config.setdefault("models", [])
        if not models:
            models.append({})
        models[0].update(
            {
                "id": model,
                "name": model,
                "api": "openai-completions",
                "reasoning": bool(profile.get("reasoning", False)),
                "input": ["text"],
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                "contextWindow": int(profile["context_window"]),
                "maxTokens": effective_max_tokens,
                "compat": {"maxTokensField": profile["max_tokens_field"]},
            }
        )

        config.setdefault("agents", {}).setdefault("defaults", {}).setdefault("model", {})[
            "primary"
        ] = f"{provider_name}/{model}"
        config.setdefault("agents", {}).setdefault("defaults", {})["workspace"] = "/workspace"

        # The template contains only a non-secret placeholder. Every generated
        # runtime receives an independent token and a loopback-only gateway.
        gateway = config.setdefault("gateway", {})
        gateway["bind"] = "loopback"
        gateway_token = secrets.token_urlsafe(32)
        gateway.setdefault("auth", {})["mode"] = "token"
        gateway["auth"]["token"] = gateway_token
        gateway.setdefault("remote", {})["token"] = gateway_token

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(output.resolve())
