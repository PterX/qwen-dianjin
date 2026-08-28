"""Small local task runner base used by RED-Agent-World rollout adapters."""

import asyncio
import csv
import json
import logging
import os
import secrets
from pathlib import Path
from typing import Any, Dict, List, Optional

from red_agent_world.common.config_generator import ConfigGenerator
from red_agent_world.common.key_pool import KeyPool
from red_agent_world.common.openai_proxy import OpenAIProxyServer


logger = logging.getLogger(__name__)


class LocalTaskRunner:
    DATASET_CHOICES = {}

    def __init__(
        self,
        config_path: str,
        dataset: Optional[str] = None,
        limit: Optional[int] = None,
        ids: Optional[List[int]] = None,
    ):
        self.config_path = Path(config_path).resolve()
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))

        if dataset and dataset in self.DATASET_CHOICES:
            dataset_info = self.DATASET_CHOICES[dataset]
            dataset_path = Path(dataset_info["path"])
            if not dataset_path.is_absolute():
                dataset_path = self.config_path.parent.parent / dataset_path
            self.dataset_name = dataset
        else:
            dataset_path = Path(self.config["paths"]["dataset"])
            if not dataset_path.is_absolute():
                dataset_path = self.config_path.parent.parent / dataset_path
            self.dataset_name = dataset_path.stem

        self.config["paths"]["dataset"] = str(dataset_path)
        self.key_pool = KeyPool(self.config["agent"]["api_keys"])
        judge_keys = self.config.get("judge", {}).get("api_keys", self.config["agent"]["api_keys"])
        self.judge_key_pool = KeyPool(judge_keys)

        openclaw_config = Path(self.config["paths"]["openclaw_config"])
        if not openclaw_config.is_absolute():
            openclaw_config = self.config_path.parent.parent / openclaw_config
        self.config_generator = ConfigGenerator(str(openclaw_config))

        self.dataset: List[Dict[str, Any]] = json.loads(dataset_path.read_text(encoding="utf-8"))
        if ids:
            wanted = set(ids)
            self.dataset = [item for item in self.dataset if item["id"] in wanted]
            missing = wanted - {item["id"] for item in self.dataset}
            if missing:
                raise ValueError(f"requested sample IDs are missing from dataset: {sorted(missing)}")
        if limit is not None:
            self.dataset = self.dataset[:limit]

        self.completed_ids = self._load_completed_ids()
        self.semaphore = asyncio.Semaphore(self.config["execution"]["concurrency"])
        self.results_lock = asyncio.Lock()
        self.agent_proxy: Optional[OpenAIProxyServer] = None
        self.agent_proxy_base_url: Optional[str] = None
        # This token is generated for every runner process and is never written
        # to the repository. It prevents other host/LAN clients from borrowing
        # the proxy's real upstream credential.
        self.agent_proxy_api_key = secrets.token_urlsafe(32)
        self.system_prompt_prefix = ""
        self.system_prompt_prefix_path = ""
        self._start_agent_proxy_if_enabled()

    def _start_agent_proxy_if_enabled(self) -> None:
        proxy_config = self.config.get("agent_proxy", {})
        if not proxy_config.get("enabled", False):
            return
        prefix_path_value = os.environ.get("RED_AGENT_WORLD_SYSTEM_PROMPT_PREFIX_FILE", "") or proxy_config.get(
            "system_prompt_prefix_file", ""
        )
        if prefix_path_value:
            prefix_path = Path(prefix_path_value).expanduser()
            if not prefix_path.is_absolute():
                prefix_path = self.config_path.parent.parent / prefix_path
            self.system_prompt_prefix_path = str(prefix_path.resolve())
            self.system_prompt_prefix = prefix_path.read_text(encoding="utf-8").strip()
        else:
            self.system_prompt_prefix = str(
                os.environ.get("RED_AGENT_WORLD_SYSTEM_PROMPT_PREFIX", "")
                or proxy_config.get("system_prompt_prefix", "")
            ).strip()
        self.agent_proxy = OpenAIProxyServer(
            host=proxy_config.get("bind_host", "0.0.0.0"),
            port=proxy_config.get("bind_port", 0),
            upstream_base_url=self.config["agent"]["base_url"],
            upstream_api_key=self.config["agent"]["api_keys"][0],
            client_api_key=self.agent_proxy_api_key,
            model_name=self.config["agent"].get("model", ""),
            system_prompt_prefix=self.system_prompt_prefix,
            upstream_reasoning_effort=proxy_config.get("upstream_reasoning_effort"),
        )
        self.agent_proxy.start()
        _, port = self.agent_proxy.server_address
        container_host = proxy_config.get("container_host", "172.17.0.1")
        self.agent_proxy_base_url = f"http://{container_host}:{port}"
        logger.info("Agent API proxy started: %s", self.agent_proxy_base_url)

    @property
    def system_prompt_experiment_enabled(self) -> bool:
        return bool(self.system_prompt_prefix and self.agent_proxy_base_url)

    def proxied_agent_endpoint(self) -> tuple[str, str]:
        if self.agent_proxy_base_url:
            return str(self.agent_proxy_base_url), self.agent_proxy_api_key
        return str(self.config["agent"]["base_url"]), ""

    def _load_completed_ids(self) -> set:
        csv_path = Path(self.config["paths"]["results_csv"])
        if not csv_path.exists():
            return set()
        completed = set()
        with csv_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("id"):
                    completed.add(int(row["id"]))
        return completed

    async def _save_result(self, result: Dict[str, Any]) -> None:
        csv_path = Path(self.config["paths"]["results_csv"])
        async with self.results_lock:
            file_exists = csv_path.exists()
            with csv_path.open("a", newline="", encoding="utf-8") as f:
                fieldnames = ["id", "sandbox_status", "sandbox_error_type", "target", "category", "jailbreak_method", "error"]
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                if not file_exists:
                    writer.writeheader()
                writer.writerow(result)

    async def _run_single_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    async def _process_item_with_semaphore(self, item: Dict[str, Any]) -> None:
        async with self.semaphore:
            try:
                result = await self._run_single_item(item)
            except Exception as exc:
                item_id = item.get("id", "")
                taxonomy = item.get("taxonomy") if isinstance(item.get("taxonomy"), dict) else {}
                logger.exception("item %s failed outside runner error handling", item_id)
                result = {
                    "id": item_id,
                    "sandbox_status": "RUNNER_ERROR",
                    "sandbox_error_type": "UNHANDLED_RUNNER_EXCEPTION",
                    "reason": "Unhandled runner exception before an item result could be produced.",
                    "target": "\n".join(str(query) for query in item.get("decomposed_query") or []),
                    "category": taxonomy.get("vulnerability", ""),
                    "jailbreak_method": taxonomy.get("intervention", ""),
                    "error": str(exc)[:1000],
                }
            await self._save_result(result)

    async def run_all(self) -> None:
        pending_items = [item for item in self.dataset if item["id"] not in self.completed_ids]
        logger.info("total=%d completed=%d pending=%d concurrency=%d", len(self.dataset), len(self.completed_ids), len(pending_items), self.config["execution"]["concurrency"])
        try:
            await asyncio.gather(
                *(self._process_item_with_semaphore(item) for item in pending_items),
            )
        finally:
            if self.agent_proxy is not None:
                self.agent_proxy.stop()
