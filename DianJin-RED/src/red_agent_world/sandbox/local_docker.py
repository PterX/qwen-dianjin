"""Local Docker client for OpenClaw sandbox containers."""

import asyncio
import json
import logging
import os
import shlex
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import aiohttp
import docker
from docker.models.containers import Container


class LocalDockerSandboxError(Exception):
    pass


class LocalDockerSandbox:
    def __init__(
        self,
        local_config_path: str = "./openclaw.json",
        image: str = "red-sandbox-base:v1",
        timeout: int = 60,
        max_retries: int = 60,
        retry_interval: int = 2,
        container_api_port: int = 9000,
        init_mode: str = "exec",
        network_mode: str = "isolated",
        volumes: Optional[Dict[str, Any]] = None,
        log_level: int = logging.INFO,
    ):
        self.local_config_path = str(local_config_path)
        self.image = image
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_interval = retry_interval
        self.container_api_port = container_api_port
        self.network_mode = network_mode or "isolated"
        self.volumes = volumes or {}
        if init_mode not in {"exec", "openclaw"}:
            raise ValueError(f"unsupported init_mode: {init_mode}")
        self.init_mode = init_mode
        self.container: Optional[Container] = None
        self.isolated_network = None
        self.isolated_gateway: Optional[str] = None
        self.container_id: Optional[str] = None
        self.host_port: Optional[int] = None
        self.http_session: Optional[aiohttp.ClientSession] = None
        self._initialized = False
        self.docker_client = docker.from_env()
        self.logger = self._build_logger(log_level)

    def _build_logger(self, log_level: int) -> logging.Logger:
        logger = logging.getLogger(f"{self.__class__.__name__}_{id(self)}")
        logger.setLevel(log_level)
        if not logger.handlers:
            log_file = f"/tmp/local_docker_sandbox_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{uuid.uuid4().hex[:8]}.log"
            formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s")
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.addHandler(stream_handler)
            logger.propagate = False
        return logger

    async def __aenter__(self):
        await self._ensure_http_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
        if self.http_session:
            await self.http_session.close()
            self.http_session = None

    async def _ensure_http_session(self):
        if self.http_session is None:
            self.http_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))

    def _find_available_port(self, start: int = 9001, end: int = 9100) -> int:
        import socket

        for port in range(start, end):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.bind(("", port))
                    return port
            except OSError:
                continue
        raise LocalDockerSandboxError(f"no available port in range {start}-{end}")

    async def start_container(self) -> str:
        try:
            host_network = self.network_mode == "host"
            effective_network_mode = self.network_mode
            extra_hosts = None
            if self.network_mode == "isolated":
                if self.isolated_network is None:
                    network_name = "red-agent-isolated-%s" % uuid.uuid4().hex[:12]
                    self.isolated_network = self.docker_client.networks.create(
                        network_name,
                        driver="bridge",
                        internal=True,
                        labels={"red-agent-world.scope": "single-runtime"},
                    )
                    self.isolated_network.reload()
                    ipam = self.isolated_network.attrs.get("IPAM", {}).get("Config", [])
                    self.isolated_gateway = str(ipam[0].get("Gateway") or "") if ipam else ""
                    if not self.isolated_gateway:
                        raise LocalDockerSandboxError("isolated network has no host gateway")
                effective_network_mode = self.isolated_network.name
                # An internal Docker network has no route to the daemon's default
                # bridge. Point the alias at this network's own host-side gateway,
                # which exposes only explicitly bound harness endpoints.
                extra_hosts = {"host.docker.internal": self.isolated_gateway}
            ports = {} if host_network else ({f"{self.container_api_port}/tcp": None} if self.init_mode == "openclaw" else {})
            self.container = self.docker_client.containers.run(
                self.image,
                detach=True,
                ports=ports,
                remove=False,
                auto_remove=False,
                network_mode=effective_network_mode,
                extra_hosts=extra_hosts,
                volumes=self.volumes or None,
                mem_limit="10g",
                cpu_count=8,
                pids_limit=1024,
                security_opt=["no-new-privileges:true"],
            )
            self.container.reload()
            if self.init_mode == "openclaw":
                if host_network:
                    self.host_port = int(self.container_api_port)
                else:
                    bindings = self.container.attrs.get("NetworkSettings", {}).get("Ports", {}).get(
                        f"{self.container_api_port}/tcp"
                    )
                    if not bindings:
                        raise LocalDockerSandboxError("docker did not publish a host port")
                    self.host_port = int(bindings[0]["HostPort"])
            self.container_id = self.container.id[:12]
            return self.container_id
        except Exception as exc:
            raise LocalDockerSandboxError(f"failed to start container: {exc}") from exc

    async def wait_for_container_ready(self) -> bool:
        if not self.container:
            raise LocalDockerSandboxError("container not started")
        for _attempt in range(1, self.max_retries + 1):
            try:
                self.container.reload()
                if self.container.status == "running":
                    return True
                if self.container.status in {"exited", "dead"}:
                    return False
            except Exception as exc:
                self.logger.warning("container status check failed: %s", exc)
            await asyncio.sleep(self.retry_interval)
        return False

    async def execute_command(self, command: str) -> dict:
        if not self.container:
            raise LocalDockerSandboxError("container not started")
        try:
            exit_code, output = await asyncio.to_thread(
                self.container.exec_run,
                cmd=["bash", "-c", command],
                demux=True,
            )
            stdout = output[0].decode("utf-8") if output and output[0] else ""
            stderr = output[1].decode("utf-8") if output and output[1] else ""
            return {"result": {"stdout": stdout, "stderr": stderr, "exit_code": exit_code}}
        except Exception as exc:
            raise LocalDockerSandboxError(f"command failed: {exc}") from exc

    async def upload_file(self, local_file_path: str, target_path: str) -> dict:
        if not self.container:
            raise LocalDockerSandboxError("container not started")
        file_path = Path(local_file_path)
        if not file_path.exists():
            raise LocalDockerSandboxError(f"local file not found: {local_file_path}")
        try:
            import io
            import tarfile

            tar_stream = io.BytesIO()
            with tarfile.open(fileobj=tar_stream, mode="w") as tar:
                tar.add(str(file_path), arcname=Path(target_path).name)
            tar_stream.seek(0)
            self.container.put_archive(str(Path(target_path).parent), tar_stream)
            return {"success": True}
        except Exception as exc:
            raise LocalDockerSandboxError(f"upload failed: {exc}") from exc

    async def stop_container(self) -> bool:
        if not self.container:
            return True
        try:
            self.container.stop(timeout=10)
            self.container.remove(force=True)
            return True
        except Exception as exc:
            self.logger.warning("container stop failed: %s", exc)
            return False

    async def restore_bind_mount_ownership(self) -> None:
        if not self.container or not self.volumes:
            return
        uid = os.getuid()
        gid = os.getgid()
        binds = []
        for spec in self.volumes.values():
            if isinstance(spec, dict) and spec.get("bind"):
                binds.append(str(spec["bind"]))
        if not binds:
            return
        commands = []
        for bind_path in sorted(set(binds)):
            quoted = shlex.quote(bind_path)
            commands.append("chown -R %s:%s %s 2>/dev/null || true" % (uid, gid, quoted))
            commands.append("chmod -R u+rwX %s 2>/dev/null || true" % quoted)
        try:
            await self.execute_command("; ".join(commands))
        except Exception as exc:
            self.logger.warning("bind mount ownership restore failed: %s", exc)

    async def _call_container_api(
        self,
        endpoint: str,
        method: str = "GET",
        json_data: Optional[dict] = None,
        timeout: int = 60,
    ) -> dict:
        await self._ensure_http_session()
        url = f"http://localhost:{self.host_port}{endpoint}"
        try:
            if method.upper() == "GET":
                async with self.http_session.get(url, timeout=timeout) as response:
                    text = await response.text()
                    if response.status != 200:
                        raise LocalDockerSandboxError(f"GET {endpoint} failed: {response.status} - {text}")
                    return json.loads(text)
            if method.upper() == "POST":
                async with self.http_session.post(
                    url,
                    json=json_data,
                    headers={"Content-Type": "application/json"},
                    timeout=timeout,
                ) as response:
                    text = await response.text()
                    if response.status != 200:
                        raise LocalDockerSandboxError(f"POST {endpoint} failed: {response.status} - {text}")
                    return json.loads(text)
            if method.upper() == "DELETE":
                async with self.http_session.delete(url, timeout=timeout) as response:
                    text = await response.text()
                    if response.status != 200:
                        raise LocalDockerSandboxError(f"DELETE {endpoint} failed: {response.status} - {text}")
                    return json.loads(text)
            raise LocalDockerSandboxError(f"unsupported HTTP method: {method}")
        except asyncio.TimeoutError as exc:
            raise LocalDockerSandboxError(f"API timeout: {endpoint}") from exc
        except json.JSONDecodeError as exc:
            raise LocalDockerSandboxError(f"API returned non-JSON: {endpoint}, error={exc}") from exc

    async def _call_container_api_text(
        self,
        endpoint: str,
        method: str = "GET",
        json_data: Optional[dict] = None,
        timeout: int = 60,
    ) -> str:
        await self._ensure_http_session()
        url = f"http://localhost:{self.host_port}{endpoint}"
        if method.upper() == "GET":
            async with self.http_session.get(url, timeout=timeout) as response:
                text = await response.text()
                if response.status != 200:
                    raise LocalDockerSandboxError(f"GET {endpoint} failed: {response.status} - {text}")
                return text
        if method.upper() == "POST":
            async with self.http_session.post(url, json=json_data, timeout=timeout) as response:
                text = await response.text()
                if response.status != 200:
                    raise LocalDockerSandboxError(f"POST {endpoint} failed: {response.status} - {text}")
                return text
        raise LocalDockerSandboxError(f"unsupported HTTP method: {method}")

    async def _wait_fastapi_up(self):
        data = await self._call_container_api("/health", "GET", timeout=10)
        if "status" not in data:
            raise LocalDockerSandboxError(f"unexpected /health response: {data}")
        return data

    async def _wait_service_healthy(self):
        data = await self._call_container_api("/health", "GET", timeout=10)
        if data.get("status") != "healthy":
            raise LocalDockerSandboxError(f"service not healthy: {data}")
        return data

    async def _retry(self, func, action_name: str, retries: Optional[int] = None, interval: Optional[int] = None) -> Any:
        retries = retries if retries is not None else self.max_retries
        interval = interval if interval is not None else self.retry_interval
        last_error = None
        for attempt in range(1, retries + 1):
            try:
                return await func()
            except Exception as exc:
                last_error = exc
                self.logger.warning("%s failed on attempt %d/%d: %s", action_name, attempt, retries, exc)
                if attempt < retries:
                    await asyncio.sleep(interval)
        raise LocalDockerSandboxError(f"{action_name} failed: {last_error}")

    async def initialize(self):
        await self._ensure_http_session()
        if self._initialized:
            return
        await self._retry(self.start_container, "start container")
        if not await self.wait_for_container_ready():
            raise LocalDockerSandboxError("container did not start")
        if self.init_mode == "exec":
            await self._retry(lambda: self.execute_command("mkdir -p /workspace"), "prepare exec workspace", retries=3, interval=1)
            self._initialized = True
            return
        await self._retry(self._wait_fastapi_up, "wait FastAPI", retries=30, interval=3)
        await self._retry(lambda: self.execute_command("mkdir -p /root/.openclaw"), "prepare config dir", retries=3, interval=1)
        await self._retry(lambda: self.upload_file(self.local_config_path, "/root/.openclaw/openclaw.json"), "upload openclaw config", retries=5, interval=2)
        await self._retry(lambda: self.execute_command("ls -l /root/.openclaw/openclaw.json"), "verify openclaw config", retries=3, interval=1)
        await self._retry(lambda: self._call_container_api("/reload-config", "POST", timeout=30), "reload OpenClaw config", retries=5, interval=3)
        await self._retry(self._wait_service_healthy, "wait OpenClaw healthy", retries=10, interval=2)
        self._initialized = True

    async def send_message(self, message: str, session_id: Optional[str] = None, timeout: int = 600) -> dict:
        if not self._initialized:
            raise LocalDockerSandboxError("initialize() must be called first")
        message = (message or "").strip()
        if not message:
            raise LocalDockerSandboxError("message cannot be empty")
        payload = {"prompt": message, "timeout": timeout}
        if session_id:
            payload["session_id"] = session_id
        result = await self._retry(
            lambda: self._call_container_api("/prompt", "POST", payload, timeout=timeout + 30),
            "send OpenClaw prompt",
            retries=3,
            interval=2,
        )
        raw_response = result.get("response")
        if isinstance(raw_response, str):
            try:
                result["response_json"] = json.loads(raw_response)
            except Exception:
                result["response_json"] = None
        return result

    async def export_session_jsonl(self, session_id: str, local_dir: str = "./exports_local") -> str:
        if not self._initialized:
            raise LocalDockerSandboxError("initialize() must be called first")
        if not session_id:
            raise LocalDockerSandboxError("session_id cannot be empty")
        output_dir = Path(local_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        file_content = await self._retry(
            lambda: self._call_container_api_text(f"/sessions/{session_id}/export", "GET", timeout=60),
            f"export session {session_id}",
            retries=3,
            interval=2,
        )
        local_path = output_dir / f"session_{session_id}.jsonl"
        local_path.write_text(file_content, encoding="utf-8")
        return str(local_path)

    async def stop(self):
        if self.container_id:
            await self.restore_bind_mount_ownership()
            await self._retry(self.stop_container, "stop container", retries=3, interval=2)
        self.container = None
        self.container_id = None
        self.host_port = None
        self._initialized = False
        if self.isolated_network is not None:
            try:
                await asyncio.to_thread(self.isolated_network.remove)
            except Exception as exc:
                self.logger.warning("isolated network cleanup failed: %s", exc)
            self.isolated_network = None
            self.isolated_gateway = None
