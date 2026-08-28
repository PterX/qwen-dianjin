"""Minimal local OpenAI-compatible proxy to keep real API keys out of containers."""

import json
import hmac
import os
import time
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
from urllib import error, request


class ProxyHandler(BaseHTTPRequestHandler):
    upstream_base_url = ""
    upstream_api_key = ""
    client_api_key = ""
    model_name = ""
    system_prompt_prefix = ""
    upstream_enable_thinking = None
    upstream_reasoning_effort = None
    debug_log_path = ""
    debug_log_lock = threading.Lock()
    response_history = {}
    response_history_lock = threading.Lock()
    thought_signature_cache = {}
    thought_signature_cache_lock = threading.Lock()

    def _authorized(self) -> bool:
        """Require the per-run proxy token before using the upstream key."""
        expected = str(self.client_api_key or "")
        supplied = str(self.headers.get("Authorization") or "")
        if supplied.lower().startswith("bearer "):
            supplied = supplied[7:].strip()
        return bool(expected) and hmac.compare_digest(supplied, expected)

    def _require_authorization(self) -> bool:
        if self._authorized():
            return True
        self._send_json(
            401,
            {"error": {"message": "missing or invalid local proxy token", "type": "authentication_error"}},
        )
        return False

    def do_GET(self):
        if not self._require_authorization():
            return
        parsed = urlparse(self.path)
        if parsed.path.rstrip("/") == "/models":
            model = self.model_name or "red-agent-world-model"
            model_record = {
                "id": model,
                "slug": model,
                "name": model,
                "display_name": model,
                "description": "Proxied model endpoint.",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "red-agent-world",
                "prefer_websockets": True,
                "minimal_client_version": "0.124.0",
                "supported_in_api": True,
                "availability_nux": {"message": ""},
                "priority": 0,
                "reasoning_summary_format": "experimental",
                "default_reasoning_level": "medium",
                "supported_reasoning_levels": [
                    {"effort": "low", "description": "Fast responses with lighter reasoning"},
                    {"effort": "medium", "description": "Balanced reasoning for most tasks"},
                    {"effort": "high", "description": "More reasoning for complex tasks"},
                ],
                "shell_type": "shell_command",
                "visibility": "list",
                "additional_speed_tiers": [],
                "service_tiers": [],
                "default_service_tier": None,
                "upgrade": None,
                "base_instructions": "",
                "model_messages": {
                    "instructions_template": "",
                    "instructions_variables": {
                        "personality_default": "",
                        "personality_friendly": "",
                        "personality_pragmatic": "",
                    },
                },
                "supports_reasoning_summaries": True,
                "default_reasoning_summary": "none",
                "support_verbosity": True,
                "default_verbosity": "low",
                "input_modalities": ["text", "image"],
                "apply_patch_tool_type": "freeform",
                "web_search_tool_type": "text_and_image",
                "truncation_policy": {"mode": "tokens", "limit": 10000},
                "supports_parallel_tool_calls": True,
                "supports_image_detail_original": True,
                "context_window": 400000,
                "max_context_window": 400000,
                "auto_compact_token_limit": None,
                "comp_hash": None,
                "experimental_supported_tools": [],
                "supports_search_tool": True,
                "use_responses_lite": False,
                "auto_review_model_override": None,
                "tool_mode": "direct",
                "multi_agent_version": "v2",
                "available_in_plans": [],
            }
            payload = {
                "object": "list",
                "data": [model_record],
                "models": [model_record],
            }
            self._send_json(200, payload)
            return
        self.send_error(404, "Not Found")

    def do_POST(self):
        if not self._require_authorization():
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        if self.path.rstrip("/") == "/responses":
            self._maybe_log_request_summary(body)
            self._handle_responses(body)
            return
        is_chat_completion = self.path.rstrip("/").endswith("/chat/completions")
        if is_chat_completion:
            body = self._inject_prefix_into_chat_body(body)
            body = self._restore_thought_signatures_in_chat_body(body)
        self._maybe_log_request_summary(body)
        target_url = f"{self.upstream_base_url.rstrip('/')}{self.path}"
        upstream_request = request.Request(
            target_url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.upstream_api_key}",
                "Content-Type": self.headers.get("Content-Type", "application/json"),
            },
            method="POST",
        )
        try:
            with request.urlopen(upstream_request, timeout=300) as response:
                response_body = response.read()
                if is_chat_completion:
                    response_body = self._prepare_chat_response_body(response_body)
                self._maybe_log_response_summary(response.status, response_body)
                self.send_response(response.status)
                self._copy_response_headers(response.headers, content_length=len(response_body))
                self.end_headers()
                self.wfile.write(response_body)
        except error.HTTPError as exc:
            response_body = exc.read()
            self._maybe_log_response_summary(exc.code, response_body)
            self.send_response(exc.code)
            self._copy_response_headers(exc.headers, content_length=len(response_body))
            self.end_headers()
            self.wfile.write(response_body)
        except Exception as exc:
            payload = json.dumps({"error": str(exc)}).encode("utf-8")
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    def _handle_responses(self, body: bytes) -> None:
        try:
            responses_payload = json.loads(body.decode("utf-8"))
        except Exception as exc:
            self._send_json(400, {"error": {"message": f"invalid JSON: {exc}"}})
            return
        chat_payload = self._responses_to_chat_payload(responses_payload)
        try:
            chat_response = self._post_chat_completion(chat_payload)
        except error.HTTPError as exc:
            response_body = exc.read()
            self._maybe_log_response_summary(exc.code, response_body)
            self._send_bytes(exc.code, response_body, exc.headers.get("Content-Type", "application/json"))
            return
        except Exception as exc:
            self._send_json(502, {"error": str(exc)})
            return

        responses_response = self._chat_to_responses_payload(responses_payload, chat_response)
        self._remember_response(responses_response.get("id"), chat_payload, chat_response)
        if responses_payload.get("stream"):

            self._send_responses_sse(responses_response)
        else:

            self._send_json(200, responses_response)
    def _remember_response(self, response_id, chat_payload, chat_response):
        if not response_id:
            return
        messages = [dict(message) for message in chat_payload.get("messages") or []]
        choice = (chat_response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        assistant = {"role": "assistant", "content": message.get("content") or ""}
        if message.get("tool_calls"):
            assistant["tool_calls"] = message["tool_calls"]
        messages.append(assistant)
        with self.response_history_lock:
            self.response_history[str(response_id)] = messages
            while len(self.response_history) > 256:
                self.response_history.pop(next(iter(self.response_history)))

    def _previous_response_messages(self, response_id):
        if not response_id:
            return []
        with self.response_history_lock:
            messages = self.response_history.get(str(response_id)) or []
            return [dict(message) for message in messages]


    def _with_qwen_cache_control(self, chat_payload):
        model = str(chat_payload.get("model") or self.model_name or "").lower()
        upstream = str(self.upstream_base_url or "").lower()
        if "qwen" not in model and "dashscope" not in upstream and "aliyuncs" not in upstream:
            return chat_payload

        messages = chat_payload.get("messages") or []
        target_index = next(
            (
                index
                for index, message in enumerate(messages)
                if isinstance(message, dict)
                and message.get("role") in {"system", "developer"}
                and message.get("content")
            ),
            None,
        )
        if target_index is None:
            target_index = next(
                (
                    index
                    for index, message in enumerate(messages)
                    if isinstance(message, dict)
                    and message.get("role") == "user"
                    and message.get("content")
                ),
                None,
            )
        if target_index is None:
            return chat_payload

        cached_payload = dict(chat_payload)
        cached_messages = [dict(message) if isinstance(message, dict) else message for message in messages]
        target_message = cached_messages[target_index]
        content = target_message.get("content")
        if isinstance(content, str):
            target_message["content"] = [
                {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
            ]
        elif isinstance(content, list):
            content_blocks = [dict(block) if isinstance(block, dict) else block for block in content]
            text_block = next(
                (block for block in reversed(content_blocks) if isinstance(block, dict) and block.get("type") == "text"),
                None,
            )
            if text_block is None:
                return chat_payload
            text_block.setdefault("cache_control", {"type": "ephemeral"})
            target_message["content"] = content_blocks
        else:
            return chat_payload
        cached_payload["messages"] = cached_messages
        return cached_payload

    @staticmethod
    def _tool_call_identity(tool_call):
        function = tool_call.get("function") if isinstance(tool_call, dict) else {}
        function = function if isinstance(function, dict) else {}
        arguments = function.get("arguments", "{}")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        else:
            try:
                arguments = json.dumps(
                    json.loads(arguments),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            except Exception:
                pass
        return str(function.get("name") or ""), arguments

    def _normalize_and_remember_chat_tool_calls(self, payload):
        """Give provider tool calls stable IDs and retain opaque Gemini signatures."""
        changed = False
        remembered = []
        for choice in payload.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if not isinstance(message, dict):
                continue
            for tool_call in message.get("tool_calls") or []:
                if not isinstance(tool_call, dict):
                    continue
                call_id = tool_call.get("id")
                if not call_id:
                    call_id = "call_%s" % uuid.uuid4().hex
                    tool_call["id"] = call_id
                    changed = True
                top_level = {
                    key: tool_call[key]
                    for key in ("thoughtSignature", "thought_signature")
                    if tool_call.get(key)
                }
                function = tool_call.get("function")
                nested = (
                    {
                        key: function[key]
                        for key in ("thoughtSignature", "thought_signature")
                        if function.get(key)
                    }
                    if isinstance(function, dict)
                    else {}
                )
                if top_level or nested:
                    remembered.append(
                        (
                            str(call_id),
                            {
                                "identity": self._tool_call_identity(tool_call),
                                "top_level": top_level,
                                "function": nested,
                            },
                        )
                    )
        if remembered:
            with self.thought_signature_cache_lock:
                for call_id, record in remembered:
                    self.thought_signature_cache[call_id] = record
                while len(self.thought_signature_cache) > 4096:
                    self.thought_signature_cache.pop(next(iter(self.thought_signature_cache)))
        return changed

    def _restore_thought_signatures(self, chat_payload):
        messages = chat_payload.get("messages")
        if not isinstance(messages, list):
            return chat_payload
        restored_payload = None
        restored_messages = None
        for message_index, message in enumerate(messages):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for call_index, tool_call in enumerate(tool_calls):
                if not isinstance(tool_call, dict):
                    continue
                call_id = str(tool_call.get("id") or "")
                if not call_id:
                    continue
                with self.thought_signature_cache_lock:
                    record = self.thought_signature_cache.get(call_id)
                if not record or record.get("identity") != self._tool_call_identity(tool_call):
                    continue
                top_level = record.get("top_level") or {}
                nested = record.get("function") or {}
                missing_top = {
                    key: value
                    for key, value in top_level.items()
                    if not tool_call.get(key)
                }
                function = tool_call.get("function")
                missing_nested = (
                    {
                        key: value
                        for key, value in nested.items()
                        if not function.get(key)
                    }
                    if isinstance(function, dict)
                    else {}
                )
                if not missing_top and not missing_nested:
                    continue
                if restored_payload is None:
                    restored_payload = dict(chat_payload)
                    restored_messages = [
                        dict(item) if isinstance(item, dict) else item
                        for item in messages
                    ]
                    restored_payload["messages"] = restored_messages
                restored_message = restored_messages[message_index]
                restored_calls = restored_message.get("tool_calls")
                if restored_calls is tool_calls:
                    restored_calls = [
                        dict(item) if isinstance(item, dict) else item
                        for item in tool_calls
                    ]
                    restored_message["tool_calls"] = restored_calls
                restored_call = restored_calls[call_index]
                restored_call.update(missing_top)
                if missing_nested:
                    restored_function = dict(restored_call.get("function") or {})
                    restored_function.update(missing_nested)
                    restored_call["function"] = restored_function
        return restored_payload or chat_payload

    def _restore_thought_signatures_in_chat_body(self, body):
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            return body
        if not isinstance(payload, dict):
            return body
        restored = self._restore_thought_signatures(payload)
        if restored is payload:
            return body
        return json.dumps(restored, ensure_ascii=False).encode("utf-8")

    def _prepare_chat_response_body(self, body):
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            return self._prepare_chat_sse_response_body(body)
        if not isinstance(payload, dict):
            return body
        changed = self._normalize_and_remember_chat_tool_calls(payload)
        if not changed:
            return body
        return json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def _prepare_chat_sse_response_body(self, body):
        try:
            text = body.decode("utf-8")
        except Exception:
            return body
        lines = text.splitlines(keepends=True)
        accumulated = {}
        changed = False
        for line_index, line in enumerate(lines):
            stripped = line.rstrip("\r\n")
            newline = line[len(stripped):]
            if not stripped.startswith("data:"):
                continue
            data = stripped[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                payload = json.loads(data)
            except Exception:
                continue
            line_changed = False
            for choice_index, choice in enumerate(payload.get("choices") or []):
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta")
                if not isinstance(delta, dict):
                    continue
                for position, tool_call in enumerate(delta.get("tool_calls") or []):
                    if not isinstance(tool_call, dict):
                        continue
                    tool_index = tool_call.get("index")
                    if tool_index is None:
                        tool_index = position
                    key = (choice_index, str(tool_index))
                    state = accumulated.setdefault(
                        key,
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    call_id = tool_call.get("id") or state.get("id")
                    if not call_id:
                        call_id = "call_%s" % uuid.uuid4().hex
                        tool_call["id"] = call_id
                        line_changed = True
                    state["id"] = str(call_id)
                    if tool_call.get("type"):
                        state["type"] = tool_call["type"]
                    function = tool_call.get("function")
                    if isinstance(function, dict):
                        if function.get("name"):
                            state["function"]["name"] += str(function["name"])
                        if function.get("arguments"):
                            state["function"]["arguments"] += str(function["arguments"])
                        for signature_key in ("thoughtSignature", "thought_signature"):
                            if function.get(signature_key):
                                state["function"][signature_key] = function[signature_key]
                    for signature_key in ("thoughtSignature", "thought_signature"):
                        if tool_call.get(signature_key):
                            state[signature_key] = tool_call[signature_key]
            if line_changed:
                lines[line_index] = "data: %s%s" % (
                    json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                    newline,
                )
                changed = True
        if accumulated:
            synthetic = {
                "choices": [
                    {
                        "message": {
                            "tool_calls": list(accumulated.values()),
                        }
                    }
                ]
            }
            self._normalize_and_remember_chat_tool_calls(synthetic)
        if not changed:
            return body
        return "".join(lines).encode("utf-8")

    def _post_chat_completion(self, chat_payload):
        chat_payload = self._restore_thought_signatures(chat_payload)
        upstream_payload = self._with_qwen_cache_control(chat_payload)
        if self.upstream_enable_thinking is not None:
            upstream_payload = {
                **upstream_payload,
                "enable_thinking": bool(self.upstream_enable_thinking),
            }
        if self.upstream_reasoning_effort is not None:
            upstream_payload = {
                **upstream_payload,
                "reasoning_effort": str(self.upstream_reasoning_effort),
            }
        if self.debug_log_path:
            self._append_debug_record(
                {
                    "stage": "upstream_chat_request",
                    "system_prompt_prefix_injected": any(
                        isinstance(message, dict)
                        and message.get("role") in {"system", "developer"}
                        and str(message.get("content") or "").strip().startswith(self.system_prompt_prefix.strip())
                        for message in chat_payload.get("messages") or []
                    ) if self.system_prompt_prefix else False,
                    "system_prompt_prefix_chars": len(self.system_prompt_prefix),
                    "thought_signature_count": sum(
                        bool(call.get("thoughtSignature") or call.get("thought_signature"))
                        for message in chat_payload.get("messages") or []
                        if isinstance(message, dict)
                        for call in message.get("tool_calls") or []
                        if isinstance(call, dict)
                    ),
                    "tools": [
                        {
                            "type": tool.get("type"),
                            "name": (tool.get("function") or {}).get("name"),
                        }
                        for tool in chat_payload.get("tools") or []
                        if isinstance(tool, dict)
                    ],
                    "messages": [
                        {
                            "role": message.get("role"),
                            "tool_call_ids": [call.get("id") for call in message.get("tool_calls") or []],
                            "tool_call_id": message.get("tool_call_id"),
                            "content_length": len(str(message.get("content") or "")),
                        }
                        for message in chat_payload.get("messages") or []
                        if isinstance(message, dict)
                    ],
                }
            )
        target_url = f"{self.upstream_base_url.rstrip('/')}/chat/completions"
        upstream_request = request.Request(
            target_url,
            data=json.dumps(upstream_payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.upstream_api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(upstream_request, timeout=300) as response:
            response_body = response.read()
            self._maybe_log_response_summary(response.status, response_body)
            if response.status >= 400:
                raise error.HTTPError(target_url, response.status, "upstream error", response.headers, None)
            payload = json.loads(response_body.decode("utf-8"))
            self._normalize_and_remember_chat_tool_calls(payload)
            return payload

    def _responses_to_chat_payload(self, payload):
        requested_model = str(payload.get("model") or "")
        model = (
            self.model_name
            if requested_model == "red-agent-world-native-mcp" and self.model_name
            else requested_model or self.model_name
        )
        messages = self._previous_response_messages(payload.get("previous_response_id"))
        instructions = payload.get("instructions")
        if instructions and not messages:
            messages.append({"role": "system", "content": str(instructions)})
        messages.extend(self._responses_input_to_messages(payload.get("input")))
        messages = self._order_tool_results(messages)
        messages = self._inject_prefix_into_messages(messages)
        chat_payload = {
            "model": model,
            "messages": messages or [{"role": "user", "content": ""}],
            "stream": False,
        }
        for src, dest in (
            ("temperature", "temperature"),
            ("top_p", "top_p"),
            ("max_output_tokens", "max_tokens"),
            ("max_completion_tokens", "max_completion_tokens"),
        ):
            if src in payload:
                chat_payload[dest] = payload[src]
        tools = self._responses_tools_to_chat_tools(payload.get("tools"))
        if tools:
            chat_payload["tools"] = tools
            if payload.get("tool_choice") is not None:
                chat_payload["tool_choice"] = payload["tool_choice"]
            if payload.get("parallel_tool_calls") is not None:
                chat_payload["parallel_tool_calls"] = bool(payload["parallel_tool_calls"])
        return chat_payload

    def _inject_prefix_into_chat_body(self, body):
        if not self.system_prompt_prefix:
            return body
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            return body
        if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
            return body
        payload["messages"] = self._inject_prefix_into_messages(payload["messages"])
        return json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def _inject_prefix_into_messages(self, messages):
        prefix = str(self.system_prompt_prefix or "").strip()
        copied = [dict(message) if isinstance(message, dict) else message for message in (messages or [])]
        if not prefix:
            return copied
        for index, message in enumerate(copied):
            if not isinstance(message, dict) or message.get("role") not in {"system", "developer"}:
                continue
            content = str(message.get("content") or "")
            if content.strip().startswith(prefix):
                return copied
            message["content"] = prefix + ("\n\n" + content if content else "")
            copied[index] = message
            return copied
        copied.insert(0, {"role": "system", "content": prefix})
        return copied

    @staticmethod
    def _order_tool_results(messages):
        tool_results = {}
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "tool":
                continue
            call_id = str(message.get("tool_call_id") or "")
            if call_id:
                tool_results.setdefault(call_id, []).append(message)

        ordered = []
        consumed = set()
        for message in messages:
            if not isinstance(message, dict):
                ordered.append(message)
                continue
            if message.get("role") == "tool":
                continue
            ordered.append(message)
            if message.get("role") != "assistant":
                continue
            for tool_call in message.get("tool_calls") or []:
                call_id = str(tool_call.get("id") or "")
                if not call_id:
                    continue
                ordered.extend(tool_results.get(call_id, []))
                consumed.add(call_id)

        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "tool":
                continue
            call_id = str(message.get("tool_call_id") or "")
            if not call_id or call_id not in consumed:
                ordered.append(message)
        return ordered

    def _responses_input_to_messages(self, value):
        if isinstance(value, str):
            return [{"role": "user", "content": value}]
        if not isinstance(value, list):
            return []
        messages = []
        for item in value:
            if not isinstance(item, dict):
                messages.append({"role": "user", "content": str(item)})
                continue
            item_type = item.get("type")
            role = item.get("role") or ("assistant" if item_type in {"message", "function_call"} else "user")
            if role == "developer":
                role = "system"
            content = item.get("content")
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict):
                        text_parts.append(str(part.get("text") or part.get("input_text") or part.get("output_text") or ""))
                    else:
                        text_parts.append(str(part))
                content = "\n".join(part for part in text_parts if part)
            elif content is None:
                content = item.get("text") or item.get("input_text") or item.get("output_text") or ""
            if item_type in {"function_call_output", "mcp_tool_call_output", "custom_tool_call_output"}:
                call_id = item.get("call_id") or item.get("id") or "call"
                output = item.get("output")
                if not isinstance(output, str):
                    output = json.dumps(output, ensure_ascii=False)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": output if output is not None else str(content or ""),
                    }
                )
            elif item_type == "function_call":
                call_id = item.get("call_id") or item.get("id") or "call"
                namespace = str(item.get("namespace") or "").rstrip("_")
                name = str(item.get("name") or "unknown_tool").lstrip("_")
                tool_call = {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": f"{namespace}__{name}" if namespace else name,
                        "arguments": item.get("arguments") or "{}",
                    },
                }
                if messages and messages[-1].get("role") == "assistant" and messages[-1].get("tool_calls"):
                    messages[-1]["tool_calls"].append(tool_call)
                else:
                    messages.append({"role": "assistant", "content": "", "tool_calls": [tool_call]})
            elif item_type == "message" or content:
                messages.append({"role": role, "content": str(content or "")})
        return messages

    @staticmethod
    def _namespace_tool_map(tools):
        mapping = {}
        for namespace in tools or []:
            if not isinstance(namespace, dict) or namespace.get("type") != "namespace":
                continue
            namespace_name = str(namespace.get("name") or "").rstrip("_")
            for nested in namespace.get("tools") or []:
                if not isinstance(nested, dict) or nested.get("type") != "function":
                    continue
                tool_name = str(nested.get("name") or "").lstrip("_")
                if namespace_name and tool_name:
                    mapping[f"{namespace_name}__{tool_name}"] = (namespace_name, tool_name)
        return mapping

    def _responses_tools_to_chat_tools(self, tools):
        if not isinstance(tools, list):
            return []
        out = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            if tool.get("type") == "function":
                out.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.get("name", ""),
                            "description": tool.get("description", ""),
                            "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
                        },
                    }
                )
            elif tool.get("type") == "namespace":
                namespace = str(tool.get("name") or "").rstrip("_")
                for nested in tool.get("tools") or []:
                    if not isinstance(nested, dict) or nested.get("type") != "function":
                        continue
                    name = str(nested.get("name") or "").lstrip("_")
                    if not namespace or not name:
                        continue
                    out.append(
                        {
                            "type": "function",
                            "function": {
                                "name": f"{namespace}__{name}",
                                "description": nested.get("description") or tool.get("description") or "",
                                "parameters": nested.get("parameters") or {"type": "object", "properties": {}},
                            },
                        }
                    )
        return out

    def _chat_to_responses_payload(self, request_payload, chat_response):
        choice = (chat_response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = message.get("content") or ""
        finish_reason = str(choice.get("finish_reason") or "").strip().lower()
        incomplete_reason = None
        if finish_reason in {"length", "max_tokens", "max_output_tokens"}:
            incomplete_reason = "max_output_tokens"
        elif finish_reason in {"content_filter", "content-filter"}:
            incomplete_reason = "content_filter"
        response_status = "incomplete" if incomplete_reason else "completed"
        if self.debug_log_path:
            self._append_debug_record(
                {
                    "stage": "upstream_chat_finish",
                    "finish_reason": finish_reason or None,
                    "response_status": response_status,
                    "tool_call_count": len(message.get("tool_calls") or []),
                    "content_length": len(str(text)),
                }
            )
        response_id = chat_response.get("id") or f"resp_{int(time.time() * 1000)}"
        output = []
        if text:
            output.append(
                {
                    "id": "msg_0",
                    "type": "message",
                    "role": "assistant",
                    "status": response_status,
                    "content": [
                        {
                            "type": "output_text",
                            "text": text,
                            "annotations": [],
                        }
                    ],
                }
            )
        namespace_tools = self._namespace_tool_map(request_payload.get("tools"))
        for idx, tool_call in enumerate(message.get("tool_calls") or []):
            function = tool_call.get("function") or {}
            wire_name = str(function.get("name") or "")
            item = {
                "id": tool_call.get("id") or f"call_{idx}",
                "type": "function_call",
                "call_id": tool_call.get("id") or f"call_{idx}",
                "name": wire_name,
                "arguments": function.get("arguments", "{}"),
                "status": "completed",
            }
            if wire_name in namespace_tools:
                item["namespace"], item["name"] = namespace_tools[wire_name]
            output.append(item)
        if not output:
            output.append(
                {
                    "id": "msg_0",
                    "type": "message",
                    "role": "assistant",
                    "status": response_status,
                    "content": [
                        {
                            "type": "output_text",
                            "text": text,
                            "annotations": [],
                        }
                    ],
                }
            )
        chat_usage = chat_response.get("usage") or {}
        input_tokens = int(chat_usage.get("prompt_tokens") or chat_usage.get("input_tokens") or 0)
        output_tokens = int(chat_usage.get("completion_tokens") or chat_usage.get("output_tokens") or 0)
        input_token_details = chat_usage.get("input_tokens_details")
        if isinstance(input_token_details, dict):
            input_token_details = dict(input_token_details)
        else:
            prompt_token_details = chat_usage.get("prompt_tokens_details")
            input_token_details = (
                {"cached_tokens": int(prompt_token_details.get("cached_tokens") or 0)}
                if isinstance(prompt_token_details, dict)
                else {"cached_tokens": 0}
            )
        input_token_details.setdefault("cached_tokens", 0)
        response = {
            "id": response_id,
            "object": "response",
            "created_at": int(time.time()),
            "status": response_status,
            "model": request_payload.get("model") or self.model_name,
            "output": output,
            "parallel_tool_calls": bool(request_payload.get("parallel_tool_calls", True)),
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": int(chat_usage.get("total_tokens") or input_tokens + output_tokens),
                "input_tokens_details": input_token_details,
                "output_tokens_details": chat_usage.get("output_tokens_details") or {"reasoning_tokens": 0},
            },
        }
        if incomplete_reason:
            response["incomplete_details"] = {"reason": incomplete_reason}
        return response

    def _send_responses_sse(self, response_payload):
        events = [
            ("response.created", {"type": "response.created", "response": {**response_payload, "status": "in_progress", "output": []}}),
        ]
        for output_index, item in enumerate(response_payload.get("output") or []):
            events.append(("response.output_item.added", {"type": "response.output_item.added", "output_index": output_index, "item": item}))
            if item.get("type") == "message":
                text = ""
                content = item.get("content") or []
                if content and isinstance(content[0], dict):
                    text = content[0].get("text") or ""
                events.append(("response.content_part.added", {"type": "response.content_part.added", "item_id": item.get("id"), "output_index": output_index, "content_index": 0, "part": {"type": "output_text", "text": ""}}))
                if text:
                    events.append(("response.output_text.delta", {"type": "response.output_text.delta", "item_id": item.get("id"), "output_index": output_index, "content_index": 0, "delta": text}))
                events.append(("response.output_text.done", {"type": "response.output_text.done", "item_id": item.get("id"), "output_index": output_index, "content_index": 0, "text": text}))
                events.append(("response.content_part.done", {"type": "response.content_part.done", "item_id": item.get("id"), "output_index": output_index, "content_index": 0, "part": {"type": "output_text", "text": text, "annotations": []}}))
            elif item.get("type") == "function_call":
                events.append(("response.function_call_arguments.done", {"type": "response.function_call_arguments.done", "item_id": item.get("id"), "output_index": output_index, "arguments": item.get("arguments", "{}")}))
            events.append(("response.output_item.done", {"type": "response.output_item.done", "output_index": output_index, "item": item}))
        final_event = "response.incomplete" if response_payload.get("status") == "incomplete" else "response.completed"
        events.append((final_event, {"type": final_event, "response": response_payload}))
        body = "".join("event: %s\ndata: %s\n\n" % (event, json.dumps(data, ensure_ascii=False)) for event, data in events).encode("utf-8")
        self._send_bytes(200, body, "text/event-stream")

    def _send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(status, body, "application/json")

    def _send_bytes(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        print(f"[OpenAIProxy] {self.client_address[0]} {self.command} {self.path} - {fmt % args}")

    def _maybe_log_request_summary(self, body: bytes) -> None:
        if not self.debug_log_path:
            return
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            payload = {"_raw_bytes": len(body)}

        summary = {
            "method": self.command,
            "path": self.path,
            "top_level_keys": sorted(payload.keys()) if isinstance(payload, dict) else None,
        }
        if isinstance(payload, dict):
            if self.system_prompt_prefix:
                messages_for_check = payload.get("messages") or []
                summary["system_prompt_prefix_injected"] = any(
                    isinstance(message, dict)
                    and message.get("role") in {"system", "developer"}
                    and str(message.get("content") or "").strip().startswith(self.system_prompt_prefix.strip())
                    for message in messages_for_check
                )
                summary["system_prompt_prefix_chars"] = len(self.system_prompt_prefix)
            for key in ("model", "stream", "temperature", "max_tokens", "max_completion_tokens", "tool_choice", "response_format", "parallel_tool_calls"):
                if key in payload:
                    value = payload[key]
                    summary[key] = value if not isinstance(value, (dict, list)) else type(value).__name__
            tools = payload.get("tools")
            if isinstance(tools, list):
                summary["tool_count"] = len(tools)
                summary["tool_names"] = [
                    str(tool.get("function", {}).get("name") or tool.get("name") or tool.get("tool_name") or "")
                    for tool in tools
                    if isinstance(tool, dict)
                ][:20]
                summary["tool_item_keys"] = sorted({key for tool in tools if isinstance(tool, dict) for key in tool.keys()})
                function_keys = sorted({
                    key
                    for tool in tools
                    if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
                    for key in tool["function"].keys()
                })
                if function_keys:
                    summary["tool_function_keys"] = function_keys
            messages = payload.get("messages")
            if isinstance(messages, list):
                summary["message_count"] = len(messages)
                summary["message_roles"] = [str(msg.get("role", "")) for msg in messages[:8] if isinstance(msg, dict)]
                summary["last_message_len"] = len(str(messages[-1].get("content", ""))) if messages and isinstance(messages[-1], dict) else 0

        self._append_debug_record(summary)

    def _append_debug_record(self, value) -> None:
        line = json.dumps(value, ensure_ascii=False, sort_keys=True)
        os.makedirs(os.path.dirname(self.debug_log_path), exist_ok=True)
        with self.debug_log_lock:
            with open(self.debug_log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        if os.environ.get("RED_AGENT_WORLD_PROXY_DEBUG_STDOUT", "0") == "1":
            print(f"[OpenAIProxyDebug] {line}")

    def _maybe_log_response_summary(self, status: int, body: bytes) -> None:
        if not self.debug_log_path:
            return
        summary = {"method": self.command, "path": self.path, "response_status": status}
        if status >= 400:
            try:
                payload = json.loads(body.decode("utf-8"))
                summary["error_keys"] = sorted(payload.keys()) if isinstance(payload, dict) else None
                error_value = payload.get("error") if isinstance(payload, dict) else None
                if isinstance(error_value, dict):
                    summary["error_type"] = error_value.get("type")
                    summary["error_code"] = error_value.get("code")
                    summary["error_message"] = str(error_value.get("message") or "")[:1000]
                elif isinstance(error_value, str):
                    summary["error_message"] = error_value[:1000]
                elif isinstance(payload, dict):
                    summary["error_message"] = str(payload)[:1000]
            except Exception:
                summary["error_message"] = body.decode("utf-8", errors="replace")[:1000]
        line = json.dumps(summary, ensure_ascii=False, sort_keys=True)
        os.makedirs(os.path.dirname(self.debug_log_path), exist_ok=True)
        with self.debug_log_lock:
            with open(self.debug_log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        if os.environ.get("RED_AGENT_WORLD_PROXY_DEBUG_STDOUT", "0") == "1":
            print(f"[OpenAIProxyDebug] {line}")

    def _copy_response_headers(self, headers, content_length=None):
        for key, value in headers.items():
            if key.lower() in {"connection", "transfer-encoding", "content-encoding", "content-length"}:
                continue
            self.send_header(key, value)
        if content_length is not None:
            self.send_header("Content-Length", str(content_length))


class OpenAIProxyServer:
    def __init__(
        self,
        host: str,
        port: int,
        upstream_base_url: str,
        upstream_api_key: str,
        client_api_key: str,
        model_name: str = "",
        system_prompt_prefix: str = "",
        upstream_reasoning_effort: str = None,
    ):
        handler = type("ConfiguredProxyHandler", (ProxyHandler,), {})
        handler.upstream_base_url = upstream_base_url
        handler.upstream_api_key = upstream_api_key
        if not client_api_key:
            raise ValueError("client_api_key must be a non-empty per-run token")
        handler.client_api_key = client_api_key
        handler.model_name = model_name
        handler.system_prompt_prefix = system_prompt_prefix
        handler.upstream_reasoning_effort = upstream_reasoning_effort
        handler.debug_log_path = os.environ.get("RED_AGENT_WORLD_PROXY_DEBUG_LOG", "")
        handler.response_history = {}
        handler.response_history_lock = threading.Lock()
        handler.thought_signature_cache = {}
        handler.thought_signature_cache_lock = threading.Lock()
        self.handler = handler
        self.server = ThreadingHTTPServer((host, port), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def server_address(self):
        return self.server.server_address

    def start(self):
        self.thread.start()

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
