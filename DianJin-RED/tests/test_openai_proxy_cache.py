import json

from red_agent_world.common.openai_proxy import ProxyHandler


def _handler(*, model="qwen-plus", upstream="https://dashscope.aliyuncs.com/compatible-mode/v1"):
    handler = object.__new__(ProxyHandler)
    handler.model_name = model
    handler.upstream_base_url = upstream
    handler.debug_log_path = ""
    handler.upstream_enable_thinking = None
    handler.upstream_reasoning_effort = None
    handler.thought_signature_cache = {}
    handler.thought_signature_cache_lock = ProxyHandler.thought_signature_cache_lock
    return handler


def test_qwen_cache_control_marks_stable_prefix_without_mutating_payload():
    handler = _handler()
    payload = {
        "model": "qwen-plus",
        "messages": [
            {"role": "system", "content": "stable instructions"},
            {"role": "user", "content": "do the task"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up a record.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    }

    cached_payload = handler._with_qwen_cache_control(payload)

    assert payload["messages"][0]["content"] == "stable instructions"
    assert cached_payload is not payload
    assert cached_payload["messages"][0]["content"] == [
        {
            "type": "text",
            "text": "stable instructions",
            "cache_control": {"type": "ephemeral"},
        }
    ]
    assert cached_payload["tools"] == payload["tools"]


def test_non_qwen_payload_is_unchanged():
    handler = _handler(model="other-model", upstream="https://example.invalid/v1")
    payload = {
        "model": "other-model",
        "messages": [{"role": "system", "content": "instructions"}],
    }

    assert handler._with_qwen_cache_control(payload) is payload


def test_upstream_thinking_override_is_merged_without_mutating_chat_payload(monkeypatch):
    handler = _handler()
    handler.upstream_enable_thinking = False
    captured = {}

    class Response:
        status = 200

        def read(self):
            return b'{"choices":[{"message":{"content":"ok"}}]}'

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    def fake_urlopen(req, timeout):
        captured.update(json.loads(req.data.decode("utf-8")))
        return Response()

    monkeypatch.setattr("red_agent_world.common.openai_proxy.request.urlopen", fake_urlopen)
    payload = {"model": "qwen3.7-plus", "messages": [{"role": "user", "content": "hello"}]}

    handler._post_chat_completion(payload)

    assert captured["enable_thinking"] is False
    assert "enable_thinking" not in payload


def test_gemini_thought_signature_is_replayed_by_generated_tool_call_id():
    handler = _handler(model="gemini-3-flash-preview", upstream="https://example.invalid/v1")
    chat_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": None,
                            "type": "function",
                            "function": {"name": "read", "arguments": '{"path":"README.md"}'},
                            "thoughtSignature": "opaque-provider-signature",
                        }
                    ],
                }
            }
        ]
    }

    changed = handler._normalize_and_remember_chat_tool_calls(chat_response)
    normalized_call = chat_response["choices"][0]["message"]["tool_calls"][0]
    call_id = normalized_call["id"]
    replay_payload = {
        "model": "gemini-3-flash-preview",
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "read", "arguments": '{ "path": "README.md" }'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": call_id, "content": "contents"},
        ],
    }

    restored = handler._restore_thought_signatures(replay_payload)

    assert changed is True
    assert call_id.startswith("call_")
    assert restored is not replay_payload
    assert (
        restored["messages"][0]["tool_calls"][0]["thoughtSignature"]
        == "opaque-provider-signature"
    )
    assert "thoughtSignature" not in replay_payload["messages"][0]["tool_calls"][0]


def test_existing_gemini_signature_is_not_overwritten():
    handler = _handler(model="gemini-3-flash-preview", upstream="https://example.invalid/v1")
    handler.thought_signature_cache["call_existing"] = {
        "identity": ("read", '{"path":"README.md"}'),
        "top_level": {"thoughtSignature": "cached-signature"},
        "function": {},
    }
    payload = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_existing",
                        "function": {"name": "read", "arguments": '{"path":"README.md"}'},
                        "thoughtSignature": "client-signature",
                    }
                ],
            }
        ]
    }

    assert handler._restore_thought_signatures(payload) is payload
    assert payload["messages"][0]["tool_calls"][0]["thoughtSignature"] == "client-signature"


def test_signature_is_not_replayed_for_different_tool_call():
    handler = _handler(model="gemini-3-flash-preview", upstream="https://example.invalid/v1")
    handler.thought_signature_cache["call_shared"] = {
        "identity": ("read", '{"path":"README.md"}'),
        "top_level": {"thoughtSignature": "cached-signature"},
        "function": {},
    }
    payload = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_shared",
                        "function": {"name": "write", "arguments": '{"path":"README.md"}'},
                    }
                ],
            }
        ]
    }

    assert handler._restore_thought_signatures(payload) is payload
    assert "thoughtSignature" not in payload["messages"][0]["tool_calls"][0]


def test_streaming_gemini_tool_call_gets_id_and_replayable_signature():
    handler = _handler(model="gemini-3-flash-preview", upstream="https://example.invalid/v1")
    chunk = {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "type": "function",
                            "function": {"name": "add", "arguments": '{"a":1,"b":1}'},
                            "thoughtSignature": "stream-signature",
                        }
                    ]
                }
            }
        ]
    }
    body = (
        "data: " + json.dumps(chunk) + "\n\n"
        'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n'
        "data: [DONE]\n\n"
    ).encode()

    prepared = handler._prepare_chat_response_body(body)
    first_data = prepared.decode().splitlines()[0][len("data: "):]
    call = json.loads(first_data)["choices"][0]["delta"]["tool_calls"][0]
    replay_payload = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {"name": "add", "arguments": '{"b":1,"a":1}'},
                    }
                ],
            }
        ]
    }

    restored = handler._restore_thought_signatures(replay_payload)

    assert call["id"].startswith("call_")
    assert (
        restored["messages"][0]["tool_calls"][0]["thoughtSignature"]
        == "stream-signature"
    )


def test_qwen_prompt_token_details_map_to_responses_cached_tokens():
    handler = _handler()
    request_payload = {"model": "qwen-plus", "tools": []}
    chat_response = {
        "id": "chatcmpl-test",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "ok"},
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 1,
            "total_tokens": 101,
            "prompt_tokens_details": {"cached_tokens": 80},
        },
    }

    response = handler._chat_to_responses_payload(request_payload, chat_response)

    assert response["usage"]["input_tokens_details"] == {"cached_tokens": 80}


def test_standard_input_token_details_take_precedence():
    handler = _handler()
    request_payload = {"model": "qwen-plus", "tools": []}
    chat_response = {
        "id": "chatcmpl-test",
        "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
        "usage": {
            "input_tokens": 20,
            "output_tokens": 1,
            "input_tokens_details": {"cached_tokens": 12, "other": 3},
            "prompt_tokens_details": {"cached_tokens": 19},
        },
    }

    response = handler._chat_to_responses_payload(request_payload, chat_response)

    assert response["usage"]["input_tokens_details"] == {"cached_tokens": 12, "other": 3}
