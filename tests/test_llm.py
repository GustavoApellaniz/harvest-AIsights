from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors, types

from app.core.exceptions import LLMError, LLMNotConfiguredError
from app.services.llm import GeminiLLM


def _response(text: str | None, finish: str = "STOP", block_reason=None):
    parts = [types.Part(text=text)] if text is not None else []
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=parts), finish_reason=finish)],
        prompt_feedback=types.GenerateContentResponsePromptFeedback(block_reason=block_reason) if block_reason else None,
    )


class StubModels:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.kwargs = result, error, None

    def generate_content(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.result


def _llm(result=None, error=None):
    llm = GeminiLLM(api_key="chave-falsa", model="m", max_tokens=10, temperature=0.2, timeout=1)
    llm._client = SimpleNamespace(models=StubModels(result, error))
    return llm


def test_missing_key_raises_not_configured():
    with pytest.raises(LLMNotConfiguredError):
        GeminiLLM(api_key=None, model="m", max_tokens=10, temperature=0.2, timeout=1)


def test_returns_text_and_maps_roles_and_config():
    llm = _llm(_response("olá"))
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]
    assert llm.generate("sistema", messages) == "olá"
    kwargs = llm._client.models.kwargs
    assert [c.role for c in kwargs["contents"]] == ["user", "model", "user"]
    assert kwargs["contents"][1].parts[0].text == "a1"
    assert kwargs["config"].system_instruction == "sistema"
    assert kwargs["config"].max_output_tokens == 10
    assert kwargs["model"] == "m"


def test_truncated_but_nonempty_text_is_returned():
    assert _llm(_response("parcial", finish="MAX_TOKENS")).generate("s", []) == "parcial"


def test_empty_with_max_tokens_mentions_setting():
    with pytest.raises(LLMError, match="LLM_MAX_TOKENS"):
        _llm(_response(None, finish="MAX_TOKENS")).generate("s", [])


def test_empty_response_and_blocked_prompt():
    with pytest.raises(LLMError):
        _llm(_response(None)).generate("s", [])
    with pytest.raises(LLMError, match="bloqueou"):
        _llm(_response(None, finish="STOP", block_reason="SAFETY")).generate("s", [])


@pytest.mark.parametrize(
    "error, expected",
    [
        (errors.ClientError(429, {"error": {"message": "quota", "status": "RESOURCE_EXHAUSTED"}}), "Limite"),
        (errors.ClientError(400, {"error": {"message": "API key not valid. Please pass a valid API key."}}), "Credencial"),
        (errors.ClientError(403, {"error": {"message": "denied"}}), "Credencial"),
        (errors.ClientError(404, {"error": {"message": "model not found"}}), "LLM_MODEL"),
        (errors.ClientError(400, {"error": {"message": "bad request"}}), "rejeitada"),
        (errors.ServerError(503, {"error": {"message": "overloaded"}}), "indisponível"),
        (httpx.ConnectError("sem rede"), "conectar"),
        (httpx.ReadTimeout("timeout"), "conectar"),
    ],
)
def test_provider_errors_are_wrapped(error, expected):
    with pytest.raises(LLMError, match=expected):
        _llm(error=error).generate("s", [])
