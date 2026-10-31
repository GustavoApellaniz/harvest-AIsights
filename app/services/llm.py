"""Integração com o provedor de LLM (Google Gemini API, SDK oficial `google-genai`)."""

import logging
from typing import Protocol

import httpx
from google import genai
from google.genai import errors, types

from app.core.exceptions import LLMError, LLMNotConfiguredError

logger = logging.getLogger(__name__)


class LLM(Protocol):
    def generate(self, system: str, messages: list[dict[str, str]]) -> str: ...


class GeminiLLM:
    def __init__(self, api_key: str | None, model: str, max_tokens: int, temperature: float, timeout: float):
        if not api_key:
            raise LLMNotConfiguredError("GEMINI_API_KEY não configurada. Veja o arquivo .env.example.")
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=int(timeout * 1000),  # o SDK espera milissegundos
                retry_options=types.HttpRetryOptions(attempts=3),
            ),
        )
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    @staticmethod
    def _to_contents(messages: list[dict[str, str]]) -> list[types.Content]:
        # O Gemini usa o papel "model" no lugar de "assistant".
        return [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[types.Part.from_text(text=m["content"])],
            )
            for m in messages
        ]

    def generate(self, system: str, messages: list[dict[str, str]]) -> str:
        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        try:
            response = self._client.models.generate_content(
                model=self._model, contents=self._to_contents(messages), config=config
            )
        except errors.ClientError as exc:
            raise self._map_client_error(exc) from exc
        except errors.ServerError as exc:
            logger.warning("Provedor de LLM indisponível: status %s", exc.code)
            raise LLMError(f"Provedor de LLM indisponível no momento (status {exc.code}).") from exc
        except errors.APIError as exc:
            logger.error("Erro do provedor de LLM: status %s", exc.code)
            raise LLMError(f"Erro do provedor de LLM (status {exc.code}).") from exc
        except httpx.HTTPError as exc:  # timeout, DNS, conexão recusada...
            logger.warning("Falha de rede com o provedor de LLM: %s", type(exc).__name__)
            raise LLMError("Não foi possível conectar ao provedor de LLM.") from exc

        return self._extract_text(response)

    @staticmethod
    def _map_client_error(exc: errors.ClientError) -> LLMError:
        code = exc.code
        message = str(getattr(exc, "message", "") or "")
        if code in (401, 403) or (code == 400 and "api key" in message.lower()):
            logger.error("Credencial rejeitada pelo provedor de LLM (status %s)", code)
            return LLMError("Credencial do provedor de LLM inválida ou sem permissão.")
        if code == 404:
            logger.error("Modelo não encontrado pelo provedor de LLM")
            return LLMError("Modelo não encontrado. Confira o valor de LLM_MODEL.")
        if code == 429:
            logger.warning("Cota/limite de requisições do provedor atingido")
            return LLMError("Limite de requisições ou cota do provedor de LLM atingido. Tente novamente.")
        logger.error("Requisição rejeitada pelo provedor de LLM (status %s)", code)
        return LLMError(f"Requisição rejeitada pelo provedor de LLM (status {code}).")

    @staticmethod
    def _extract_text(response: types.GenerateContentResponse) -> str:
        candidate = response.candidates[0] if response.candidates else None
        finish = str(candidate.finish_reason) if candidate and candidate.finish_reason else ""
        text = (response.text or "").strip() if candidate else ""
        if text:
            if "MAX_TOKENS" in finish:
                logger.warning("Resposta truncada por LLM_MAX_TOKENS")
            return text
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            raise LLMError("O provedor bloqueou a requisição.")
        if "MAX_TOKENS" in finish:
            raise LLMError("Resposta vazia: o limite de tokens foi consumido. Aumente LLM_MAX_TOKENS.")
        if "SAFETY" in finish:
            raise LLMError("O provedor bloqueou a resposta por política de segurança.")
        raise LLMError("O provedor de LLM retornou uma resposta vazia.")
