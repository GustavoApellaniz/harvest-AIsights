"""Exceções de domínio. O handler em main.py as converte em respostas JSON."""


class AppError(Exception):
    status_code = 500

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class LLMError(AppError):
    """Falha na comunicação com o provedor de LLM."""

    status_code = 502


class LLMNotConfiguredError(LLMError):
    """Credencial do provedor ausente."""

    status_code = 503


class IndexEmptyError(AppError):
    """Nenhum documento indexado ainda."""

    status_code = 503
