class VLMGenerationError(RuntimeError):
    """Raised when a VLM backend fails to produce a usable response.

    Adapters must raise this instead of returning the error text as
    VLMResponse.content — otherwise transport failures end up persisted
    as event descriptions in the vector store.
    """
