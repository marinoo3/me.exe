class UnsafeRequestException(Exception):
    """Custom exception for unsafe requests detected by guardrail."""

    def __init__(
        self,
        message: str,
        confidence: float = 0.0,
        checkpoint: str = "",
    ):
        super().__init__(message)
        self.confidence = confidence
        self.checkpoint = checkpoint

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON response."""
        return {
            "error": str(self),
            "code": "INJECTION_DETECTED",
            "confidence": self.confidence,
            "checkpoint": self.checkpoint,
        }