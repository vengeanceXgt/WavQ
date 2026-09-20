from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

class FECResult:
    def __init__(self, fec_type: str, decoded_bits, validation_evidence: Dict[str, Any]):
        self.fec_type = fec_type
        self.decoded_bits = decoded_bits
        self.validation_evidence = validation_evidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fec_type": self.fec_type,
            "decoded_bits": self.decoded_bits,
            "validation_evidence": self.validation_evidence,
        }

class FECScheme(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def identify_and_decode(self, llr) -> Optional[FECResult]:
        """
        Attempts to identify and decode the given LLRs.
        Returns an FECResult if the scheme believes it is a match (even a weak one),
        or None if it mathematically cannot be this scheme.
        """
        pass
