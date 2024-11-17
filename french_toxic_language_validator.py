from typing import Any, Dict, Optional, Callable
from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)
import re
from MauvaiseLangue import scrape_insultes  # Fonction pour obtenir la liste des insultes

@register_validator(name="guardrails/french_toxic_language", data_type="string")
class FrenchToxicLanguage(Validator):
    """Validator to detect French toxic language in input text.
    
    This validator checks if the given text contains any French insults based on a predefined list of insults.
    """

    def __init__(self, on_fail: Optional[Callable] = None):
        """
        Initializes the FrenchToxicLanguage validator.
        
        Args:
            on_fail (Callable, optional): Action to perform when validation fails (e.g., reask, fix, filter).
        """
        super().__init__(on_fail=on_fail)
        # Load insults using the scrape function
        self.insultes = scrape_insultes()

    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        """
        Validates the given string to check if it contains any French insults.
        
        Args:
            value (Any): The input string to validate.
            metadata (Dict): Additional metadata (optional).
        
        Returns:
            ValidationResult: PassResult if valid, FailResult otherwise.
        """
        detected_insultes = self.detect_insultes(value)
        
        if detected_insultes:
            return FailResult(
                error_message=f"Le texte contient un langage inapproprié : {', '.join(detected_insultes)}",
                fix_value=None  # Optionally, you could suggest filtering the detected insults
            )
        
        return PassResult()

    def detect_insultes(self, text: str) -> list:
        """
        Detects French insults in the given text using a list of known insults.
        
        This function uses regular expressions to check for the presence of each insult as a whole word 
        in the text, ensuring partial matches (e.g., "con" in "conseil") are ignored.
        
        Args:
            text (str): The input text to check for insults.
        
        Returns:
            list: A list of detected insults found in the text.
        """
        text = text.lower()  # Convert text to lowercase for case-insensitive matching
        detected_insultes = []
        
        for insult in self.insultes:
            # Create a regex pattern to match the insult as a whole word
            pattern = r'\b' + re.escape(insult.lower()) + r'\b'
            if re.search(pattern, text):
                detected_insultes.append(insult)
        
        return detected_insultes

# Tests
class TestFrenchToxicLanguage:
    """Test suite for the FrenchToxicLanguage validator."""
    
    def test_success_case(self):
        """Test case where no toxic language is present in the input."""
        validator = FrenchToxicLanguage()
        result = validator.validate("Bonjour, comment ça va ?", {})
        assert isinstance(result, PassResult)

    def test_failure_case(self):
        """Test case where toxic language is present in the input."""
        validator = FrenchToxicLanguage()
        result = validator.validate("Tu es un idiot.", {})
        assert isinstance(result, FailResult)
        assert "idiot" in result.error_message
