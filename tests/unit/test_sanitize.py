"""Unit tests for input sanitization utilities."""

import pytest

from scratchy.utils.sanitize import sanitize_prompt, validate_dimensions, is_safe_filename


class TestSanitizePrompt:
    """Tests for prompt sanitization."""

    def test_basic_prompt(self):
        """Test that basic prompts pass through."""
        prompt = "A beautiful sunset over the ocean"
        result = sanitize_prompt(prompt)
        assert result == prompt

    def test_empty_prompt(self):
        """Test empty prompt handling."""
        assert sanitize_prompt("") == ""
        assert sanitize_prompt(None) == ""

    def test_unicode_normalization(self):
        """Test Unicode normalization."""
        # Different ways to represent the same character
        prompt = "café"  # Using composed form
        result = sanitize_prompt(prompt)
        assert result == "café"

    def test_control_character_removal(self):
        """Test that control characters are removed."""
        prompt = "Hello\x00World\x0bTest"
        result = sanitize_prompt(prompt)
        assert "\x00" not in result
        assert "\x0b" not in result
        assert "Hello" in result
        assert "World" in result

    def test_preserves_newlines(self):
        """Test that newlines are preserved."""
        prompt = "Line 1\nLine 2\nLine 3"
        result = sanitize_prompt(prompt)
        assert "\n" in result

    def test_collapses_multiple_spaces(self):
        """Test that multiple spaces are collapsed."""
        prompt = "Too    many     spaces"
        result = sanitize_prompt(prompt)
        assert "  " not in result
        assert "Too many spaces" == result

    def test_collapses_excessive_newlines(self):
        """Test that excessive newlines are collapsed."""
        prompt = "Line 1\n\n\n\n\nLine 2"
        result = sanitize_prompt(prompt)
        assert "\n\n\n" not in result

    def test_truncation(self):
        """Test that long prompts are truncated."""
        prompt = "a" * 3000
        result = sanitize_prompt(prompt, max_length=2000)
        assert len(result) == 2000

    def test_strips_whitespace(self):
        """Test that leading/trailing whitespace is stripped."""
        prompt = "   trimmed   "
        result = sanitize_prompt(prompt)
        assert result == "trimmed"


class TestValidateDimensions:
    """Tests for dimension validation."""

    def test_valid_dimensions(self):
        """Test valid dimensions pass through."""
        width, height, warnings = validate_dimensions(1024, 1024)
        assert width == 1024
        assert height == 1024
        assert len(warnings) == 0

    def test_rounds_to_multiple_of_64(self):
        """Test dimensions are rounded to multiple of 64."""
        width, height, warnings = validate_dimensions(1000, 1000)
        assert width == 1024  # Rounded up
        assert height == 1024
        assert len(warnings) == 2

    def test_clamps_minimum(self):
        """Test minimum dimension clamping."""
        width, height, warnings = validate_dimensions(100, 100)
        assert width == 256
        assert height == 256
        assert any("minimum" in w for w in warnings)

    def test_clamps_maximum(self):
        """Test maximum dimension clamping."""
        width, height, warnings = validate_dimensions(3000, 3000)
        assert width == 2048
        assert height == 2048
        assert any("maximum" in w for w in warnings)

    def test_custom_multiple(self):
        """Test custom multiple."""
        width, height, warnings = validate_dimensions(100, 100, multiple=8, min_dim=64)
        assert width % 8 == 0
        assert height % 8 == 0


class TestIsSafeFilename:
    """Tests for filename safety checking."""

    def test_safe_filename(self):
        """Test safe filename."""
        assert is_safe_filename("image.png") is True
        assert is_safe_filename("my_image_123.jpg") is True

    def test_path_traversal(self):
        """Test path traversal detection."""
        assert is_safe_filename("../etc/passwd") is False
        assert is_safe_filename("../../file.txt") is False
        assert is_safe_filename("..") is False

    def test_forward_slash(self):
        """Test forward slash detection."""
        assert is_safe_filename("path/to/file") is False

    def test_backslash(self):
        """Test backslash detection."""
        assert is_safe_filename("path\\to\\file") is False

    def test_null_byte(self):
        """Test null byte detection."""
        assert is_safe_filename("file\x00.txt") is False

    def test_empty_filename(self):
        """Test empty filename."""
        assert is_safe_filename("") is False
        assert is_safe_filename(None) is False
