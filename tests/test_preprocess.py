"""Tests for data preprocessing utilities."""

import pytest
from src.data.preprocess import toy_normalize, rich_normalize


def test_toy_normalize():
    """Test basic normalization."""
    text = "Hello https://example.com 123!"
    expected = "hello <URL> <NUM>!"
    assert toy_normalize(text) == expected


def test_rich_normalize():
    """Test comprehensive normalization."""
    text = "Hello @user #hashtag 123!!! 😀"
    result = rich_normalize(text)
    assert "<user>" in result
    assert "<hashtag>" in result
    assert "<num>" in result
    assert "<emoji>" in result
    assert result == result.lower()  # Should be lowercase