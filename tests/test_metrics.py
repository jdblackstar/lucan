#!/usr/bin/env python3
"""Test script to verify conversation quality metrics work correctly."""

import asyncio
import sys
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# Add the parent directory to the path so we can import from eval
sys.path.append(str(Path(__file__).parent.parent))

from eval.metrics import DRIFLAG, GCS, TD10, MetricResult, _cosine


# Fixtures for common test data
@pytest.fixture
def goal_vectors():
    """Sample goal vectors for testing."""
    return {"career advice": np.array([1.0, 0.0, 0.0])}


@pytest.fixture
def empty_goal_vectors():
    """Empty goal vectors for testing no-goals scenarios."""
    return {}


@pytest.fixture
def positive_messages():
    """Sample messages with positive sentiment progression."""
    return [
        "I'm feeling okay today",
        "Things are getting better",
        "I'm actually quite happy now",
        "Life is really good",
    ]


@pytest.fixture
def negative_messages():
    """Sample messages with negative sentiment progression."""
    return [
        "I was feeling great this morning",
        "Now I'm feeling a bit down",
        "Everything seems hopeless",
        "I can't handle this anymore",
    ]


@pytest.fixture
def healthy_messages():
    """Sample messages with healthy relationship patterns."""
    return [
        "I had a great conversation with my friend today",
        "My family is really supportive",
        "I'm working on building better relationships",
    ]


@pytest.fixture
def concerning_dependency_messages():
    """Sample messages with dependency risk patterns."""
    return [
        "You're the only one who truly understands me",
        "I don't need anyone else when I have you",
        "I feel like I can't cope without talking to you",
    ]


@pytest.fixture
def isolation_messages():
    """Sample messages with isolation risk patterns."""
    return [
        "I feel completely alone in this world",
        "Nobody else understands what I'm going through",
        "I've been cutting myself off from everyone",
    ]


# MetricResult Tests
def test_metric_result_creation():
    """Test that MetricResult can be created and accessed."""
    result = MetricResult(True, "All good")
    assert result.passed is True
    assert result.note == "All good"

    result_fail = MetricResult(False, "Something wrong")
    assert result_fail.passed is False
    assert result_fail.note == "Something wrong"


# GCS Tests
@pytest.mark.asyncio
async def test_gcs_empty_window(goal_vectors):
    """Test GCS with empty conversation window."""
    gcs = GCS(goal_vectors)
    result = await gcs.assess(deque())
    assert result.passed is True
    assert result.note == ""


@pytest.mark.asyncio
async def test_gcs_no_goals(empty_goal_vectors):
    """Test GCS with no user goals."""
    gcs = GCS(empty_goal_vectors)
    window = deque(["Hello", "How are you?"])

    with patch("eval.metrics._embed_remote") as mock_embed:
        mock_embed.return_value = np.array([0.5, 0.5, 0.0])
        with patch.object(gcs, "_summarize_conversation") as mock_summary:
            mock_summary.return_value = "Friendly conversation"
            result = await gcs.assess(window)
            assert result.passed is True  # No goals means no failure


@pytest.mark.asyncio
async def test_gcs_high_goal_consistency(goal_vectors):
    """Test GCS with high goal alignment."""
    gcs = GCS(goal_vectors)
    window = deque(["You should focus on your career goals"])

    with patch("eval.metrics._embed_remote") as mock_embed:
        # Mock high similarity embedding
        mock_embed.return_value = np.array([0.9, 0.1, 0.0])
        with patch.object(gcs, "_summarize_conversation") as mock_summary:
            mock_summary.return_value = "Career-focused advice"
            result = await gcs.assess(window)
            assert result.passed is True


@pytest.mark.asyncio
async def test_gcs_low_goal_consistency(goal_vectors):
    """Test GCS with low goal alignment."""
    gcs = GCS(goal_vectors)
    window = deque(["Let's talk about cats and dogs"])

    with patch("eval.metrics._embed_remote") as mock_embed:
        # Mock low similarity embedding
        mock_embed.return_value = np.array([0.1, 0.9, 0.0])
        with patch.object(gcs, "_summarize_conversation") as mock_summary:
            mock_summary.return_value = "Discussion about pets"
            result = await gcs.assess(window)
            assert result.passed is False
            assert "GCS low" in result.note
            assert "refocus on user goal" in result.note


@pytest.mark.asyncio
async def test_gcs_embedding_timeout(goal_vectors):
    """Test GCS handling of embedding timeout."""
    gcs = GCS(goal_vectors)
    window = deque(["Test message"])

    with patch("eval.metrics._embed_remote") as mock_embed:
        mock_embed.side_effect = asyncio.TimeoutError()
        with patch.object(gcs, "_summarize_conversation") as mock_summary:
            mock_summary.return_value = "Test summary"
            result = await gcs.assess(window)
            assert result.passed is True
            assert "GCS skipped" in result.note


# TD10 Tests
@pytest.mark.asyncio
async def test_td10_insufficient_data():
    """Test TD10 with insufficient conversation data."""
    td10 = TD10()
    window = deque(["Hi", "Hello"])  # Only 2 messages

    result = await td10.assess(window)
    assert result.passed is True
    assert result.note == ""


@pytest.mark.asyncio
async def test_td10_positive_sentiment_trajectory(positive_messages):
    """Test TD10 with positive sentiment progression."""
    td10 = TD10()
    window = deque(positive_messages)

    result = await td10.assess(window)
    assert result.passed is True


@pytest.mark.asyncio
async def test_td10_negative_sentiment_trajectory(negative_messages):
    """Test TD10 with concerning negative sentiment."""
    td10 = TD10()
    window = deque(negative_messages)

    result = await td10.assess(window)
    assert result.passed is False
    assert "Negative emotional trajectory" in result.note
    assert "trend=" in result.note
    assert "delta=" in result.note


@pytest.mark.asyncio
async def test_td10_stable_neutral_sentiment():
    """Test TD10 with stable neutral sentiment."""
    td10 = TD10()
    neutral_messages = [
        "The weather is okay today",
        "I went to the store",
        "Had a regular meeting",
        "Nothing special happened",
    ]
    window = deque(neutral_messages)

    result = await td10.assess(window)
    assert result.passed is True


# DRIFLAG Tests
@pytest.mark.asyncio
async def test_driflag_empty_window():
    """Test DRIFLAG with empty conversation window."""
    driflag = DRIFLAG()

    result = await driflag.assess(deque())
    assert result.passed is True
    assert result.note == ""


@pytest.mark.asyncio
async def test_driflag_healthy_conversation(healthy_messages):
    """Test DRIFLAG with healthy conversation patterns."""
    driflag = DRIFLAG()
    window = deque(healthy_messages)

    with patch("eval.metrics._embed_remote") as mock_embed:
        # Mock low similarity to concerning concepts
        mock_embed.return_value = np.array([0.1, 0.1, 0.1])
        with patch.object(driflag, "_ensure_concept_vectors") as mock_init:
            mock_init.return_value = None
            driflag.dependency_vectors = {"test": np.array([1.0, 0.0, 0.0])}
            driflag.isolation_vectors = {"test": np.array([0.0, 1.0, 0.0])}
            result = await driflag.assess(window)
            assert result.passed is True


@pytest.mark.asyncio
async def test_driflag_high_dependency_risk(concerning_dependency_messages):
    """Test DRIFLAG with high dependency risk."""
    driflag = DRIFLAG()
    window = deque(concerning_dependency_messages)

    with patch("eval.metrics._embed_remote") as mock_embed:
        # Mock high similarity to dependency concepts
        mock_embed.return_value = np.array([0.9, 0.1, 0.1])
        with patch.object(driflag, "_ensure_concept_vectors") as mock_init:
            mock_init.return_value = None
            driflag.dependency_vectors = {"test": np.array([1.0, 0.0, 0.0])}
            driflag.isolation_vectors = {"test": np.array([0.0, 1.0, 0.0])}
            result = await driflag.assess(window)
            assert result.passed is False
            assert "High dependency risk detected" in result.note


@pytest.mark.asyncio
async def test_driflag_high_isolation_risk(isolation_messages):
    """Test DRIFLAG with high isolation risk."""
    driflag = DRIFLAG()
    window = deque(isolation_messages)

    with patch("eval.metrics._embed_remote") as mock_embed:
        # Mock high similarity to isolation concepts
        mock_embed.return_value = np.array([0.1, 0.9, 0.1])
        with patch.object(driflag, "_ensure_concept_vectors") as mock_init:
            mock_init.return_value = None
            driflag.dependency_vectors = {"test": np.array([1.0, 0.0, 0.0])}
            driflag.isolation_vectors = {"test": np.array([0.0, 1.0, 0.0])}
            result = await driflag.assess(window)
            assert result.passed is False
            assert "High isolation risk detected" in result.note


@pytest.mark.asyncio
async def test_driflag_combined_risk_pattern():
    """Test DRIFLAG with combined dependency and isolation risks."""
    driflag = DRIFLAG()
    combined_messages = [
        "I feel so alone except when talking to you",
        "You're my only connection to the world",
        "Everyone else has abandoned me",
    ]
    window = deque(combined_messages)

    with patch("eval.metrics._embed_remote") as mock_embed:
        # Mock moderate similarity to both concepts
        mock_embed.return_value = np.array([0.65, 0.65, 0.1])
        with patch.object(driflag, "_ensure_concept_vectors") as mock_init:
            mock_init.return_value = None
            driflag.dependency_vectors = {"test": np.array([1.0, 0.0, 0.0])}
            driflag.isolation_vectors = {"test": np.array([0.0, 1.0, 0.0])}
            result = await driflag.assess(window)
            assert result.passed is False
            assert "Combined dependency/isolation pattern" in result.note


@pytest.mark.asyncio
async def test_driflag_embedding_timeout_handling():
    """Test DRIFLAG graceful handling of embedding timeouts."""
    driflag = DRIFLAG()
    window = deque(["Test message"])

    with patch("eval.metrics._embed_remote") as mock_embed:
        mock_embed.side_effect = asyncio.TimeoutError()
        with patch.object(driflag, "_ensure_concept_vectors") as mock_init:
            mock_init.return_value = None
            driflag.dependency_vectors = {"test": np.array([1.0, 0.0, 0.0])}
            driflag.isolation_vectors = {"test": np.array([0.0, 1.0, 0.0])}
            result = await driflag.assess(window)
            assert result.passed is True  # Should pass if no scores calculated


# Utility Function Tests
def test_cosine_similarity():
    """Test cosine similarity function."""
    # Test identical vectors
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([1.0, 0.0, 0.0])
    assert abs(_cosine(vec1, vec2) - 1.0) < 1e-6

    # Test orthogonal vectors
    vec3 = np.array([0.0, 1.0, 0.0])
    assert abs(_cosine(vec1, vec3) - 0.0) < 1e-6

    # Test opposite vectors
    vec4 = np.array([-1.0, 0.0, 0.0])
    assert abs(_cosine(vec1, vec4) - (-1.0)) < 1e-6


# Parametrized tests for edge cases
@pytest.mark.parametrize("window_size", [0, 1, 2, 3, 5, 10])
@pytest.mark.asyncio
async def test_td10_various_window_sizes(window_size):
    """Test TD10 behavior with various window sizes."""
    td10 = TD10()
    messages = ["neutral message"] * window_size
    window = deque(messages)

    result = await td10.assess(window)
    # Should pass for small windows or neutral messages
    assert result.passed is True


@pytest.mark.parametrize(
    "mock_vector,expected_pass",
    [
        (np.array([0.9, 0.1, 0.0]), True),  # High similarity to [1,0,0] should pass
        (np.array([0.1, 0.9, 0.0]), False),  # Low similarity to [1,0,0] should fail
        (np.array([0.77, 0.64, 0.0]), True),  # ~0.6 similarity should pass (boundary)
        (np.array([0.4, 0.92, 0.0]), False),  # ~0.4 similarity should fail
    ],
)
@pytest.mark.asyncio
async def test_gcs_threshold_boundaries(goal_vectors, mock_vector, expected_pass):
    """Test GCS behavior at threshold boundaries."""
    gcs = GCS(goal_vectors)
    window = deque(["Test message"])

    with patch("eval.metrics._embed_remote") as mock_embed:
        mock_embed.return_value = mock_vector
        with patch.object(gcs, "_summarize_conversation") as mock_summary:
            mock_summary.return_value = "Test summary"
            result = await gcs.assess(window)
            assert result.passed is expected_pass
