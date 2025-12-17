import pytest
import os
import sys

# Smoke Tests for ML Pipeline Infrastructure
# These tests verify the environment and project structure are sound.

def test_environment_setup():
    """Verify that Python environment describes itself correctly"""
    assert sys.version_info.major == 3, "Requires Python 3"

def test_project_structure():
    """Verify key project directories exist"""
    required_dirs = ['api', 'models', 'ml-32m-split']
    for d in required_dirs:
        assert os.path.isdir(d), f"Missing directory: {d}"

def test_ci_cd_placeholder():
    """
    Placeholder for future CI/CD integration.
    This ensures that the test runner (pytest) picks up the suite
    and reports success, verifying the testing pipeline itself works.
    """
    assert True
