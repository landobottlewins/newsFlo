"""Tests for package importability and health-check execution."""

import subprocess
import sys

import recommender
from recommender.__main__ import main


def test_package_import():
    """Verify the package can be imported and exposes its version."""
    assert hasattr(recommender, "__version__")
    assert isinstance(recommender.__version__, str)


def test_submodules_import():
    """Verify subpackages can be imported."""
    import recommender.evaluation
    import recommender.ingestion
    import recommender.models
    import recommender.processing
    import recommender.recommendation
    import recommender.users

    assert recommender.models is not None
    assert recommender.ingestion is not None
    assert recommender.processing is not None
    assert recommender.recommendation is not None
    assert recommender.users is not None
    assert recommender.evaluation is not None


def test_main_health_check_function():
    """Verify main() function runs and returns exit code 0."""
    result = main()
    assert result == 0


def test_main_cli_execution():
    """Verify python -m recommender executes successfully via subprocess."""
    result = subprocess.run(
        [sys.executable, "-m", "recommender"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    output = result.stdout + result.stderr
    assert "Recommender system initialized" in output
