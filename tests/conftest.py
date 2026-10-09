import os
import shutil
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "mini"


@pytest.fixture
def mini_dir() -> Path:
    return FIXTURE


def spark_available() -> bool:
    try:
        import pyspark  # noqa: F401
    except ImportError:
        return False
    return bool(shutil.which("java") or os.environ.get("JAVA_HOME"))


requires_spark = pytest.mark.skipif(not spark_available(), reason="pyspark or a JVM is not available")


@pytest.fixture(scope="session")
def spark():
    if not spark_available():
        pytest.skip("pyspark or a JVM is not available")
    from recsys_lab.spark_job import get_spark

    s = get_spark("recsys-tests")
    yield s
    s.stop()
