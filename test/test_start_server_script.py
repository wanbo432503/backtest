import os
import subprocess
from pathlib import Path


def test_start_server_script_restarts_port_before_starting_app():
    script = Path("scripts/start_server.sh")

    assert script.exists()
    assert os.access(script, os.X_OK)

    content = script.read_text(encoding="utf-8")
    assert 'PORT="${PORT:-8005}"' in content
    assert "lsof -tiTCP" in content
    assert '"$PYTHON_BIN" -m uvicorn main:app' in content

    subprocess.run(["bash", "-n", str(script)], check=True)
