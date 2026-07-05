import json
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from tradingagents_adapter import SUBPROCESS_JSON_PREFIX, run_tradingagents_analysis  # noqa: E402
from tradingagents_models import TradingAgentsAnalysisRequest  # noqa: E402


def main() -> int:
    payload = json.loads(sys.stdin.read())
    env_path = Path(payload.pop("env_path"))
    repo_path = Path(payload.pop("repo_path"))
    response = run_tradingagents_analysis(
        TradingAgentsAnalysisRequest(**payload),
        env_path=env_path,
        repo_path=repo_path,
    )
    print(SUBPROCESS_JSON_PREFIX + response.model_dump_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
