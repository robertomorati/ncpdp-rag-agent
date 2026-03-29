import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


def save_run_log(result: Dict[str, Any], output_path: str = "outputs/run_log.jsonl") -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        **result,
    }

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")