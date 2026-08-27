"""Machine-Readable JSON Report Serializer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
from promptdiff.core.models import DiffReport


def generate_json_report(report: DiffReport, output_path: Optional[str] = None) -> str:
    """Serialize DiffReport to formatted JSON."""
    raw_dict = report.model_dump()
    json_str = json.dumps(raw_dict, indent=2)

    if output_path:
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json_str, encoding="utf-8")

    return json_str
