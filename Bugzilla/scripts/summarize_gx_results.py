## ERA ASSENTE


from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

GX_JSON_PATH = Path("reports/gx_train_validation_result.json")
GX_MD_SUMMARY_PATH = Path("reports/gx_train_validation_summary.md")


def load_gx_results(path: Path = GX_JSON_PATH) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"GE JSON report not found at {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_markdown_summary(results: Dict[str, Any]) -> str:
    stats = results.get("statistics", {})
    success = results.get("success", False)
    expectations_results: List[Dict[str, Any]] = results.get("results", [])

    total = stats.get("evaluated_expectations", len(expectations_results))
    successful = stats.get("successful_expectations")

    lines: List[str] = []

    lines.append("# Great Expectations – Train data validation\n")

    lines.append(f"- Overall success: {'✅ PASSED' if success else '❌ FAILED'}")

    if successful is not None and total is not None:
        lines.append(f"- Expectations passed: {successful}/{total}")
    elif total is not None:
        lines.append(f"- Expectations evaluated: {total}")

    lines.append("\n## Expectations\n")

    for r in expectations_results:
        exp_type = r.get("expectation_type", "unknown_expectation")
        kwargs = r.get("kwargs", {})
        column = kwargs.get("column")
        success_flag = r.get("success", False)
        status_emoji = "✅" if success_flag else "❌"

        if column is not None:
            label = f"{status_emoji} `{exp_type}` on column `{column}`"
        else:
            label = f"{status_emoji} `{exp_type}`"

        lines.append(f"- {label}")

    lines.append("")
    return "\n".join(lines)


def save_markdown(text: str, path: Path = GX_MD_SUMMARY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)
    print(f"[summarize_gx] Saved markdown summary to {path}")


def main() -> None:
    results = load_gx_results()
    md = build_markdown_summary(results)
    save_markdown(md)


if __name__ == "__main__":
    main()
