"""Build the offline dashboard using the bundled benchmark measurements and forecasts."""
from pathlib import Path
import json

root = Path(__file__).resolve().parent
template = (root / "dashboard.template.html").read_text(encoding="utf-8")
payload = json.loads((root / "benchmark-data.json").read_text(encoding="utf-8"))
serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<" + chr(92) + "/")
(root / "index.html").write_text(template.replace("__PAYLOAD__", serialized), encoding="utf-8")
print("Dashboard built: index.html")
