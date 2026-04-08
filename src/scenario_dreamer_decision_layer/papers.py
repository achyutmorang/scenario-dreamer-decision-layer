from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .config import project_root

ARXIV_IDS = [
    "2503.22496",
    "2403.19918",
    "2310.08710",
    "2305.12032",
    "2405.15677",
    "2408.01584",
    "2408.16375",
    "2408.15538",
    "2405.17372"
]

WHY = {
    "2503.22496": "Primary simulator stack and baseline environment-generation paper.",
    "2403.19918": "Frozen pretrained behaviour model used as the baseline backbone.",
    "2310.08710": "Waymax reference for later transfer-check design and metric framing.",
    "2305.12032": "Official WOSAC evaluation and realism-metric reference point.",
    "2405.15677": "Previous SMART baseline line; useful contrast with the new stack choice.",
    "2408.01584": "Fast simulator reference and adjacent environment stack for planning research.",
    "2408.16375": "Waymax planning baseline with evaluation-robustness perspective.",
    "2408.15538": "Controllable safety-critical traffic simulation relevant to scenario-conditioned methods.",
    "2405.17372": "Strong compact WOSAC baseline family adjacent to decision-layer work."
}


def references_dir() -> Path:
    return project_root() / "references" / "papers"


def manifest_path() -> Path:
    return project_root() / "references" / "papers_manifest.json"


def _api_url(ids: Iterable[str]) -> str:
    return "https://export.arxiv.org/api/query?id_list=" + ",".join(ids)


def fetch_arxiv_metadata(ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    with urllib.request.urlopen(_api_url(ids), timeout=30) as response:
        root = ET.fromstring(response.read())
    ns = {"a": "http://www.w3.org/2005/Atom"}
    data: Dict[str, Dict[str, Any]] = {}
    for entry in root.findall("a:entry", ns):
        abs_url = entry.find("a:id", ns).text
        arxiv_id = abs_url.rsplit("/", 1)[-1].split("v", 1)[0]
        title = " ".join(entry.find("a:title", ns).text.split())
        published = entry.find("a:published", ns).text
        authors = [a.find("a:name", ns).text for a in entry.findall("a:author", ns)]
        data[arxiv_id] = {
            "title": title,
            "authors": authors,
            "year": int(published[:4]),
            "source_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            "arxiv_abs": f"https://arxiv.org/abs/{arxiv_id}",
        }
    return data


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(arxiv_id: str, title: str) -> str:
    stem = title.lower().replace(":", "").replace(",", "").replace("/", "-")
    stem = "_".join(stem.split())[:90]
    return f"{arxiv_id}_{stem}.pdf"


def download_curated_papers(force: bool = False) -> List[Dict[str, Any]]:
    refs = references_dir()
    refs.mkdir(parents=True, exist_ok=True)
    metadata = fetch_arxiv_metadata(ARXIV_IDS)
    manifest: List[Dict[str, Any]] = []
    for arxiv_id in ARXIV_IDS:
        item = metadata[arxiv_id]
        filename = _safe_name(arxiv_id, item["title"])
        destination = refs / filename
        if force or not destination.exists():
            request = urllib.request.Request(item["source_url"], headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                destination.write_bytes(response.read())
        manifest.append({
            "id": arxiv_id,
            "title": item["title"],
            "authors": item["authors"],
            "year": item["year"],
            "source_url": item["source_url"],
            "local_path": str(destination.relative_to(project_root())),
            "sha256": _sha256_file(destination),
            "why_it_matters": WHY[arxiv_id],
        })
    manifest_path().write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def validate_manifest() -> Dict[str, Any]:
    payload = json.loads(manifest_path().read_text(encoding="utf-8"))
    missing = []
    mismatched = []
    for paper in payload:
        local = project_root() / paper["local_path"]
        if not local.exists():
            missing.append(paper["local_path"])
            continue
        sha = _sha256_file(local)
        if sha != paper["sha256"]:
            mismatched.append({"path": paper["local_path"], "expected": paper["sha256"], "actual": sha})
    return {"count": len(payload), "missing": missing, "mismatched": mismatched}
