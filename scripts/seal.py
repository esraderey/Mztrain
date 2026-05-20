#!/usr/bin/env python3
"""
MZTrain - Sello criptografico de la obra.

Recorre el arbol del proyecto, calcula SHA-256 y SHA-512 de cada fichero
relevante (codigo fuente, documentacion legal, tests, benchmarks,
configuracion de build), y produce dos artefactos:

  - MANIFEST.sha256  : formato compatible con `sha256sum -c`
  - SEAL.json        : manifiesto JSON con doble hash, raiz de Merkle,
                       timestamp ISO-8601 UTC y metadatos de autoria.

Modos:
  seal   : genera MANIFEST.sha256 + SEAL.json (sobrescribe el anterior).
  verify : recalcula y compara contra el SEAL.json existente, devolviendo
           exit code 0 si todo coincide, !=0 si hay divergencias.
  print  : imprime el SEAL.json existente.

Uso:
  python scripts/seal.py seal
  python scripts/seal.py verify
  python scripts/seal.py print

Disenado para ser dependencia-cero (solo stdlib) para que el sello
pueda recalcularse en cualquier maquina con Python 3.8+.

(c) 2025-2026 MSC Star Team. Distribuido bajo MSL-R 1.0.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# ----------------------------------------------------------------------
# Configuracion: que se incluye y que se excluye del sellado
# ----------------------------------------------------------------------

# Extensiones de fichero a sellar (texto / codigo / config / docs)
INCLUDE_SUFFIXES = {
    ".py", ".pyi", ".md", ".rst", ".txt", ".toml", ".yaml", ".yml",
    ".json", ".cfg", ".ini", ".in", ".sh", ".ps1", ".bat",
}

# Ficheros sin extension pero con nombre exacto que SI se sellan
INCLUDE_NAMES = {
    "LICENSE", "NOTICE", "CODEOWNERS", "MANIFEST.in", ".gitignore",
}

# Ficheros legales / sellados que SIEMPRE se incluyen (ruta exacta)
LEGAL_FILES = [
    "LICENSE", "AUTHORSHIP.md", "NOTICE.md", "PRIOR_ART.md",
    "CLA.md", "TRADEMARK.md", "SECURITY.md", "CODEOWNERS",
    "README.md", "CHANGELOG.md", "CONTRIBUTING.md",
]

# Directorios que NUNCA se recorren
EXCLUDE_DIRS = {
    ".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".tox", ".benchmarks", ".tmp",
    "htmlcov", "build", "dist", ".eggs", "node_modules",
    "mneme_storage", "mneme_storage_zcoder1b", "data",
}

# Ficheros que NUNCA se incluyen (artefactos efimeros, sellos propios,
# o pesos binarios que tendrian su propio manifest)
EXCLUDE_FILES = {
    "MANIFEST.sha256", "SEAL.json", ".coverage", "coverage.xml",
    "analysis_results.json",
}

# Patrones de sufijo binario que no se sellan en este manifest
# (los checkpoints .pt se sellan por separado si se desea)
EXCLUDE_SUFFIXES = {
    ".pt", ".pth", ".bin", ".onnx", ".pyc", ".so", ".dll", ".dylib",
    ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".gz", ".tar",
    ".whl", ".egg",
}


# ----------------------------------------------------------------------
# Funciones de hashing
# ----------------------------------------------------------------------

def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _sha512_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha512()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _merkle_root(leaf_hex_hashes: List[str]) -> str:
    """Raiz de Merkle binaria sobre SHA-256 (duplica el ultimo nodo si impar)."""
    if not leaf_hex_hashes:
        return hashlib.sha256(b"").hexdigest()
    layer = [bytes.fromhex(h) for h in sorted(leaf_hex_hashes)]
    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])
        nxt = []
        for i in range(0, len(layer), 2):
            nxt.append(hashlib.sha256(layer[i] + layer[i + 1]).digest())
        layer = nxt
    return layer[0].hex()


# ----------------------------------------------------------------------
# Recorrido del arbol
# ----------------------------------------------------------------------

def _should_include(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    parts = rel.parts
    # excluye por directorio
    for part in parts[:-1]:
        if part in EXCLUDE_DIRS:
            return False
    name = path.name
    if name in EXCLUDE_FILES:
        return False
    if path.suffix.lower() in EXCLUDE_SUFFIXES:
        return False
    if path.suffix.lower() in INCLUDE_SUFFIXES:
        return True
    if name in INCLUDE_NAMES:
        return True
    return False


def _iter_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        # poda directorios in-place
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fn in filenames:
            p = Path(dirpath) / fn
            if _should_include(p, root):
                yield p


def _hash_all(root: Path) -> List[Tuple[str, str, str, int]]:
    """Devuelve lista de (rel_path_posix, sha256, sha512, size_bytes)."""
    out = []
    for p in _iter_files(root):
        rel = p.relative_to(root).as_posix()
        sha256 = _sha256_file(p)
        sha512 = _sha512_file(p)
        size = p.stat().st_size
        out.append((rel, sha256, sha512, size))
    out.sort(key=lambda r: r[0])
    return out


# ----------------------------------------------------------------------
# Generacion / verificacion
# ----------------------------------------------------------------------

SEAL_VERSION = "1.0"


def _author_block() -> Dict[str, object]:
    return {
        "team": "MSC Star Team",
        "authors": [
            {"name": "Esraderey",
             "role": "co-titular y co-inventor",
             "contact": "msc.framework@gmail.com"},
            {"name": "Raul Cruz Acosta",
             "role": "co-titular y co-inventor",
             "contact": "raul.cruz.acosta@example.com"},
        ],
        "copyright": "(c) 2025-2026 MSC Star Team. Todos los derechos reservados.",
        "license": "MSL-R 1.0 (ver LICENSE)",
    }


def _seal(root: Path) -> Tuple[Dict[str, object], str]:
    entries = _hash_all(root)
    files_json = [
        {"path": rel, "sha256": h256, "sha512": h512, "size": size}
        for (rel, h256, h512, size) in entries
    ]
    leaf_hashes = [e["sha256"] for e in files_json]
    merkle = _merkle_root(leaf_hashes)

    # contenido canonico para hash global (sin self-referencia al campo seal_hash)
    canonical_files = json.dumps(files_json, sort_keys=True, ensure_ascii=False)
    global_sha256 = hashlib.sha256(canonical_files.encode("utf-8")).hexdigest()
    global_sha512 = hashlib.sha512(canonical_files.encode("utf-8")).hexdigest()

    seal = {
        "seal_version": SEAL_VERSION,
        "work": {
            "name": "MZTrain",
            "long_name": "Motor de Entrenamiento en Espacio Comprimido",
            "version": "1.0",
            "package": "mztrain",
        },
        "owner": _author_block(),
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tool": {
            "name": "scripts/seal.py",
            "python": sys.version.split()[0],
            "platform": sys.platform,
        },
        "algorithms": ["SHA-256", "SHA-512", "Merkle (SHA-256)"],
        "file_count": len(files_json),
        "total_bytes": sum(e["size"] for e in files_json),
        "merkle_root_sha256": merkle,
        "files_canonical_sha256": global_sha256,
        "files_canonical_sha512": global_sha512,
        "files": files_json,
        "verification_command": "python scripts/seal.py verify",
        "notice": (
            "Cualquier modificacion posterior a la fecha 'generated_at_utc' "
            "sera detectable por divergencia entre el manifesto y los hashes "
            "recalculados. Este sello no sustituye a un timestamp RFC 3161 "
            "ni a un anclaje en blockchain; se recomienda combinar ambos."
        ),
    }
    return seal, merkle


def cmd_seal(root: Path) -> int:
    seal, merkle = _seal(root)

    # Escribir SEAL.json (indentado, UTF-8, sin BOM)
    seal_path = root / "SEAL.json"
    seal_path.write_text(
        json.dumps(seal, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Escribir MANIFEST.sha256 compatible con `sha256sum -c`
    manifest_path = root / "MANIFEST.sha256"
    lines = [
        "# MZTrain MANIFEST.sha256",
        "# (c) 2025-2026 MSC Star Team. Distribuido bajo MSL-R 1.0.",
        f"# generated_at_utc = {seal['generated_at_utc']}",
        f"# merkle_root_sha256 = {merkle}",
        f"# file_count = {seal['file_count']}",
        "#",
        "# Verificacion: python scripts/seal.py verify",
        "# (o, en Linux/macOS: sha256sum -c MANIFEST.sha256)",
        "",
    ]
    for entry in seal["files"]:
        lines.append(f"{entry['sha256']}  {entry['path']}")
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[seal] {seal['file_count']} ficheros, "
          f"{seal['total_bytes']} bytes")
    print(f"[seal] merkle_root_sha256 = {merkle}")
    print(f"[seal] generated_at_utc   = {seal['generated_at_utc']}")
    print(f"[seal] escrito           : {seal_path.relative_to(root)}")
    print(f"[seal] escrito           : {manifest_path.relative_to(root)}")
    return 0


def cmd_verify(root: Path) -> int:
    seal_path = root / "SEAL.json"
    if not seal_path.exists():
        print(f"[verify] ERROR: no existe {seal_path}", file=sys.stderr)
        return 2
    expected = json.loads(seal_path.read_text(encoding="utf-8"))
    current_entries = _hash_all(root)
    current_by_path = {rel: (h256, h512, size)
                       for (rel, h256, h512, size) in current_entries}

    expected_by_path = {e["path"]: (e["sha256"], e["sha512"], e["size"])
                        for e in expected["files"]}

    ok = True
    missing = []
    modified = []
    added = []

    for path, (h256, h512, size) in expected_by_path.items():
        if path not in current_by_path:
            ok = False
            missing.append(path)
            continue
        ch256, ch512, csize = current_by_path[path]
        if (ch256, ch512) != (h256, h512):
            ok = False
            modified.append(path)

    for path in current_by_path:
        if path not in expected_by_path:
            ok = False
            added.append(path)

    leaf_hashes = [h for (_p, h, _h2, _s) in current_entries]
    cur_merkle = _merkle_root(leaf_hashes)
    if cur_merkle != expected.get("merkle_root_sha256"):
        ok = False

    print(f"[verify] sello: {expected.get('generated_at_utc')}")
    print(f"[verify] merkle esperado : {expected.get('merkle_root_sha256')}")
    print(f"[verify] merkle actual   : {cur_merkle}")
    print(f"[verify] file_count esperado : {expected.get('file_count')}")
    print(f"[verify] file_count actual   : {len(current_entries)}")

    if missing:
        print(f"[verify] FALTAN ({len(missing)}):")
        for p in missing:
            print(f"   - {p}")
    if modified:
        print(f"[verify] MODIFICADOS ({len(modified)}):")
        for p in modified:
            print(f"   ~ {p}")
    if added:
        print(f"[verify] AGREGADOS ({len(added)}):")
        for p in added:
            print(f"   + {p}")

    if ok:
        print("[verify] OK: la obra coincide con el sello.")
        return 0
    print("[verify] FAIL: la obra NO coincide con el sello.")
    return 1


def cmd_print(root: Path) -> int:
    seal_path = root / "SEAL.json"
    if not seal_path.exists():
        print(f"[print] ERROR: no existe {seal_path}", file=sys.stderr)
        return 2
    print(seal_path.read_text(encoding="utf-8"))
    return 0


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Sello criptografico de MZTrain (SHA-256 + SHA-512 + Merkle)."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("seal", help="generar MANIFEST.sha256 + SEAL.json")
    sub.add_parser("verify", help="verificar el sello actual")
    sub.add_parser("print", help="imprimir SEAL.json")
    parser.add_argument(
        "--root", default=None,
        help="Raiz del proyecto (default: dos niveles arriba de este script).",
    )
    args = parser.parse_args(argv)

    if args.root:
        root = Path(args.root).resolve()
    else:
        root = Path(__file__).resolve().parent.parent

    if args.cmd == "seal":
        return cmd_seal(root)
    if args.cmd == "verify":
        return cmd_verify(root)
    if args.cmd == "print":
        return cmd_print(root)
    parser.error(f"comando desconocido: {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
