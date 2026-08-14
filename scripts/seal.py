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

(c) 2025-2026 MSC Star Team. Distribuido bajo licencia MIT.
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

# Firma Ed25519 opcional: prueba de AUTORIA (no solo integridad). Si
# 'cryptography' no esta instalada, el sello degrada a solo-hash y los
# comandos de firma avisan en vez de romper.
try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    HAS_CRYPTO = True
except ImportError:  # pragma: no cover - depende del entorno
    HAS_CRYPTO = False

SIGNATURE_ALGORITHM = "Ed25519"

# Claves publicas de confianza (hex de la clave Ed25519 raw, 32 bytes = 64 hex).
# El autor genera su par con `seal.py keygen`, PUBLICA esta clave publica (aqui
# y por un canal externo: README/web) y guarda la privada FUERA del repo. Con la
# lista no vacia, `verify` exige que el sello este firmado por una de estas
# claves; asi un atacante que re-firme con su propia clave es rechazado.
TRUSTED_PUBLIC_KEYS: List[str] = [
    "1c1765ef1730ea1173e33cebd6a909c86ae8de5dba326e60fb6e6ee5d44a4eb9",
]

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

# Directorios basura que NUNCA se recorren, a cualquier profundidad
# (siempre son artefactos, su nombre no colisiona con codigo fuente).
EXCLUDE_DIRS = {
    ".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".tox", ".benchmarks", ".tmp",
    "htmlcov", "build", "dist", ".eggs", "node_modules",
}

# Directorios de datos excluidos SOLO en la raiz del proyecto. Su nombre puede
# coincidir con un paquete de codigo fuente legitimo en profundidad
# (p.ej. src/mztrain/data/), que SI debe sellarse: excluirlos por nombre a
# cualquier nivel dejaba el tokenizador y el dataset fuera del sello.
EXCLUDE_ROOT_DIRS = {
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
# Firma Ed25519 (autoria)
# ----------------------------------------------------------------------

def _generate_keypair(priv_path: Path) -> str:
    """Generar un par Ed25519, guardar la privada PEM y devolver la publica hex."""
    priv = Ed25519PrivateKey.generate()
    pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    priv_path.write_bytes(pem)
    try:
        os.chmod(priv_path, 0o600)  # best-effort (POSIX); en Windows es no-op
    except OSError:  # pragma: no cover - depende del SO/FS
        pass
    pub_raw = priv.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    return pub_raw.hex()


_SIGNATURE_FIELDS = ("signature", "signature_public_key", "signature_algorithm")


def _seal_signing_bytes(seal: Dict[str, object]) -> bytes:
    """Bytes canonicos del sello EXCLUYENDO los campos de firma: es lo que se
    firma, de modo que la firma cubre TODO el manifiesto (autores, fecha, work,
    files y su merkle), no solo el merkle root."""
    payload = {k: v for k, v in seal.items() if k not in _SIGNATURE_FIELDS}
    return json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")


def _sign_bytes(message: bytes, priv_path: Path) -> Tuple[str, str]:
    """Firmar un mensaje con la clave privada; devolver (sig_hex, pub_hex)."""
    priv = serialization.load_pem_private_key(priv_path.read_bytes(), password=None)
    if not isinstance(priv, Ed25519PrivateKey):
        raise ValueError("la clave privada no es Ed25519")
    sig = priv.sign(message)
    pub_raw = priv.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    return sig.hex(), pub_raw.hex()


def _verify_sig(message: bytes, sig_hex: str, pub_hex: str) -> bool:
    """Verificar una firma Ed25519 sobre 'message' con la clave publica dada."""
    try:
        pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pub_hex))
        pub.verify(bytes.fromhex(sig_hex), message)
        return True
    except Exception:
        return False


# ----------------------------------------------------------------------
# Recorrido del arbol
# ----------------------------------------------------------------------

def _should_include(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    parts = rel.parts
    # excluye por nombre de directorio basura a cualquier profundidad
    for part in parts[:-1]:
        if part in EXCLUDE_DIRS:
            return False
    # excluye directorios de datos SOLO si estan en la raiz (primer componente)
    if len(parts) > 1 and parts[0] in EXCLUDE_ROOT_DIRS:
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
        # poda in-place: directorios basura a cualquier profundidad; los
        # directorios de datos solo en la raiz del proyecto.
        at_root = Path(dirpath) == root
        dirnames[:] = [
            d for d in dirnames
            if d not in EXCLUDE_DIRS and not (at_root and d in EXCLUDE_ROOT_DIRS)
        ]
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
        "copyright": "(c) 2025-2026 MSC Star Team.",
        "license": "MIT (ver LICENSE)",
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


def _write_artifacts(root: Path, seal: Dict[str, object], merkle: str) -> None:
    """Escribir SEAL.json + MANIFEST.sha256 (compartido por seal y sign)."""
    seal_path = root / "SEAL.json"
    seal_path.write_text(
        json.dumps(seal, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    manifest_path = root / "MANIFEST.sha256"
    lines = [
        "# MZTrain MANIFEST.sha256",
        "# (c) 2025-2026 MSC Star Team. Distribuido bajo licencia MIT.",
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


def cmd_seal(root: Path) -> int:
    seal, merkle = _seal(root)
    _write_artifacts(root, seal, merkle)
    return 0


def cmd_keygen(out_path: Path) -> str:
    """Generar un par Ed25519; guardar la privada y devolver la publica hex."""
    if not HAS_CRYPTO:
        print("[keygen] ERROR: falta 'cryptography' (pip install cryptography)",
              file=sys.stderr)
        return ""
    pub_hex = _generate_keypair(out_path)
    print(f"[keygen] clave privada escrita : {out_path}")
    print(f"[keygen] clave PUBLICA (hex)   : {pub_hex}")
    print("[keygen] Anade esa clave publica a TRUSTED_PUBLIC_KEYS en scripts/seal.py")
    print("[keygen] y publicala por un canal externo (README/web). GUARDA la privada")
    print("[keygen] FUERA del repositorio; nunca la subas al control de versiones.")
    return pub_hex


def cmd_sign(root: Path, key_path: Path) -> int:
    """Generar el sello y firmarlo (Ed25519) con la clave privada del autor."""
    if not HAS_CRYPTO:
        print("[sign] ERROR: falta 'cryptography' (pip install cryptography)",
              file=sys.stderr)
        return 2
    if not key_path.exists():
        print(f"[sign] ERROR: no existe la clave privada {key_path}", file=sys.stderr)
        return 2
    seal, merkle = _seal(root)
    # Firmar el sello COMPLETO (sin los campos de firma), no solo el merkle:
    # asi la firma cubre autores, fecha y work, no solo el contenido.
    sig_hex, pub_hex = _sign_bytes(_seal_signing_bytes(seal), key_path)
    seal["signature_algorithm"] = SIGNATURE_ALGORITHM
    seal["signature_public_key"] = pub_hex
    seal["signature"] = sig_hex
    _write_artifacts(root, seal, merkle)
    print(f"[sign] firmado {SIGNATURE_ALGORITHM}; clave publica: {pub_hex}")
    if TRUSTED_PUBLIC_KEYS and pub_hex not in TRUSTED_PUBLIC_KEYS:
        print("[sign] AVISO: la clave publica no esta en TRUSTED_PUBLIC_KEYS; "
              "verify la rechazara hasta que la anadas.")
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

    for path, (h256, h512, _size) in expected_by_path.items():
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

    # Verificar el hash canonico global (antes se publicaba sin comprobarse).
    cur_files_json = [
        {"path": rel, "sha256": h256, "sha512": h512, "size": size}
        for (rel, h256, h512, size) in current_entries
    ]
    cur_canonical = json.dumps(cur_files_json, sort_keys=True, ensure_ascii=False)
    cur_canonical_sha256 = hashlib.sha256(cur_canonical.encode("utf-8")).hexdigest()
    if cur_canonical_sha256 != expected.get("files_canonical_sha256"):
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

    # Verificar la firma de autoria (Ed25519). Con TRUSTED_PUBLIC_KEYS poblada
    # el modo es ESTRICTO: la firma es OBLIGATORIA y debe ser de una clave de
    # confianza; esto cierra el downgrade de re-sellar sin firma.
    sig = expected.get("signature")
    sig_pub = expected.get("signature_public_key")
    strict = bool(TRUSTED_PUBLIC_KEYS)

    if sig and sig_pub:
        if not HAS_CRYPTO:
            # fail-closed: el sello dice estar firmado pero no podemos verificarlo
            ok = False
            print("[verify] FALLO: el sello esta firmado pero falta 'cryptography' "
                  "para verificar la firma.")
        elif not _verify_sig(_seal_signing_bytes(expected), sig, sig_pub):
            ok = False
            print("[verify] FALLO: firma invalida sobre el sello.")
        elif strict and sig_pub not in TRUSTED_PUBLIC_KEYS:
            ok = False
            print(f"[verify] FALLO: firmado por una clave publica NO confiable "
                  f"({sig_pub[:16]}...).")
        else:
            trust = ("clave de confianza" if strict
                     else "clave NO anclada a confianza: TRUSTED_PUBLIC_KEYS vacia")
            print(f"[verify] firma {expected.get('signature_algorithm', '?')} "
                  f"valida ({trust}).")
    elif strict:
        # sin firma pero hay lista de confianza -> exigirla (anti-downgrade)
        ok = False
        print("[verify] FALLO: el sello no lleva firma y TRUSTED_PUBLIC_KEYS exige "
              "una firma de clave de confianza (posible downgrade).")
    else:
        print("[verify] NOTA: el sello no lleva firma de autoria (solo hashes).")

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
    p_sign = sub.add_parser("sign", help="generar el sello y firmarlo (Ed25519)")
    p_sign.add_argument("--key", required=True,
                        help="ruta a la clave privada Ed25519 (PEM)")
    p_keygen = sub.add_parser("keygen", help="generar un par de claves Ed25519")
    p_keygen.add_argument("--out", required=True,
                          help="ruta de salida de la clave privada (PEM)")
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
    if args.cmd == "sign":
        return cmd_sign(root, Path(args.key).resolve())
    if args.cmd == "keygen":
        return 0 if cmd_keygen(Path(args.out).resolve()) else 2
    parser.error(f"comando desconocido: {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
