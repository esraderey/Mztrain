"""Regenera el staging de la pieza ElasticShape desde el arbol real (D:/mztrain).

El banco copia la PIEZA; la pieza es este staging (src/, tests/, pyproject.toml) porque el
repo completo pesa 2.3 GB en checkpoints/datos que no forman parte del modulo. Toda edicion
va al arbol real; toda verificacion = sync + corrida nueva del banco.

pytest.ini del staging: `pythonpath = src` (importa la copia, no el editable del venv) y sin el
gate --cov-fail-under=85 (deuda preexistente 81.7%, declarada en la orden de trabajo).
"""
import shutil, sys, pathlib
REAL = pathlib.Path("D:/mztrain")
STAGING = pathlib.Path(__file__).resolve().parent / "staging"
if STAGING.exists():
    shutil.rmtree(STAGING)
STAGING.mkdir(parents=True)
shutil.copytree(REAL / "src", STAGING / "src", ignore=shutil.ignore_patterns("__pycache__"))
shutil.copytree(REAL / "tests", STAGING / "tests", ignore=shutil.ignore_patterns("__pycache__"))
shutil.copy2(REAL / "pyproject.toml", STAGING / "pyproject.toml")
shutil.copytree(REAL / "scripts", STAGING / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
for f in ("SEAL.json", "MANIFEST.sha256", "LICENSE", "README.md"):
    if (REAL / f).exists():
        shutil.copy2(REAL / f, STAGING / f)
(STAGING / "pytest.ini").write_text(
    "[pytest]\ntestpaths = tests\npythonpath = src\naddopts = -q --strict-markers -p no:cacheprovider\n"
    "markers =\n    slow: marks tests as slow\n    gpu: marks tests that require GPU\n"
    "    integration: integration\n    unit: unit\n    mneme: requires MNEME\n",
    encoding="utf-8",
)
print("staging listo:", STAGING)
