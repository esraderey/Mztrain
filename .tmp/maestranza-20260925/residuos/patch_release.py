"""Release 1.4.0: extra [mneme] -> mnemosys, version, README, CHANGELOG, requirements."""
import pathlib, re

def patch(path, pairs):
    p = pathlib.Path(path); raw = p.read_bytes(); crlf = b"\r\n" in raw
    s = raw.decode("utf-8").replace("\r\n", "\n")
    for old, new in pairs:
        assert s.count(old) == 1, (path, old[:70], s.count(old)); s = s.replace(old, new)
    if crlf: s = s.replace("\n", "\r\n")
    p.write_bytes(s.encode("utf-8")); print("ok", path)

R = "D:/mztrain/"
patch(R + "pyproject.toml", [
 ('version = "1.3.2"', 'version = "1.4.0"'),
 ("""[project.optional-dependencies]
# NOTA: MNEME es un backend opcional que NO se distribuye por PyPI.
# Instalarlo manualmente desde su repositorio (ver README, seccion MNEME).
# No se declara como extra para evitar resolver un paquete ajeno homonimo.
dev = [""",
  """[project.optional-dependencies]
# MNEME se distribuye en PyPI como `mnemosys` (mismos autores) y conserva el import
# historico `import mneme`: pip install "mztrain[mneme]". Ver README, seccion MNEME/MNEMOSYS.
mneme = [
    "mnemosys>=1.0.1",
]
dev = ["""),
])
patch(R + "setup.py", [
 ('version="1.3.2"', 'version="1.4.0"'),
 ('''        "gpu": [
            "torch[cuda]>=2.0.0",
        ],
        "all": [''',
  '''        "gpu": [
            "torch[cuda]>=2.0.0",
        ],
        "mneme": [
            "mnemosys>=1.0.1",
        ],
        "all": ['''),
])
patch(R + "src/mztrain/__init__.py", [('__version__ = "1.3.2"', '__version__ = "1.4.0"')])
patch(R + "requirements.txt", [
 ("# MNEME - Opcional (para compresion avanzada de activaciones)\n# mneme>=2.0.0\n",
  "# MNEME - Opcional. En PyPI se llama `mnemosys` (import `mneme`): pip install \"mztrain[mneme]\"\n# mnemosys>=1.0.1\n"),
])
patch(R + "README.md", [
 ("""```bash
pip install mztrain            # desde PyPI
```
""",
  """```bash
pip install mztrain            # desde PyPI
pip install "mztrain[mneme]"   # + MNEME (paquete PyPI `mnemosys`, import `mneme`)
```
"""),
 ("""### MNEME (backend opcional de almacenamiento/cuantización)

MNEME **no se distribuye por PyPI**; instálalo manualmente desde
[su repositorio](https://github.com/esraderey/MNEME---Motor-de-Memoria-Neural-M-rfica)
si quieres el store de activaciones ZSpace o el despliegue GPTQ INT4. Sin MNEME,
MZTrain usa su fallback INT8 integrado (acotado y verificado).
""",
  """### MNEME / MNEMOSYS (backend opcional de almacenamiento y cuantización)

MNEME se distribuye en PyPI como [`mnemosys`](https://pypi.org/project/mnemosys/)
(mismos autores; el paquete conserva el import histórico `import mneme`). Con
`pip install "mztrain[mneme]"` MZTrain lo detecta y habilita:

- **Compresión post-entrenamiento** (`compress_model`, GPTQ INT4) y **almacenamiento
  seguro** (`SecureStorageBackend`) en `ZCodeBERT` y en los ejemplos.
- **Store de activaciones ZSpace** para `ZActivationCheckpoint`, **opt-in** con
  `ZTrainConfig(mneme_activation_store=True)`. Medido con `mnemosys` 1.0.1: fidelidad
  INT8 (error relativo ~6e-3, MZTrain fuerza `quantization_type="int8"` porque el
  enrutado automático descompone tensorialmente las activaciones grandes) y un coste de
  ~0.2-0.6 s por activación porque persiste cada tensor en disco (directorio temporal
  propio, sin cifrado en reposo). Por eso el valor por defecto sigue siendo el INT8
  interno (milisegundos). MZTrain exige al backend evicción por clave (`delete`/`remove`)
  y hace una sonda de fidelidad en el primer uso; si algo falla, avisa y cae al INT8
  interno en vez de fugar memoria.

Sin `mnemosys`, MZTrain usa su fallback INT8 integrado (acotado y verificado). El
[repositorio de MNEME](https://github.com/esraderey/mnemosys) es la otra mitad del
pipeline: MZTrain entrena por rango, MNEME comprime por bits.
"""),
 ("""- [MNEME — Motor de Memoria Neural Mórfica](https://github.com/esraderey/MNEME---Motor-de-Memoria-Neural-M-rfica):
  la otra mitad del pipeline — MZTrain entrena por rango, MNEME comprime por bits""",
  """- [MNEME / MNEMOSYS — Motor de Memoria Neural Mórfica](https://github.com/esraderey/mnemosys)
  ([PyPI: `mnemosys`](https://pypi.org/project/mnemosys/), `pip install "mztrain[mneme]"`):
  la otra mitad del pipeline — MZTrain entrena por rango, MNEME comprime por bits"""),
])
patch(R + "CHANGELOG.md", [
 ("## [Sin publicar] - 2026-09-07\n",
  """## [1.4.0] - 2026-09-26

### MNEMOSYS desde PyPI
- MNEME ya se instala desde PyPI como `mnemosys` (import `mneme`): nuevo extra `pip install "mztrain[mneme]"`
  (`mnemosys>=1.0.1`). `ZActivationCheckpoint` gana un adaptador de backend: fuerza `quantization_type="int8"`
  al registrar (el enrutado automatico de mnemosys descompone tensorialmente las activaciones grandes: error
  relativo ~0.9 medido; con INT8 ~6e-3), tolera `load()` sin kwarg `device` y restaura el device original,
  exige evicción por clave (`delete`/`remove`) y hace una sonda de fidelidad en el primer uso; si el backend no
  cumple, avisa con `RuntimeWarning` y usa el fallback INT8 interno en vez de fugar memoria.
- El store ZSpace pasa a ser **opt-in** (`ZTrainConfig.mneme_activation_store=False` por defecto): con
  mnemosys 1.0.1 cuesta ~0.2-0.6 s por activacion porque persiste cada tensor en disco (MZTrain usa un directorio
  temporal propio y desactiva el cifrado en reposo para ese store efimero). Cierra la fuga de activaciones
  observada con backends sin API de evicción (`example_zcodebert` llegaba a OOM tras 57 min).

### ElasticShape: cirugia reversible y migracion AdamW
"""),
])
print("release patch aplicado")
