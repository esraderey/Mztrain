# NOTICE — Avisos y atribuciones

MZTrain — Motor de Entrenamiento en Espacio Comprimido
© 2025-2026 MSC Star Team (Esraderey, Raúl Cruz Acosta).

Este software se distribuye bajo la licencia **MIT** (ver [`LICENSE`](LICENSE)):
permite uso, copia, modificación, redistribución y uso comercial, con la única
condición de conservar el aviso de copyright y el texto de la licencia. La autoría
se documenta en [`AUTHORSHIP.md`](AUTHORSHIP.md) y las invenciones, como prior art
defensivo, en [`PRIOR_ART.md`](PRIOR_ART.md).

## 1. Componentes de terceros

MZTrain usa, sin incorporar su código fuente, las siguientes bibliotecas en
tiempo de ejecución. Sus licencias permanecen aplicables a sus componentes:

| Componente | Versión mínima | Licencia | Fuente |
|------------|----------------|----------|--------|
| PyTorch | ≥ 2.0 | BSD-3-Clause | https://github.com/pytorch/pytorch |
| NumPy | ≥ 1.21 | BSD-3-Clause | https://numpy.org/ |
| cryptography (opcional, firma del sello) | ≥ 41 | Apache-2.0 / BSD | https://github.com/pyca/cryptography |
| MNEME (opcional) | ≥ 2.0 | Ver licencia MNEME | https://github.com/esraderey/MNEME---Motor-de-Memoria-Neural-M-rfica |

Estos componentes no se redistribuyen como parte de MZTrain; se referencian por
dependencia. Los avisos de copyright de terceros son obligatorios cuando se
redistribuyan sus binarios.

## 2. Benchmarks y checkpoints

Los benchmarks, métricas y resultados en `bench/` y `docs/` se distribuyen bajo la
misma licencia MIT que el código.

Los checkpoints binarios (`zcodebert_*.pt`, `zcoder_1b_*.pt`) son modelos
entrenados por los autores. Los **pesos** pueden estar sujetos a términos propios
(dependen de los datos de entrenamiento) y no forman parte del código MIT; consulte
con los autores antes de asumir su licencia.

## 3. Marcas

"MZTrain", "ElasticRank", "VRAM Governor", "Loss-Guard" y los nombres de
componentes (`ZTrainEngine`, etc.) son marcas de MSC Star Team. La licencia MIT
cubre el **código**, no las **marcas**: su uso se rige por
[`TRADEMARK.md`](TRADEMARK.md). Las marcas de terceros (PyTorch, NumPy, NVIDIA,
CUDA) pertenecen a sus titulares y se mencionan con fines descriptivos.

## 4. Verificación de integridad (opcional)

La versión se acompaña de un sello criptográfico para **comprobar integridad**
(que los archivos no han sido alterados), no como restricción de uso:

- `MANIFEST.sha256` — hashes SHA-256 por archivo.
- `SEAL.json` — doble hash (SHA-256 + SHA-512), raíz de Merkle, timestamp y firma
  Ed25519 opcional.

```bash
python scripts/seal.py verify
```

## 5. Contacto

- Cuestiones sobre el proyecto o las marcas: msc.framework@gmail.com
- Seguridad: ver [`SECURITY.md`](SECURITY.md)
- Contribuciones: ver [`CONTRIBUTING.md`](CONTRIBUTING.md)
