# NOTICE — Avisos legales y derechos de terceros

MZTrain — Motor de Entrenamiento en Espacio Comprimido
© 2025-2026 MSC Star Team (Esraderey, Raúl Cruz Acosta). Todos los
derechos reservados.

Este producto se distribuye bajo la licencia **MSL-R 1.0** definida
en [`LICENSE`](LICENSE). La autoría se declara formalmente en
[`AUTHORSHIP.md`](AUTHORSHIP.md) y las invenciones reivindicadas en
[`PRIOR_ART.md`](PRIOR_ART.md).

## 1. Reserva expresa de derechos

Todos los derechos no concedidos expresamente en `LICENSE` quedan
**reservados**. La ausencia de mención específica de un uso en
`LICENSE` debe interpretarse como NO autorizada.

## 2. Componentes de terceros

MZTrain hace uso, sin incorporar su código fuente directamente, de
las siguientes bibliotecas en tiempo de ejecución. Sus respectivas
licencias permanecen aplicables a sus respectivos componentes:

| Componente | Versión mínima | Licencia | Fuente |
|------------|----------------|----------|--------|
| PyTorch | ≥ 2.0 | BSD-3-Clause | https://github.com/pytorch/pytorch |
| NumPy | ≥ 1.21 | BSD-3-Clause | https://numpy.org/ |
| MNEME (opcional) | ≥ 2.0 | Ver licencia MNEME | https://github.com/esraderey/MNEME---Motor-de-Memoria-Neural-M-rfica |

Estos componentes **no** se redistribuyen como parte de MZTrain; se
referencian por dependencia. Los avisos de copyright de PyTorch y
NumPy son obligatorios cuando se redistribuyan sus binarios.

## 3. Datos de entrenamiento y benchmarks

Los benchmarks, métricas y resultados experimentales contenidos en
`bench/`, `docs/` y en mensajes de commit son **datos derivados del
trabajo de los autores** y se encuentran cubiertos por la misma
licencia que la Obra Licenciada.

Los checkpoints binarios distribuidos junto al repositorio
(`zcodebert_*.pt`, `zcoder_1b_*.pt`) son modelos entrenados por los
autores con datos de su propiedad o de uso autorizado, y se
distribuyen bajo las mismas condiciones del `LICENSE`.

## 4. Marcas

Los nombres y signos distintivos siguientes son marcas (registradas o
en proceso) de MSC Star Team y se encuentran protegidos bajo
`TRADEMARK.md`:

- **MZTrain®**
- **MSC Star®**
- **ElasticRank™**
- **VRAM Governor™**
- **Loss-Guard™**
- **ZTrain™**, **ZCompressedAdam™**, **ZGradientCompressor™**,
  **ZRankScheduler™**, **ZFactorizedLinear/Attention/TransformerBlock™**

Las marcas de terceros (PyTorch, NumPy, NVIDIA, CUDA, etc.) pertenecen
a sus respectivos titulares y se mencionan únicamente con fines
descriptivos.

## 5. Identificación de versión sellada

La presente versión está sellada criptográficamente. Verifique:

- `MANIFEST.sha256` — lista de archivos y hashes SHA-256.
- `SEAL.json` — manifiesto JSON con doble hash (SHA-256 + SHA-512),
  raíz de Merkle y timestamp ISO-8601.

Cualquier copia de MZTrain cuyo `SEAL.json` no se verifique con
`python scripts/seal.py verify` no es una copia íntegra autorizada.

## 6. Contacto

- Cuestiones legales y de licencias: msc.framework@gmail.com
- Seguridad: ver [`SECURITY.md`](SECURITY.md)
- Contribuciones: ver [`CONTRIBUTING.md`](CONTRIBUTING.md) y
  [`CLA.md`](CLA.md)
