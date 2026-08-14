# DECLARACIÓN FORMAL DE AUTORÍA Y TITULARIDAD

**Obra protegida:** MZTrain — Motor de Entrenamiento en Espacio Comprimido
**Versión declarada:** 1.0 (con extensiones ElasticRank v1/v2/v3 y VRAM Governor MVP)
**Fecha de primera publicación:** 2025-01-15
**Fecha de esta declaración:** 2026-05-20
**Titular del copyright:** © 2025-2026 MSC Star Team — Distribuido bajo licencia MIT

---

## 1. Autores y atribución

Esta obra es resultado exclusivo del trabajo intelectual y técnico de las
personas físicas abajo nombradas, integradas bajo el equipo **MSC Star Team**:

| Autor | Rol | Contribuciones principales | Identificación |
|-------|-----|----------------------------|----------------|
| **Esraderey** | Arquitecto principal y co-inventor | Diseño base del motor, factorización SVD entrenable, integración MNEME, motor de entrenamiento `ZTrainEngine`, scheduler de rango | msc.framework@gmail.com |
| **Raúl Cruz Acosta** | Co-inventor y co-desarrollador | Co-diseño del motor, optimizaciones, especificaciones técnicas de ElasticRank (señales híbridas AND, política de redundancia, loss-guard como núcleo de decisión), VRAM Governor, validación experimental | raul.cruz.acosta@example.com |

Ambos autores declaran ser **co-titulares en partes iguales (50/50)** de
todos los derechos patrimoniales y morales sobre la obra, salvo pacto
escrito en contrario y debidamente firmado por ambos.

## 2. Originalidad

Los autores declaran bajo su propia responsabilidad que:

1. La obra es **original e inédita**, fruto de su propia capacidad
   intelectual.
2. No infringe derechos de terceros: las dependencias externas
   (PyTorch, NumPy, MNEME) se utilizan conforme a sus respectivas
   licencias y aparecen documentadas en `NOTICE.md`.
3. **No se ha empleado, total ni parcialmente, código generado por
   sistemas de IA de terceros sin curación, revisión y reescritura
   sustancial por los autores humanos**. Todo asistente automatizado
   utilizado ha actuado bajo supervisión y dirección directa de los
   autores, y las decisiones de diseño relevantes son humanas.
4. Las ideas conceptuales centrales (ver §3) fueron concebidas y
   diseñadas en sesiones de co-diseño documentadas por los autores
   antes de cualquier asistencia automatizada en su implementación.

## 3. Innovaciones reivindicadas (resumen — ver `PRIOR_ART.md` para detalle técnico fechado)

Las siguientes contribuciones técnicas se reivindican como invenciones
**originales** de los autores y constituyen el núcleo defendido por esta
obra. Cada una incluye fecha de concepción / primera implementación
documentada:

1. **Entrenamiento directo en espacio factorizado SVD con rango
   progresivo** (publicado 2025-01-15) — entrenar `U·diag(S)·Vᵀ`
   manteniendo los factores como parámetros optimizables de primer orden,
   en lugar de comprimir un modelo ya entrenado.
2. **ElasticRank — Rango bidireccional con sleep bank en CPU/baja
   precisión** (publicado 2026-05-15) — capacidad de dormir, revivir y
   podar direcciones singulares preservando su estado de optimizador
   (momentum), para acotar memoria sin perder información.
3. **Señal de sleep híbrida con AND de umbrales separados, no producto**
   (publicado 2026-05-15) — evita matar direcciones con bajo score
   espectral pero alto score de actualización.
4. **Señal de redundancia funcional v2: coherencia estadística por
   leverage del Gram de términos rank-1 escalados (ElasticRank v2,
   publicado 2026-05-15)** — `coherence_i = 1 − 1/(R+λI)⁻¹_ii` sobre la
   matriz de correlación de columnas unitarias.
5. **Loss-Guard como núcleo de decisión, no como salvaguarda
   (ElasticRank v3, publicado 2026-05-15)** — toda compactación se
   aplica como tentativa y se revierte estructuralmente
   (factores + buffers + estados de optimizador + contadores +
   cooldowns) si el delta de pérdida supera un umbral.
6. **VRAM Governor con gating de crecimiento de rango bajo presión
   EMA + recuperación de OOM transitorio** (publicado 2026-05-15) —
   validado en GPU real (RTX 4060, 15/15 PASS).
7. **Probe de sensibilidad de pérdida por dirección como diagnóstico
   permanente** (publicado 2026-05-15) — Spearman entre proxies
   weight-space y delta-loss real fue medida ≈ 0.13, lo que motivó la
   adopción del loss-delta medido como único criterio de decisión.
8. **Topology-aware error feedback en compresión de gradientes**
   (publicado 2026-05-16) — los buffers de error feedback se invalidan
   cuando cambia la forma de un factor (rank growth, ElasticRank,
   refactorize).

## 4. Licencia y autoría

- El código se distribuye bajo la licencia **MIT** (ver `LICENSE`), permisiva:
  permite uso, copia, modificación y redistribución conservando el aviso de
  copyright y la licencia. Esta declaración de autoría **no restringe** esos
  derechos; documenta la autoría y la fecha de las invenciones (§3).
- Los autores conservan el **copyright** y el crédito de autoría; MIT exige
  mantener el aviso de copyright en las copias, lo que preserva la atribución.
- Las contribuciones externas se aceptan bajo la misma licencia MIT
  (inbound = outbound); ver `CONTRIBUTING.md`.
- La publicación de las invenciones (§3, `PRIOR_ART.md`) es **prior art
  defensivo** —establece fecha de invención frente a patentes de terceros— y es
  independiente de la licencia MIT del código.

## 5. Verificación e integridad

La integridad de los archivos de esta obra está garantizada por el
sellado criptográfico en `MANIFEST.sha256` y `SEAL.json`. La cadena de
custodia se documenta en `SEAL.json` (raíz de Merkle, timestamps
ISO‑8601, algoritmos SHA‑256 y SHA‑512).

Para verificar:

```bash
python scripts/seal.py verify
```

Cualquier modificación posterior a la fecha de sellado quedará
detectable por divergencia entre el manifiesto y los hashes recalculados.

## 6. Anclaje temporal (timestamping)

Se recomienda anclar este documento, el `LICENSE`, el `PRIOR_ART.md` y
el `SEAL.json` en una o varias de las siguientes vías para prueba
externa de fecha cierta:

- Commit firmado con clave GPG en repositorio público (`git commit -S`).
- Servicio de timestamping RFC 3161 (TSA) sobre el `SEAL.json`.
- Publicación en archivo público con fecha trazable (Internet Archive,
  Software Heritage, Zenodo con DOI).
- Anclaje en blockchain a través de OpenTimestamps (`ots stamp SEAL.json`).

## 7. Firma

Esta declaración se otorga de buena fe por los autores arriba
mencionados, quienes asumen plena responsabilidad por su contenido.

**Esraderey** — Co-titular ___________________________ Fecha: 2026-05-20

**Raúl Cruz Acosta** — Co-titular ___________________________ Fecha: 2026-05-20

---

> Este documento acompaña a MZTrain como declaración de autoría y prior art;
> se recomienda distribuirlo junto con el código. El aviso de copyright debe
> conservarse conforme a la licencia MIT (`LICENSE`).
