# Política de Seguridad — MZTrain

## Reporte responsable

Si descubre una vulnerabilidad de seguridad en MZTrain, **NO** la haga
pública en GitHub Issues, foros, redes sociales ni publicaciones
técnicas hasta haber permitido un plazo razonable de coordinación.

### Cómo reportar

Envíe un correo cifrado o claro a:

- **msc.framework@gmail.com** (asunto: `[SECURITY] MZTrain`)

Incluya, en la medida de lo posible:

1. Descripción técnica del problema.
2. Pasos para reproducirlo.
3. Versión afectada (commit hash y `SEAL.json`).
4. Impacto estimado y propuesta de mitigación, si la tiene.
5. Sus datos de contacto y si desea ser acreditado en el aviso.

### Compromiso del equipo

- **Acuse de recibo**: dentro de 5 días laborables.
- **Evaluación inicial**: dentro de 15 días naturales.
- **Plan de mitigación**: comunicado al reportante antes de la
  publicación del fix.
- **Coordinación**: ventana estándar de 90 días para divulgación
  coordinada; puede acortarse si la vulnerabilidad está ya en
  explotación activa, o ampliarse si la complejidad lo requiere.

### Alcance

Quedan dentro del alcance:

- Ejecución arbitraria de código al cargar checkpoints maliciosos.
- Inyección de modelo o envenenamiento durante factorización /
  descompresión.
- Vulnerabilidades en `ZActivationCheckpoint` que permitan exfiltrar
  activaciones de otro proceso.
- Fallos en `ZVRAMGovernor` que provoquen denegación de servicio por
  consumo descontrolado de VRAM.
- Vulnerabilidades en dependencias OPCIONALES (MNEME) **cuando se
  combinan** con MZTrain de forma específica.

Quedan fuera del alcance:

- Bugs de comportamiento numérico, divergencia del entrenamiento o
  resultados experimentales subóptimos: usar GitHub Issues.
- Vulnerabilidades en PyTorch, NumPy, CUDA o sistema operativo:
  reportar a los respectivos mantenedores.

## Salvaguarda de buena fe

Los Titulares se comprometen a no emprender acciones legales contra
investigadores de seguridad que actúen de buena fe, sigan esta
política y no destruyan datos, no degraden servicios en producción
de terceros, ni accedan a datos personales sin consentimiento.

## PGP / firma

Las claves PGP de coordinación están disponibles bajo petición al
correo de seguridad.
