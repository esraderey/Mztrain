# Guia de Contribucion - MZTrain

Gracias por tu interes en contribuir a MZTrain. Esta guia te ayudara a comenzar.

## Configuracion del Entorno

### Requisitos Previos
- Python 3.8+
- PyTorch 2.0+
- Git

### Instalacion para Desarrollo

```bash
git clone https://github.com/esraderey/mztrain.git
cd mztrain
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
pip install -e ".[dev]"
```

## Flujo de Trabajo

### 1. Crear una rama

```bash
git checkout -b feature/mi-nueva-funcionalidad
```

### 2. Hacer cambios

- Seguir el estilo de codigo existente
- Agregar tests para nueva funcionalidad
- Actualizar documentacion si es necesario

### 3. Ejecutar tests

```bash
# Tests completos
pytest tests/ -v

# Con cobertura
pytest tests/ -v --cov=src/mztrain --cov-report=html

# Tests rapidos (sin GPU)
pytest tests/ -v -m "not gpu and not slow"
```

### 4. Verificar calidad

```bash
# Formateo
black src/ tests/ --line-length 127
isort src/ tests/ --profile black --line-length 127

# Linting
flake8 src/mztrain/ --max-line-length 127
ruff check src/mztrain/

# Type checking
mypy src/mztrain/ --ignore-missing-imports
```

### 5. Commit y Pull Request

```bash
git add .
git commit -m "feat: descripcion clara del cambio"
git push origin feature/mi-nueva-funcionalidad
```

Luego crear un Pull Request en GitHub.

## Estilo de Codigo

- **Formato**: black con line-length=127
- **Imports**: isort con profile=black
- **Docstrings**: Estilo Google con tipos
- **Nombres**: snake_case para funciones, PascalCase para clases
- **Idioma**: Docstrings y comentarios en espanol (codigo en ingles)

## Estructura de Tests

```
tests/
├── conftest.py          # Fixtures compartidos
├── test_config.py       # Tests de configuracion
├── test_layers.py       # Tests de capas factorizadas
├── test_optimizer.py    # Tests del optimizer
├── test_gradient.py     # Tests de compresion de gradientes
├── test_scheduler.py    # Tests del scheduler de rango
├── test_engine.py       # Tests del motor de entrenamiento
├── test_checkpoint.py   # Tests de activation checkpointing
└── test_utils.py        # Tests de utilidades
```

## Convencion de Commits

Usar [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` Nueva funcionalidad
- `fix:` Correccion de bug
- `docs:` Documentacion
- `test:` Tests
- `refactor:` Refactorizacion
- `perf:` Mejora de rendimiento
- `ci:` Cambios en CI/CD

## Reporte de Bugs

Al reportar un bug, incluir:

1. Version de MZTrain, Python y PyTorch
2. Sistema operativo
3. Pasos para reproducir
4. Comportamiento esperado vs actual
5. Logs o mensajes de error

## Solicitud de Funcionalidades

Al solicitar una funcionalidad:

1. Describir el caso de uso
2. Proponer una API tentativa
3. Considerar impacto en rendimiento y memoria

## Licencia

MZTrain se distribuye bajo la licencia **MIT**. Al contribuir, aceptas que tus
contribuciones se licencien bajo los mismos términos (inbound = outbound). No se
exige cesión de copyright: conservas la autoría de tu contribución. Basta con
firmar tus commits (`git commit -s`, Developer Certificate of Origin) para
declarar que tienes derecho a aportarla.
