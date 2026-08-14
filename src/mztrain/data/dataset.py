"""
MZTrain - Datasets para pre-entrenamiento de modelos de codigo.

Incluye generador de codigo sintetico multi-lenguaje y datasets
para Masked Language Modeling (MLM) estilo BERT.
"""

import random
import logging
from typing import List, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .tokenizer import CLS_ID, CodeTokenizer, MASK_ID, PAD_ID

logger = logging.getLogger("mztrain")


class SyntheticCodeGenerator:
    """Generador de snippets de codigo sintetico multi-lenguaje.

    Genera codigo realista en Python, JavaScript, Java, Go, Ruby y PHP
    para pre-entrenamiento sin necesidad de datasets externos.

    Args:
        languages: Lista de lenguajes a generar. Default: todos.
        seed: Semilla para reproducibilidad.

    Example:
        >>> gen = SyntheticCodeGenerator(seed=42)
        >>> snippets = gen.generate(100)
        >>> print(snippets[0][:50])
    """

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ):
        self.languages = languages or ["python", "javascript", "java", "go", "ruby", "php"]
        self.rng = random.Random(seed)

        # Templates por lenguaje
        self._templates = {
            "python": self._python_templates(),
            "javascript": self._javascript_templates(),
            "java": self._java_templates(),
            "go": self._go_templates(),
            "ruby": self._ruby_templates(),
            "php": self._php_templates(),
        }

    def generate(self, n: int) -> List[str]:
        """Generar N snippets de codigo.

        Args:
            n: Numero de snippets a generar.

        Returns:
            Lista de strings con codigo fuente.
        """
        snippets = []
        for _ in range(n):
            lang = self.rng.choice(self.languages)
            templates = self._templates.get(lang, self._templates["python"])
            template = self.rng.choice(templates)
            snippet = self._fill_template(template, lang)
            snippets.append(snippet)
        return snippets

    def _fill_template(self, template: str, lang: str) -> str:
        """Rellenar un template con nombres aleatorios."""
        names = ["data", "result", "value", "items", "count", "total",
                 "buffer", "output", "cache", "index", "config", "options",
                 "params", "response", "request", "handler", "manager",
                 "service", "client", "server", "queue", "stack"]
        types = ["int", "str", "float", "bool", "list", "dict", "set"]
        nums = [str(self.rng.randint(0, 1000)) for _ in range(5)]
        methods = ["process", "compute", "transform", "validate", "parse",
                   "format", "encode", "decode", "serialize", "filter",
                   "sort", "merge", "split", "join", "convert"]

        result = template
        for i in range(10):
            result = result.replace(f"{{name{i}}}", self.rng.choice(names))
            result = result.replace(f"{{method{i}}}", self.rng.choice(methods))
            result = result.replace(f"{{type{i}}}", self.rng.choice(types))
            result = result.replace(f"{{num{i}}}", self.rng.choice(nums))

        return result

    def _python_templates(self) -> List[str]:
        return [
            # Funcion simple
            "def {method0}({name0}, {name1}):\n"
            "    {name2} = []\n"
            "    for item in {name0}:\n"
            "        if item > {num0}:\n"
            "            {name2}.append(item * {num1})\n"
            "    return {name2}\n",

            # Clase
            "class {name0}Manager:\n"
            "    def __init__(self, {name1}=None):\n"
            "        self.{name1} = {name1} or []\n"
            "        self.{name2} = {num0}\n"
            "\n"
            "    def {method0}(self, item):\n"
            "        self.{name1}.append(item)\n"
            "        self.{name2} += 1\n"
            "        return len(self.{name1})\n"
            "\n"
            "    def {method1}(self):\n"
            "        return [x for x in self.{name1} if x is not None]\n",

            # Decorador y contexto
            "import functools\n"
            "\n"
            "def {method0}(func):\n"
            "    @functools.wraps(func)\n"
            "    def wrapper(*args, **kwargs):\n"
            "        {name0} = func(*args, **kwargs)\n"
            "        return {name0}\n"
            "    return wrapper\n"
            "\n"
            "@{method0}\n"
            "def {method1}({name1}):\n"
            "    with open({name1}) as f:\n"
            "        return f.read()\n",

            # Comprehension y lambda
            "def {method0}({name0}):\n"
            "    {name1} = {{k: v for k, v in {name0}.items() if v > {num0}}}\n"
            "    {name2} = sorted({name1}, key=lambda x: x[1], reverse=True)\n"
            "    return {name2}[:{num1}]\n",

            # Try/except
            "def {method0}({name0}, {name1}={num0}):\n"
            "    try:\n"
            "        {name2} = int({name0}) + {name1}\n"
            "        if {name2} < 0:\n"
            "            raise ValueError(f'Invalid: {{{name2}}}')\n"
            "        return {name2}\n"
            "    except (ValueError, TypeError) as e:\n"
            "        print(f'Error: {{e}}')\n"
            "        return None\n",

            # Generator
            "def {method0}({name0}, {name1}={num0}):\n"
            "    {name2} = 0\n"
            "    for item in {name0}:\n"
            "        if {name2} >= {name1}:\n"
            "            break\n"
            "        yield item\n"
            "        {name2} += 1\n",

            # Async
            "import asyncio\n"
            "\n"
            "async def {method0}({name0}):\n"
            "    {name1} = []\n"
            "    async for item in {name0}:\n"
            "        {name2} = await {method1}(item)\n"
            "        {name1}.append({name2})\n"
            "    return {name1}\n",

            # Dataclass
            "from dataclasses import dataclass\n"
            "from typing import List, Optional\n"
            "\n"
            "@dataclass\n"
            "class {name0}Config:\n"
            "    {name1}: int = {num0}\n"
            "    {name2}: float = 0.{num1}\n"
            "    {name3}: Optional[str] = None\n"
            "    {name4}: List[int] = None\n"
            "\n"
            "    def __post_init__(self):\n"
            "        if self.{name4} is None:\n"
            "            self.{name4} = []\n",
        ]

    def _javascript_templates(self) -> List[str]:
        return [
            # Arrow function
            "const {method0} = ({name0}, {name1}) => {{\n"
            "    const {name2} = {name0}.filter(x => x > {num0});\n"
            "    return {name2}.map(x => x * {num1});\n"
            "}};\n",

            # Class
            "class {name0}Service {{\n"
            "    constructor({name1}) {{\n"
            "        this.{name1} = {name1};\n"
            "        this.{name2} = new Map();\n"
            "    }}\n"
            "\n"
            "    async {method0}(key) {{\n"
            "        if (this.{name2}.has(key)) {{\n"
            "            return this.{name2}.get(key);\n"
            "        }}\n"
            "        const {name3} = await this.{method1}(key);\n"
            "        this.{name2}.set(key, {name3});\n"
            "        return {name3};\n"
            "    }}\n"
            "}}\n",

            # Promise chain
            "function {method0}({name0}) {{\n"
            "    return fetch({name0})\n"
            "        .then(res => res.json())\n"
            "        .then({name1} => {{\n"
            "            const {name2} = {name1}.filter(x => x.id > {num0});\n"
            "            return {name2};\n"
            "        }})\n"
            "        .catch(err => console.error(err));\n"
            "}}\n",

            # Destructuring
            "const {method0} = ({{ {name0}, {name1}, ...rest }}) => {{\n"
            "    const {name2} = {{ {name0}, {name1}: {name1} + {num0} }};\n"
            "    return {{ ...{name2}, ...rest }};\n"
            "}};\n",
        ]

    def _java_templates(self) -> List[str]:
        return [
            # Class with method
            "public class {name0}Handler {{\n"
            "    private List<String> {name1};\n"
            "    private int {name2};\n"
            "\n"
            "    public {name0}Handler() {{\n"
            "        this.{name1} = new ArrayList<>();\n"
            "        this.{name2} = {num0};\n"
            "    }}\n"
            "\n"
            "    public boolean {method0}(String item) {{\n"
            "        if (item == null || item.isEmpty()) {{\n"
            "            return false;\n"
            "        }}\n"
            "        this.{name1}.add(item);\n"
            "        this.{name2}++;\n"
            "        return true;\n"
            "    }}\n"
            "\n"
            "    public List<String> {method1}() {{\n"
            "        return Collections.unmodifiableList(this.{name1});\n"
            "    }}\n"
            "}}\n",

            # Interface + implementation
            "public interface {name0}Processor {{\n"
            "    void {method0}(String {name1});\n"
            "    int {method1}();\n"
            "}}\n"
            "\n"
            "public class Default{name0}Processor implements {name0}Processor {{\n"
            "    private int {name2} = 0;\n"
            "\n"
            "    @Override\n"
            "    public void {method0}(String {name1}) {{\n"
            "        System.out.println({name1});\n"
            "        this.{name2}++;\n"
            "    }}\n"
            "\n"
            "    @Override\n"
            "    public int {method1}() {{\n"
            "        return this.{name2};\n"
            "    }}\n"
            "}}\n",

            # Stream API
            "public List<String> {method0}(List<String> {name0}) {{\n"
            "    return {name0}.stream()\n"
            "        .filter(s -> s.length() > {num0})\n"
            "        .map(String::toUpperCase)\n"
            "        .sorted()\n"
            "        .collect(Collectors.toList());\n"
            "}}\n",
        ]

    def _go_templates(self) -> List[str]:
        return [
            # Struct and method
            "type {name0}Store struct {{\n"
            "    {name1} []string\n"
            "    {name2} int\n"
            "}}\n"
            "\n"
            "func New{name0}Store() *{name0}Store {{\n"
            "    return &{name0}Store{{\n"
            "        {name1}: make([]string, 0),\n"
            "        {name2}: {num0},\n"
            "    }}\n"
            "}}\n"
            "\n"
            "func (s *{name0}Store) {method0}(item string) error {{\n"
            "    if item == \"\" {{\n"
            "        return fmt.Errorf(\"empty item\")\n"
            "    }}\n"
            "    s.{name1} = append(s.{name1}, item)\n"
            "    return nil\n"
            "}}\n",

            # Goroutines
            "func {method0}(ctx context.Context, {name0} <-chan string) <-chan string {{\n"
            "    {name1} := make(chan string)\n"
            "    go func() {{\n"
            "        defer close({name1})\n"
            "        for item := range {name0} {{\n"
            "            select {{\n"
            "            case <-ctx.Done():\n"
            "                return\n"
            "            case {name1} <- strings.ToUpper(item):\n"
            "            }}\n"
            "        }}\n"
            "    }}()\n"
            "    return {name1}\n"
            "}}\n",

            # Error handling
            "func {method0}({name0} string) (int, error) {{\n"
            "    {name1}, err := strconv.Atoi({name0})\n"
            "    if err != nil {{\n"
            "        return 0, fmt.Errorf(\"{method0}: %w\", err)\n"
            "    }}\n"
            "    if {name1} < {num0} {{\n"
            "        return 0, fmt.Errorf(\"value too small: %d\", {name1})\n"
            "    }}\n"
            "    return {name1} * {num1}, nil\n"
            "}}\n",
        ]

    def _ruby_templates(self) -> List[str]:
        return [
            # Class
            "class {name0}Manager\n"
            "  attr_reader :{name1}, :{name2}\n"
            "\n"
            "  def initialize({name1} = [])\n"
            "    @{name1} = {name1}\n"
            "    @{name2} = {num0}\n"
            "  end\n"
            "\n"
            "  def {method0}(item)\n"
            "    raise ArgumentError, 'nil item' if item.nil?\n"
            "    @{name1} << item\n"
            "    @{name2} += 1\n"
            "    self\n"
            "  end\n"
            "\n"
            "  def {method1}\n"
            "    @{name1}.select {{ |x| x.to_i > {num1} }}\n"
            "  end\n"
            "end\n",

            # Block and enumerable
            "def {method0}({name0})\n"
            "  {name0}\n"
            "    .map {{ |x| x.to_s }}\n"
            "    .select {{ |x| x.length > {num0} }}\n"
            "    .each_with_object({{}}) do |item, {name1}|\n"
            "      {name1}[item] = item.length\n"
            "    end\n"
            "end\n",
        ]

    def _php_templates(self) -> List[str]:
        return [
            # Class
            "<?php\n"
            "class {name0}Repository {{\n"
            "    private array ${name1};\n"
            "    private int ${name2};\n"
            "\n"
            "    public function __construct() {{\n"
            "        $this->{name1} = [];\n"
            "        $this->{name2} = {num0};\n"
            "    }}\n"
            "\n"
            "    public function {method0}(string $item): bool {{\n"
            "        if (empty($item)) {{\n"
            "            return false;\n"
            "        }}\n"
            "        $this->{name1}[] = $item;\n"
            "        $this->{name2}++;\n"
            "        return true;\n"
            "    }}\n"
            "\n"
            "    public function {method1}(): array {{\n"
            "        return array_filter($this->{name1}, function($x) {{\n"
            "            return strlen($x) > {num1};\n"
            "        }});\n"
            "    }}\n"
            "}}\n",

            # Function
            "<?php\n"
            "function {method0}(array ${name0}, int ${name1} = {num0}): array {{\n"
            "    ${name2} = array_map(function($item) use (${name1}) {{\n"
            "        return $item * ${name1};\n"
            "    }}, ${name0});\n"
            "    return array_filter(${name2}, fn($x) => $x > {num1});\n"
            "}}\n",
        ]


class CodeDataset(Dataset):
    """Dataset de codigo tokenizado.

    Tokeniza snippets de codigo y los prepara para entrenamiento.

    Args:
        snippets: Lista de strings con codigo fuente.
        tokenizer: Instancia de CodeTokenizer.
        max_length: Longitud maxima de secuencia.

    Example:
        >>> gen = SyntheticCodeGenerator(seed=42)
        >>> snippets = gen.generate(100)
        >>> dataset = CodeDataset(snippets, CodeTokenizer())
        >>> input_ids, attention_mask = dataset[0]
    """

    def __init__(
        self,
        snippets: List[str],
        tokenizer: CodeTokenizer,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = []

        for snippet in snippets:
            ids = tokenizer.encode(snippet, max_length=max_length)
            self.encodings.append(ids)

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retorna (input_ids, attention_mask)."""
        ids = self.encodings[idx]
        input_ids = torch.tensor(ids, dtype=torch.long)
        attention_mask = (input_ids != PAD_ID).long()
        return input_ids, attention_mask


class MLMDataset(Dataset):
    """Dataset con Masked Language Modeling (MLM) estilo BERT.

    Aplica masking al 15% de tokens: 80% [MASK], 10% random, 10% sin cambio.

    Args:
        code_dataset: CodeDataset base.
        mask_prob: Probabilidad de masking (default: 0.15).
        seed: Semilla para reproducibilidad.
        ensure_at_least_one_mask: Si True, fuerza al menos un token enmascarado
            por muestra cuando existen tokens elegibles. Evita ejemplos con
            loss vacio en secuencias cortas o datasets pequenos.

    Example:
        >>> gen = SyntheticCodeGenerator(seed=42)
        >>> snippets = gen.generate(100)
        >>> base = CodeDataset(snippets, CodeTokenizer())
        >>> mlm = MLMDataset(base, mask_prob=0.15)
        >>> input_ids, attention_mask, labels = mlm[0]
    """

    def __init__(
        self,
        code_dataset: CodeDataset,
        mask_prob: float = 0.15,
        seed: Optional[int] = None,
        ensure_at_least_one_mask: bool = True,
    ):
        if not 0.0 <= mask_prob <= 1.0:
            raise ValueError(f"mask_prob debe estar en [0, 1], recibido: {mask_prob}")
        self.code_dataset = code_dataset
        self.mask_prob = mask_prob
        self.vocab_size = code_dataset.tokenizer.vocab_size
        self.rng = random.Random(seed)
        self.ensure_at_least_one_mask = ensure_at_least_one_mask

    def __len__(self) -> int:
        return len(self.code_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retorna (masked_input_ids, attention_mask, labels).

        Labels tiene -100 para tokens no maskeados (ignorados en loss).
        """
        input_ids, attention_mask = self.code_dataset[idx]

        masked_ids = input_ids.clone()
        labels = torch.full_like(input_ids, -100)

        # Solo maskear tokens que no son especiales ni padding
        special_ids = {PAD_ID, 1, 2, 3, 4}  # PAD, UNK, CLS, SEP, MASK
        candidate_positions = []
        masked_count = 0

        for i in range(len(input_ids)):
            token_id = input_ids[i].item()

            if token_id in special_ids:
                continue

            candidate_positions.append(i)
            if self.rng.random() < self.mask_prob:
                self._mask_position(masked_ids, labels, i, token_id)
                masked_count += 1

        if self.ensure_at_least_one_mask and masked_count == 0 and candidate_positions:
            i = self.rng.choice(candidate_positions)
            self._mask_position(masked_ids, labels, i, input_ids[i].item())

        return masked_ids, attention_mask, labels

    def _mask_position(
        self,
        masked_ids: torch.Tensor,
        labels: torch.Tensor,
        idx: int,
        token_id: int,
    ) -> None:
        """Aplicar la regla BERT 80/10/10 a una posicion elegida."""
        labels[idx] = token_id

        rand = self.rng.random()
        if rand < 0.8:
            # 80%: reemplazar con [MASK]
            masked_ids[idx] = MASK_ID
        elif rand < 0.9:
            # 10%: reemplazar con token aleatorio
            masked_ids[idx] = self.rng.randint(5, self.vocab_size - 1)
        # else 10%: dejar sin cambio


class CausalCodeDataset(Dataset):
    """Dataset para objetivo causal/prefix-LM sobre codigo tokenizado.

    A diferencia de MLMDataset, no oculta tokens. Retorna la secuencia completa
    y labels compatibles con modelos autoregresivos: cada posicion predice el
    siguiente token dentro del modelo.

    Args:
        code_dataset: CodeDataset base.
        ignore_special_tokens: Si True, ignora PAD/CLS en la loss.
    """

    def __init__(
        self,
        code_dataset: CodeDataset,
        ignore_special_tokens: bool = True,
    ):
        self.code_dataset = code_dataset
        self.ignore_special_tokens = ignore_special_tokens

    def __len__(self) -> int:
        return len(self.code_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retorna (input_ids, attention_mask, labels)."""
        input_ids, attention_mask = self.code_dataset[idx]
        labels = input_ids.clone()

        if self.ignore_special_tokens:
            labels[(labels == PAD_ID) | (labels == CLS_ID)] = -100
        else:
            labels[labels == PAD_ID] = -100

        return input_ids, attention_mask, labels
