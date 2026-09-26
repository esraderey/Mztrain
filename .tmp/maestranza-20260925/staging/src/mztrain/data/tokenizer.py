"""
MZTrain - Tokenizador code-aware.

Tokenizador especializado para codigo fuente con soporte para
splitting camelCase/snake_case, operadores, y keywords multi-lenguaje.
"""

import re
import logging
from typing import List, Dict, Optional
from collections import Counter

logger = logging.getLogger("mztrain")

# Tokens especiales
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
MASK_TOKEN = "[MASK]"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN]
PAD_ID = 0
UNK_ID = 1
CLS_ID = 2
SEP_ID = 3
MASK_ID = 4

# Keywords por lenguaje
_PYTHON_KEYWORDS = [
    "def", "class", "return", "if", "else", "elif", "for", "while",
    "import", "from", "as", "try", "except", "finally", "with",
    "yield", "lambda", "pass", "break", "continue", "raise",
    "True", "False", "None", "and", "or", "not", "in", "is",
    "global", "nonlocal", "assert", "del", "async", "await",
    "self", "print", "len", "range", "list", "dict", "set",
    "tuple", "int", "float", "str", "bool", "type", "super",
    "__init__", "__name__", "__main__", "__str__", "__repr__",
]

_JS_KEYWORDS = [
    "function", "const", "let", "var", "return", "if", "else",
    "for", "while", "do", "switch", "case", "break", "continue",
    "class", "extends", "new", "this", "super", "import", "export",
    "default", "async", "await", "try", "catch", "finally", "throw",
    "true", "false", "null", "undefined", "typeof", "instanceof",
    "console", "log", "document", "window", "require", "module",
    "Promise", "Array", "Object", "String", "Number", "Boolean",
    "Math", "JSON", "Map", "Set", "Symbol", "Error",
]

_JAVA_KEYWORDS = [
    "public", "private", "protected", "static", "final", "abstract",
    "class", "interface", "extends", "implements", "new", "this",
    "super", "return", "if", "else", "for", "while", "do", "switch",
    "case", "break", "continue", "try", "catch", "finally", "throw",
    "throws", "void", "int", "long", "double", "float", "boolean",
    "char", "byte", "short", "String", "null", "true", "false",
    "import", "package", "enum", "instanceof", "synchronized",
    "volatile", "transient", "native", "strictfp", "assert",
    "System", "out", "println", "Override", "List", "Map",
    "ArrayList", "HashMap", "Iterator", "Collection",
]

_GO_KEYWORDS = [
    "func", "package", "import", "return", "if", "else", "for",
    "range", "switch", "case", "default", "break", "continue",
    "go", "chan", "select", "defer", "map", "struct", "interface",
    "type", "var", "const", "nil", "true", "false", "make",
    "append", "len", "cap", "new", "delete", "copy", "close",
    "panic", "recover", "error", "string", "int", "float64",
    "bool", "byte", "rune", "fmt", "Println", "Printf",
]

_RUBY_KEYWORDS = [
    "def", "end", "class", "module", "return", "if", "else",
    "elsif", "unless", "while", "until", "for", "do", "begin",
    "rescue", "ensure", "raise", "yield", "block_given?",
    "self", "super", "nil", "true", "false", "require",
    "include", "extend", "attr_accessor", "attr_reader",
    "puts", "print", "each", "map", "select", "reduce",
    "new", "initialize", "to_s", "to_i", "to_f",
]

_PHP_KEYWORDS = [
    "function", "class", "public", "private", "protected", "static",
    "return", "if", "else", "elseif", "for", "foreach", "while",
    "do", "switch", "case", "break", "continue", "try", "catch",
    "finally", "throw", "new", "extends", "implements", "interface",
    "abstract", "namespace", "use", "require", "include",
    "echo", "print", "array", "null", "true", "false",
    "$this", "->", "=>", "::", "isset", "empty", "unset",
]

# Operadores y simbolos comunes
_OPERATORS = [
    "==", "!=", "<=", ">=", "&&", "||", "++", "--", "+=", "-=",
    "*=", "/=", "%=", "**", "//", "<<", ">>", "->", "=>", "::",
    "...", "..", "??", "?.", "!!", "<>", "|>", "~>",
    "+", "-", "*", "/", "%", "=", "<", ">", "!", "&", "|",
    "^", "~", "?", ":", ";", ",", ".", "@", "#",
    "(", ")", "[", "]", "{", "}", "\\", "'", '"', "`",
]

# Caracteres individuales comunes en codigo
_CODE_CHARS = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


class CodeTokenizer:
    """Tokenizador code-aware con splitting inteligente.

    Soporta splitting camelCase, snake_case, operadores multi-caracter,
    y un vocabulario pre-construido de ~50K tokens para codigo.

    Args:
        vocab_size: Tamano maximo del vocabulario.
        max_length: Longitud maxima de secuencia.

    Example:
        >>> tok = CodeTokenizer(vocab_size=50000)
        >>> ids = tok.encode("def hello_world(): return 42")
        >>> text = tok.decode(ids)
    """

    # Cota superior de caracteres por token para acotar el input crudo de
    # encode() antes de tokenizar (defensa frente a input desmesurado).
    _MAX_CHARS_PER_TOKEN = 64

    def __init__(self, vocab_size: int = 50000, max_length: int = 512):
        self.vocab_size = vocab_size
        self.max_length = max_length

        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        self._build_vocabulary()

    def _build_vocabulary(self) -> None:
        """Construir vocabulario code-aware."""
        tokens: List[str] = []

        # 1. Tokens especiales (IDs 0-4)
        tokens.extend(SPECIAL_TOKENS)

        # 2. Caracteres individuales
        tokens.extend(_CODE_CHARS)

        # 3. Digitos y numeros comunes
        for i in range(256):
            tok = str(i)
            if tok not in tokens:
                tokens.append(tok)

        # 4. Operadores
        for op in _OPERATORS:
            if op not in tokens:
                tokens.append(op)

        # 5. Keywords de todos los lenguajes
        all_keywords = set()
        for kw_list in [
            _PYTHON_KEYWORDS, _JS_KEYWORDS, _JAVA_KEYWORDS,
            _GO_KEYWORDS, _RUBY_KEYWORDS, _PHP_KEYWORDS,
        ]:
            all_keywords.update(kw_list)

        for kw in sorted(all_keywords):
            if kw not in tokens:
                tokens.append(kw)

        # 6. Subwords comunes en codigo
        common_subwords = [
            "get", "set", "add", "del", "put", "pop", "has",
            "is_", "to_", "on_", "in_", "by_", "no_", "my_",
            "max", "min", "sum", "avg", "cnt", "num", "idx",
            "val", "key", "err", "msg", "buf", "tmp", "ctx",
            "arg", "opt", "cfg", "src", "dst", "res", "req",
            "init", "load", "save", "send", "recv", "read", "write",
            "open", "close", "start", "stop", "run", "exec",
            "push", "pull", "find", "sort", "iter", "next",
            "name", "data", "info", "node", "item", "size",
            "path", "file", "line", "text", "code", "test",
            "type", "mode", "flag", "mask", "hash", "lock",
            "cache", "queue", "stack", "array", "table", "index",
            "model", "layer", "param", "batch", "epoch", "loss",
            "train", "valid", "input", "output", "embed",
            "weight", "bias", "grad", "optim", "sched",
            "config", "module", "tensor", "linear", "conv",
            "attn", "norm", "drop", "pool", "block",
            "encoder", "decoder", "attention", "transform",
            "forward", "backward", "predict", "compute",
            "String", "Integer", "Double", "Float", "Boolean",
            "Exception", "Error", "Warning", "Logger",
            "http", "url", "api", "json", "xml", "html",
            "async", "sync", "thread", "process", "task",
            "query", "insert", "update", "delete", "select",
            "create", "remove", "modify", "change", "check",
            "enable", "disable", "toggle", "switch", "reset",
            "server", "client", "socket", "connect",
            "request", "response", "handler", "callback",
            "parse", "format", "encode", "decode", "convert",
            "validate", "verify", "assert", "expect",
            "public", "private", "protected", "internal",
            "abstract", "virtual", "override", "static",
            "const", "final", "readonly", "mutable",
            "import", "export", "require", "include",
            "package", "module", "class", "struct",
            "function", "method", "lambda", "closure",
            "interface", "protocol", "trait", "mixin",
            "generic", "template", "macro", "annotation",
            "iterator", "generator", "promise", "future",
            "channel", "mutex", "semaphore", "barrier",
            "allocate", "free", "malloc", "realloc",
            "pointer", "reference", "value", "copy",
            "serialize", "deserialize", "marshal",
            "compress", "decompress", "encrypt", "decrypt",
            "register", "unregister", "subscribe",
            "publish", "dispatch", "emit", "listen",
            "render", "display", "draw", "paint",
            "width", "height", "depth", "length",
            "color", "font", "style", "theme",
            "button", "label", "panel", "view",
            "login", "logout", "auth", "token",
            "user", "admin", "role", "permission",
            "session", "cookie", "header", "body",
            "database", "schema", "migration",
            "repository", "service", "controller",
            "factory", "builder", "singleton",
            "observer", "adapter", "proxy", "facade",
            "strategy", "command", "state", "bridge",
            "visitor", "mediator", "memento",
        ]
        for sw in common_subwords:
            if sw not in tokens:
                tokens.append(sw)

        # 7. Variantes con prefijos/sufijos comunes
        prefixes = ["get_", "set_", "is_", "has_", "on_", "do_", "_"]
        suffixes = ["_id", "_name", "_type", "_size", "_count", "_list", "_map",
                     "_data", "_info", "_path", "_file", "_dir", "_key", "_val",
                     "ed", "er", "ing", "tion", "ment", "ness", "able", "ible",
                     "ful", "less", "ous", "ive", "ize", "ify"]
        for p in prefixes:
            if p not in tokens:
                tokens.append(p)
        for s in suffixes:
            if s not in tokens:
                tokens.append(s)

        # 8. Strings comunes
        common_strings = [
            '""', "''", "``", "[]", "{}", "()", "<>",
            "\\n", "\\t", "\\r", "\\\\", '\\"', "\\'",
            "//", "/*", "*/", "<!--", "-->", "/**",
            "TODO", "FIXME", "HACK", "NOTE", "XXX",
            "utf-8", "ascii", "utf8", "latin1",
            "GET", "POST", "PUT", "DELETE", "PATCH",
            "OK", "ERROR", "WARN", "INFO", "DEBUG",
        ]
        for s in common_strings:
            if s not in tokens:
                tokens.append(s)

        # 9. Indentation tokens
        for i in range(1, 9):
            indent = " " * i
            if indent not in tokens:
                tokens.append(indent)
        tokens.append("\t")
        tokens.append("\n")

        # 10. Subword fragments (bi/tri-grams de letras comunes)
        bigrams = [
            "th", "he", "in", "er", "an", "re", "on", "at", "en", "nd",
            "ti", "es", "or", "te", "of", "ed", "is", "it", "al", "ar",
            "st", "to", "nt", "ng", "se", "ha", "as", "ou", "io", "le",
            "ve", "co", "me", "de", "hi", "ri", "ro", "ic", "ne", "ea",
            "ra", "ce", "li", "ch", "ll", "be", "ma", "si", "om", "ur",
        ]
        for bg in bigrams:
            if bg not in tokens:
                tokens.append(bg)

        # Truncar al tamano de vocabulario
        tokens = tokens[:self.vocab_size]

        # Rellenar con tokens placeholder si hace falta
        while len(tokens) < self.vocab_size:
            tokens.append(f"[UNUSED_{len(tokens)}]")

        # Construir mappings
        self.token_to_id = {tok: idx for idx, tok in enumerate(tokens)}
        self.id_to_token = {idx: tok for idx, tok in enumerate(tokens)}

    def _split_code(self, code: str) -> List[str]:
        """Splitting code-aware: camelCase, snake_case, operadores.

        Args:
            code: Codigo fuente a tokenizar.

        Returns:
            Lista de tokens.
        """
        tokens = []
        i = 0
        n = len(code)

        while i < n:
            c = code[i]

            # Whitespace
            if c in (' ', '\t'):
                # Contar espacios consecutivos
                j = i
                while j < n and code[j] == c:
                    j += 1
                space = code[i:j]
                if space in self.token_to_id:
                    tokens.append(space)
                else:
                    for _ in range(j - i):
                        tokens.append(c)
                i = j
                continue

            # Newlines
            if c == '\n':
                tokens.append('\n')
                i += 1
                continue

            # Multi-char operators (check longest first)
            if c in '=!<>&|+-*/%^~?.:#':
                best = c
                for length in (3, 2):
                    candidate = code[i:i+length]
                    if candidate in self.token_to_id:
                        best = candidate
                        break
                tokens.append(best)
                i += len(best)
                continue

            # Brackets and delimiters
            if c in '()[]{},:;@\\`':
                tokens.append(c)
                i += 1
                continue

            # String literals (simplified - just extract the quotes)
            if c in ('"', "'"):
                quote = c
                j = i + 1
                while j < n and code[j] != quote:
                    if code[j] == '\\':
                        j += 1  # Skip escaped char
                    j += 1
                j = min(j + 1, n)
                string_content = code[i:j]
                # Break long strings into subwords
                if len(string_content) <= 20 and string_content in self.token_to_id:
                    tokens.append(string_content)
                else:
                    tokens.append(quote)
                    for sc in self._split_identifier(string_content[1:-1] if len(string_content) > 2 else ""):
                        tokens.append(sc)
                    if j <= n and len(string_content) > 1:
                        tokens.append(quote)
                i = j
                continue

            # Numbers
            if c.isdigit():
                j = i
                while j < n and (code[j].isdigit() or code[j] in '.eExXbBoO_'):
                    j += 1
                num = code[i:j]
                if num in self.token_to_id:
                    tokens.append(num)
                else:
                    for ch in num:
                        tokens.append(ch)
                i = j
                continue

            # Identifiers
            if c.isalpha() or c == '_' or c == '$':
                j = i
                while j < n and (code[j].isalnum() or code[j] == '_' or code[j] == '$'):
                    j += 1
                ident = code[i:j]

                # Check if full identifier is in vocab
                if ident in self.token_to_id:
                    tokens.append(ident)
                else:
                    # Split camelCase/snake_case
                    parts = self._split_identifier(ident)
                    tokens.extend(parts)
                i = j
                continue

            # Default: single character
            tokens.append(c)
            i += 1

        return tokens

    def _split_identifier(self, ident: str) -> List[str]:
        """Split un identificador en sub-tokens.

        Soporta camelCase, PascalCase, snake_case, UPPER_CASE.

        Args:
            ident: Identificador a dividir.

        Returns:
            Lista de sub-tokens.
        """
        if not ident:
            return []

        if ident in self.token_to_id:
            return [ident]

        # Split por underscore primero
        parts = ident.split('_')
        result = []

        for part in parts:
            if not part:
                result.append('_')
                continue

            if part in self.token_to_id:
                result.append(part)
                continue

            # Split camelCase
            subparts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\b)|[A-Z]|[0-9]+', part)
            if not subparts:
                subparts = [part]

            for sp in subparts:
                lower = sp.lower()
                if sp in self.token_to_id:
                    result.append(sp)
                elif lower in self.token_to_id:
                    result.append(lower)
                else:
                    # Character-level fallback
                    for ch in sp:
                        result.append(ch)

            if part != parts[-1] and '_' in ident:
                result.append('_')

        # Remove trailing underscore if added
        if result and result[-1] == '_' and not ident.endswith('_'):
            result.pop()

        return result

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = True,
    ) -> List[int]:
        """Codificar texto a IDs de tokens.

        Args:
            text: Codigo fuente a codificar.
            add_special_tokens: Agregar [CLS] y [SEP].
            max_length: Longitud maxima (default: self.max_length).
            padding: Aplicar padding a max_length.

        Returns:
            Lista de IDs de tokens.
        """
        max_len = max_length or self.max_length
        # Acotar el input crudo: mas alla de este tope todo se truncaria de
        # todos modos, asi evitamos materializar una lista de tokens sin techo
        # con un input adversarialmente grande.
        cap = max_len * self._MAX_CHARS_PER_TOKEN
        if len(text) > cap:
            text = text[:cap]
        tokens = self._split_code(text)

        # Convert to IDs
        ids = [self.token_to_id.get(t, UNK_ID) for t in tokens]

        # Add special tokens
        if add_special_tokens:
            ids = [CLS_ID] + ids[:max_len - 2] + [SEP_ID]
        else:
            ids = ids[:max_len]

        # Padding
        if padding and len(ids) < max_len:
            ids = ids + [PAD_ID] * (max_len - len(ids))

        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Decodificar IDs de tokens a texto.

        Args:
            ids: Lista de IDs de tokens.
            skip_special: Omitir tokens especiales.

        Returns:
            Texto decodificado.
        """
        tokens = []
        special_ids = {PAD_ID, UNK_ID, CLS_ID, SEP_ID, MASK_ID}

        for token_id in ids:
            if skip_special and token_id in special_ids:
                continue
            token = self.id_to_token.get(token_id, UNK_TOKEN)
            if skip_special and token.startswith("[UNUSED_"):
                continue
            tokens.append(token)

        return "".join(tokens)

    @property
    def pad_token_id(self) -> int:
        return PAD_ID

    @property
    def cls_token_id(self) -> int:
        return CLS_ID

    @property
    def sep_token_id(self) -> int:
        return SEP_ID

    @property
    def mask_token_id(self) -> int:
        return MASK_ID

    @property
    def unk_token_id(self) -> int:
        return UNK_ID
