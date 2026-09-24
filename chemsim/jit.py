"""Необязательная JIT-компиляция через numba.

Если numba не установлена, функции выполняются как обычный Python (корректно,
но в сотни раз медленнее — годится только для крошечных тестов).

Кэш скомпилированного кода numba проверяет только файл самой функции, но не
файлы вызываемых ею функций.  Поэтому каталог кэша привязывается к хэшу всех
модулей с JIT-кодом: любое их изменение даёт чистую перекомпиляцию.
"""

import hashlib
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_JIT_SOURCES = ("jit.py", "kernel.py", "md.py", "params.py")


def _source_hash():
    h = hashlib.sha1()
    for name in _JIT_SOURCES:
        try:
            with open(os.path.join(_HERE, name), "rb") as fh:
                h.update(fh.read())
        except OSError:
            pass
    return h.hexdigest()[:12]


if "NUMBA_CACHE_DIR" not in os.environ:
    os.environ["NUMBA_CACHE_DIR"] = os.path.join(_HERE, "__pycache__", "numba-" + _source_hash())

try:  # pragma: no cover - зависит от окружения
    from numba import njit as _njit

    HAVE_NUMBA = True

    def njit(func=None, **kwargs):
        kwargs.setdefault("cache", True)
        if func is not None:
            return _njit(**kwargs)(func)
        return _njit(**kwargs)

except ImportError:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(func=None, **kwargs):
        if func is not None:
            return func
        return lambda f: f
