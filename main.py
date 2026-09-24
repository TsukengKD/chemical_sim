"""Запуск интерактивного химического симулятора:  python main.py"""

import importlib.util
import sys

REQUIRED = {"numpy": "numpy", "numba": "numba", "pygame": "pygame-ce"}


def _check_dependencies():
    missing = [pkg for mod, pkg in REQUIRED.items() if importlib.util.find_spec(mod) is None]
    if not missing:
        return True
    print("Не установлены нужные пакеты: " + ", ".join(missing))
    print()
    print("Установите их командой (из папки с игрой):")
    print(f'    "{sys.executable}" -m pip install -r requirements.txt')
    print()
    print(f"Ваша версия Python: {sys.version.split()[0]}.")
    print("Если pip не может установить какой-то пакет, скорее всего для вашей версии")
    print("Python ещё нет готовой сборки: поставьте Python 3.12 с python.org.")
    return False


if __name__ == "__main__":
    if not _check_dependencies():
        sys.exit(1)
    from chemsim.ui.app import main

    main()
