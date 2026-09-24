"""Дымовой тест интерфейса: все инструменты, клавиши и окна без ошибок."""

import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
pygame = pytest.importorskip("pygame")


def test_ui_smoke(tmp_path):
    from chemsim.scenarios import SCENARIOS
    from chemsim.ui.app import App

    app = App(1400, 860, warmup_thread=False)
    app._warmup()                      # синхронно (в игре — в фоновом потоке)
    app.update()                       # ready -> run, загружается первый опыт
    assert app.state == "run"
    app.draw()
    rc = app.view.rect
    center = rc.center

    def click(pos, button=1):
        app.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=button))
        app.handle(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=pos, button=button))

    def key(k, mod=0):
        app.handle(pygame.event.Event(pygame.KEYDOWN, key=k, mod=mod, unicode=""))

    # все кнопки-вещества и инструменты
    for b in list(app.buttons):
        if b.group in ("species", "tool"):
            b.on_click(b)
            app.update()
            app.draw()
    # каждый инструмент в реакторе
    n0 = app.sim.n
    app._set_tool(type("B", (), {"value": "add"})())
    app.selected = "H2O"
    click(center)
    assert app.sim.n > n0
    for tool in ("spark", "heat", "cool", "erase"):
        app._set_tool(type("B", (), {"value": tool})())
        app.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=center, button=1))
        app.update()
        app.handle(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=center, button=1))
        app.draw()
    click(center, button=3)            # ПКМ — искра
    app.handle(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=1))
    app.handle(pygame.event.Event(pygame.MOUSEMOTION, pos=center, rel=(0, 0), buttons=(0, 0, 0)))
    # клавиши
    for k in (pygame.K_SPACE, pygame.K_SPACE, pygame.K_PLUS, pygame.K_MINUS,
              pygame.K_RIGHTBRACKET, pygame.K_LEFTBRACKET, pygame.K_m, pygame.K_q,
              pygame.K_b, pygame.K_l, pygame.K_LEFT, pygame.K_RIGHT):
        key(k)
        app.update()
        app.draw()
    # окна: справка и опыты
    key(pygame.K_h)
    app.draw()
    key(pygame.K_ESCAPE)
    key(pygame.K_o)
    app.draw()
    assert app.overlay == "scenarios" and len(app.scenario_rects) == len(SCENARIOS)
    click(app.scenario_rects[3][0].center)
    assert app.current_scenario is SCENARIOS[3]
    for _ in range(5):
        app.update()
        app.draw()
    # все кнопки нижней панели
    for b in list(app.buttons):
        if b.group not in ("species", "tool") and b.on_click:
            b.on_click(b)
            app.overlay = None
            app.update()
            app.draw()
    # изменение размера окна
    app.handle(pygame.event.Event(pygame.VIDEORESIZE, w=1200, h=760, size=(1200, 760)))
    app.update()
    app.draw()
    path = app.screenshot(str(tmp_path / "shot.png"))
    assert os.path.exists(path)
    pygame.quit()
