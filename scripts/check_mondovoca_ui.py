"""Real browser/synthesis checks against an isolated temporary studio.

Run with a Python environment containing Playwright. --python selects the
tiny-voca environment. No production history or browser preferences are touched.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import urllib.request

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True)
    parser.add_argument("--chromium", required=True)
    parser.add_argument("--firefox")
    parser.add_argument("--shots", default="/tmp/mondovoca-ui")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    shots = Path(args.shots)
    shots.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mondovoca-ui-") as temp:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        env = dict(os.environ, PYTHONPATH=str(root / "src"))
        log = (Path(temp) / "server.log").open("w+")
        server = subprocess.Popen([args.python, "-m", "mondovoca.server", "--port", str(port),
                                   "--data-dir", temp], env=env, stdout=log, stderr=log, cwd=root)
        try:
            for _ in range(100):
                try:
                    with urllib.request.urlopen(url + "/api/health", timeout=.2) as response:
                        assert json.load(response)["ready"]
                    break
                except OSError:
                    time.sleep(.1)
            else:
                raise AssertionError("Test studio failed to start")
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(executable_path=args.chromium, headless=True)
                page = browser.new_page(viewport={"width": 1440, "height": 1000})
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(url)
                page.wait_for_function("() => !document.getElementById('generate').disabled")
                assert page.locator('[name=voice]:checked').input_value() == "F2"
                assert page.locator("#text").input_value() == ""
                assert page.locator("#text").get_attribute("placeholder")
                assert page.locator("#characters").inner_text() == "0 / 5,000"
                page.locator("#text").press_sequentially("Hello")
                assert page.locator("#text").input_value() == "Hello"
                page.locator("#example").click()
                assert page.locator("#text").input_value().startswith("Welcome to the workshop.")
                page.locator("#age").fill("7")
                assert page.locator("#age-range").input_value() == "7"
                assert float(page.locator("#pitch").input_value()) > 3
                page.locator("#quick-age").select_option("80")
                assert page.locator("#age").input_value() == "80"
                assert float(page.locator("#speed").input_value()) < 1
                page.locator("#use-age").uncheck()
                assert page.locator("#age").is_disabled()
                assert page.locator("#speed").input_value() == "1"
                page.locator("#use-age").check()
                page.locator("#quick-age").select_option("30")
                page.locator("#preset-name").fill("Narrator")
                page.locator("#save-preset").click()
                page.locator("#quick-age").select_option("5")
                page.locator("#presets").select_option("Narrator")
                assert page.locator("#age").input_value() == "30"
                page.locator("#theme").select_option("purple")
                page.reload()
                page.wait_for_function("() => !document.getElementById('generate').disabled")
                assert page.locator("html").get_attribute("data-theme") == "purple"
                page.locator("#theme").select_option("teal")
                page.locator("#text").fill("Welcome to Mondovoca. A small voice with a story to tell.")
                page.locator("#autoplay").uncheck()
                page.locator("#generate").click()
                page.wait_for_function("() => document.querySelectorAll('.take').length === 1", timeout=60000)
                page.wait_for_function("() => !document.getElementById('generate').disabled")
                assert page.locator("audio").evaluate("a => a.src.includes('/clips/')")
                assert page.locator("audio").evaluate("a => a.volume") == 0.4
                clip = page.locator("audio").get_attribute("src")
                response = page.request.get(url + clip)
                assert response.body()[:4] == b"RIFF"
                assert len(response.body()) > 1000
                with page.expect_download() as pending:
                    page.get_by_role("button", name="Export recipe").first.click()
                recipe = json.loads(Path(pending.value.path()).read_text())
                assert recipe["voice"] == "F2" and recipe["text"].startswith("Welcome")
                page.locator("#compare").click()
                page.wait_for_function("() => document.querySelectorAll('.take').length === 3", timeout=60000)
                page.wait_for_function("() => !document.getElementById('generate').disabled")
                headings = page.locator(".take h3").all_text_contents()
                assert any("F2" in heading for heading in headings)
                assert any("M2" in heading for heading in headings)
                assert page.locator("audio").evaluate_all("players => players.every(a => a.volume === 0.4)")
                page.screenshot(path=str(shots / "desktop.png"))
                # The retro skins imported from Mondocraft keep one hue on pure black and set a
                # monospace page. The palette must still drive the studio's own markup, and the
                # scanlines and phosphor glow that studio.css maps onto that markup must switch
                # off under reduced motion.
                for theme, ink, sidebar in (("crtgreen", "rgb(51, 255, 51)", "rgb(7, 21, 7)"),
                                            ("crtamber", "rgb(255, 176, 0)", "rgb(21, 14, 0)")):
                    page.locator("#theme").select_option(theme)
                    assert page.locator("html").get_attribute("data-theme") == theme, theme
                    body = page.evaluate("() => {const style = getComputedStyle(document.body);"
                                         "return [style.backgroundColor, style.color, style.fontFamily];}")
                    assert body[0] == "rgb(0, 0, 0)" and body[1] == ink, body
                    assert "monospace" in body[2], body
                    assert page.evaluate("() => getComputedStyle(document.querySelector('aside')).backgroundColor") == sidebar, theme
                    assert page.evaluate("() => getComputedStyle(document.querySelector('aside')).backgroundImage") != "none", theme
                    assert page.evaluate("() => getComputedStyle(document.querySelector('.results-bar strong')).textShadow") != "none", theme
                    page.screenshot(path=str(shots / f"{theme}-desktop.png"))
                page.emulate_media(reduced_motion="reduce")
                decorations = page.evaluate("() => [getComputedStyle(document.querySelector('aside')).backgroundImage,"
                                            "getComputedStyle(document.querySelector('h2')).textShadow,"
                                            "getComputedStyle(document.querySelector('.results-bar strong')).textShadow];")
                assert decorations == ["none", "none", "none"], decorations
                page.emulate_media(reduced_motion="no-preference")
                page.set_viewport_size({"width": 390, "height": 844})
                for theme in ("crtgreen", "crtamber"):
                    page.locator("#theme").select_option(theme)
                    assert page.locator("html").get_attribute("data-theme") == theme, theme
                    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), theme
                    page.screenshot(path=str(shots / f"{theme}-mobile.png"), full_page=True)
                page.set_viewport_size({"width": 1440, "height": 1000})
                page.locator("#theme").select_option("teal")
                page.locator("#integration").click()
                assert page.locator("#connection").is_visible()
                assert json.loads(page.locator("#api-params").inner_text())["response_format"] == "wav"
                page.keyboard.press("Escape")
                assert not page.locator("#connection").is_visible()
                page.locator("#text").fill("This is a longer sentence for a cancellation test. " * 80)
                page.locator("#generate").click()
                page.locator("#stop").click()
                page.wait_for_function("() => !document.getElementById('generate').disabled", timeout=30000)
                assert page.locator("#status").inner_text() == "Stopped."
                assert page.locator(".take").count() == 3
                page.reload()
                page.wait_for_function("() => document.querySelectorAll('.take').length === 3")
                assert page.locator("audio").evaluate_all("players => players.every(a => a.volume === 0.4)")
                page.set_viewport_size({"width": 390, "height": 844})
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                page.screenshot(path=str(shots / "mobile.png"), full_page=True)
                assert not errors, errors
                browser.close()
                if args.firefox:
                    browser = playwright.firefox.launch(executable_path=args.firefox, headless=True)
                    page = browser.new_page(no_viewport=True)
                    page.on("pageerror", lambda error: errors.append(str(error)))
                    page.goto(url)
                    page.wait_for_function("() => !document.getElementById('generate').disabled")
                    page.locator("#quick-age").select_option("65")
                    assert page.locator("#age").input_value() == "65"
                    page.locator("#text").fill("This is the Firefox voice test.")
                    page.locator("#autoplay").uncheck()
                    page.locator("#generate").click()
                    page.wait_for_function("() => document.querySelectorAll('.take').length === 4", timeout=60000)
                    page.screenshot(path=str(shots / "firefox.png"))
                    assert not errors, errors
                    browser.close()
            # The exact Open WebUI discovery paths are covered by unit tests; verify real speech bytes here.
            request = urllib.request.Request(url + "/v1/audio/speech", data=json.dumps({
                "model": "tiny-voca", "input": "The local speech connection works.", "voice": "F2",
                "age": 70, "response_format": "wav"}).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=60) as response:
                assert response.headers["Content-Type"] == "audio/wav"
                assert response.read(4) == b"RIFF"
            print(json.dumps({"result": "passed", "checks": ["age controls", "saved presets", "themes",
                  "retro CRT themes", "real WAV",
                  "recipe export", "both voices", "cancellation", "history reload", "mobile layout", "Open WebUI speech API"]
                  + (["Firefox generation"] if args.firefox else []), "screenshots": str(shots)}))
        finally:
            server.terminate()
            server.wait(timeout=15)
            log.close()


if __name__ == "__main__":
    main()
