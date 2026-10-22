from pathlib import Path
import json
from playwright.sync_api import sync_playwright

root = Path(__file__).resolve().parents[1]
out = root / "local/html_check"
out.mkdir(exist_ok=True)
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
    page = browser.new_page(
        viewport={"width": 1440, "height": 1000}, device_scale_factor=1
    )
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto((root / "docs/index.html").as_uri())
    page.screenshot(path=str(out / "desktop.png"), full_page=False)
    assert page.locator("h1").count() == 1
    assert page.locator("main section").count() == 12
    assert page.locator("#forceDemo").inner_text().find("0.0082") >= 0
    page.locator("#sharedBias").click()
    assert "0.4000" in page.locator("#forceDemo").inner_text()
    broken = page.evaluate(
        """() => [...document.querySelectorAll('a[href^="#"]')].filter(a=>!document.querySelector(a.getAttribute('href'))).map(a=>a.getAttribute('href'))"""
    )
    assert not broken, broken
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    page.set_viewport_size({"width": 390, "height": 844})
    page.goto((root / "docs/index.html").as_uri())
    page.locator(".mobile-menu").click()
    assert page.locator("#toc").is_visible()
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    page.screenshot(path=str(out / "mobile.png"), full_page=False)
    assert not errors, errors
    browser.close()
    (out / "result.json").write_text(
        json.dumps(
            {
                "desktop_width": 1440,
                "mobile_width": 390,
                "sections": 12,
                "interactive_demo": "passed",
                "anchor_links": "passed",
                "horizontal_overflow": False,
                "page_errors": errors,
            },
            indent=2,
        )
    )
    print("HTML checks passed")
