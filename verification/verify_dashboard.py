from playwright.sync_api import sync_playwright

def run_cuj(page):
    # Wait for server to be ready
    page.goto("http://localhost:8000")
    page.wait_for_timeout(2000)

    # Wait for data to load
    page.wait_for_selector(".sc-card", timeout=30000)
    page.wait_for_timeout(2000)

    # Take screenshot of the dashboard
    page.screenshot(path="verification/screenshots/dashboard.png")

    # Click on some tabs to show logs/history
    page.click("text=Sistem Logu")
    page.wait_for_timeout(1000)
    page.screenshot(path="verification/screenshots/logs.png")

    page.click("text=PnL Grafiği")
    page.wait_for_timeout(1000)

    # Scroll down to see scanner
    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    page.wait_for_timeout(2000)
    page.screenshot(path="verification/screenshots/scanner.png")
    page.wait_for_timeout(1000)

if __name__ == "__main__":
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            record_video_dir="verification/videos"
        )
        page = context.new_page()
        try:
            run_cuj(page)
        finally:
            context.close()
            browser.close()
