from playwright.sync_api import sync_playwright
import os

def run_cuj(page):
    # Go to main dashboard
    page.goto("http://localhost:8000")
    page.wait_for_timeout(1000)

    # Take screenshot of dashboard
    page.screenshot(path="/home/jules/verification/screenshots/dashboard.png")
    page.wait_for_timeout(500)

    # Click on ML Detayları button
    # Using text "ML Detayları" since it's a link
    page.get_by_text("ML Detayları").click()
    page.wait_for_timeout(1000)

    # Verify we are on ml-details page
    # It should have the title or some specific text
    page.wait_for_selector("text=ML Detayları & Veri Akışı")
    page.wait_for_timeout(1000)

    # Take screenshot of ML details page
    page.screenshot(path="/home/jules/verification/screenshots/ml_details.png")
    page.wait_for_timeout(1000)

    # Click back to dashboard
    page.get_by_text("← Dashboard'a Dön").click()
    page.wait_for_timeout(1000)

    # Verify back on dashboard
    page.wait_for_selector("text=MEXC · ML TRADER")
    page.wait_for_timeout(1000)

if __name__ == "__main__":
    os.makedirs("/home/jules/verification/videos", exist_ok=True)
    os.makedirs("/home/jules/verification/screenshots", exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            record_video_dir="/home/jules/verification/videos",
            viewport={'width': 1280, 'height': 800}
        )
        page = context.new_page()
        try:
            run_cuj(page)
        finally:
            context.close()
            browser.close()
