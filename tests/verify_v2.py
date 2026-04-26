from playwright.sync_api import sync_playwright
import os

def run_verification(page):
    # Go to main dashboard
    page.goto("http://localhost:8000")
    page.wait_for_timeout(2000)

    # Check Model Accuracy
    acc_text = page.locator("#sAcc").inner_text()
    print(f"Model Accuracy: {acc_text}")
    assert "undefined" not in acc_text, "Model Accuracy should not be undefined"

    # Check Active Trades Panel (it might be hidden if no trades)
    active_panel = page.locator("#activeTradesPanel")
    print(f"Active Trades Panel Visible: {active_panel.is_visible()}")

    # Check Tabs
    history_tab = page.locator("text=İşlem Geçmişi")
    assert history_tab.is_visible(), "History tab should be visible"
    history_tab.click()
    page.wait_for_timeout(1000)

    # Take screenshot
    page.screenshot(path="/home/jules/verification/screenshots/dashboard_v2.png")

    # Check ML Details page again to be sure
    page.goto("http://localhost:8000/ml-details")
    page.wait_for_timeout(2000)
    page.screenshot(path="/home/jules/verification/screenshots/ml_details_v2.png")

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
            # We need the backend running for this
            run_verification(page)
        except Exception as e:
            print(f"Verification failed: {e}")
        finally:
            context.close()
            browser.close()
