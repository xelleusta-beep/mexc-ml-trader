from playwright.sync_api import sync_playwright
import os

def run_verification(page):
    # Go to main dashboard
    page.goto("http://localhost:8000")
    page.wait_for_timeout(3000)

    # Verify auto-connect (connPill should be conn-ok or conn-wait/REST)
    conn_pill = page.locator("#connPill")
    print(f"Connection Status Class: {conn_pill.get_attribute('class')}")

    # Check Volume Formatting in Scanner
    # Look for B or M in the volume text
    sc_vol = page.locator(".sc-vol").first.inner_text()
    print(f"Scanner Volume Example: {sc_vol}")
    assert ("B" in sc_vol or "M" in sc_vol), f"Volume formatting failed: {sc_vol}"

    # Check Ticker Volume
    ticker_text = page.locator("#tickerEl").inner_text()
    print(f"Ticker Example: {ticker_text[:50]}...")
    assert "V:$" in ticker_text, "Ticker volume missing"

    # Check Position Counts
    pos_count = page.locator("#sPos").inner_text()
    print(f"Live Position Count: {pos_count}")

    # Check PnL subtext
    pnl_sub = page.locator("#sPnlSub").inner_text()
    print(f"PnL Subtext: {pnl_sub}")
    assert "Closed" in pnl_sub or "Hold" in pnl_sub, f"PnL subtext missing info: {pnl_sub}"

    # Take screenshot
    page.screenshot(path="/home/jules/verification/screenshots/dashboard_v3.png")

    # ML Details Page
    page.goto("http://localhost:8000/ml-details")
    page.wait_for_timeout(2000)

    gbm_backend = page.locator("#gbmBackend").inner_text()
    print(f"GBM Backend: {gbm_backend}")

    page.screenshot(path="/home/jules/verification/screenshots/ml_details_v3.png")

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
            run_verification(page)
        except Exception as e:
            print(f"Verification failed: {e}")
        finally:
            context.close()
            browser.close()
