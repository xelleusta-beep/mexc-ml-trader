from playwright.sync_api import sync_playwright
import os
import json

def run_verification(page):
    # Go to main dashboard
    page.goto("http://localhost:8000")
    page.wait_for_timeout(3000)

    # Check Capital Status (Kasa Durumu)
    cap_val = page.locator("#sCap").inner_text()
    print(f"Capital Value: {cap_val}")
    assert "$" in cap_val, "Capital should be formatted as currency"

    # Check ML Ensemble detail on main page
    arch = page.locator("#mArch").inner_text()
    print(f"Architecture Detail: {arch}")
    assert "GradientBoosting" in arch, "Architecture should be detailed"

    page.screenshot(path="/home/jules/verification/screenshots/dashboard_v4.png")

    # Go to ML Details page
    page.goto("http://localhost:8000/ml-details")
    page.wait_for_timeout(2000)

    # Check Feature Engineering section
    feat_status = page.locator("#featStatus").inner_text()
    print(f"Feature Status: {feat_status}")
    assert "Durum" in feat_status

    # Check estimators
    gbm_est = page.locator("#gbmEst").inner_text()
    print(f"GBM Estimators: {gbm_est}")
    assert gbm_est == "300"

    page.screenshot(path="/home/jules/verification/screenshots/ml_details_v4.png")

def test_persistence():
    # Check if persistence file is created
    if os.path.exists("persistence.json"):
        print("Persistence file exists.")
        with open("persistence.json", "r") as f:
            data = json.load(f)
            print(f"Portfolio in file: {data.get('portfolio')}")
    else:
        print("Persistence file not found yet.")

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
            test_persistence()
        except Exception as e:
            print(f"Verification failed: {e}")
        finally:
            context.close()
            browser.close()
