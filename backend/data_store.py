import json
import os
import time
import shutil
from pathlib import Path

DATA_DIR = Path(os.environ.get("PERSIST_DIR", Path(__file__).parent.parent / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE = DATA_DIR / "trading_state.json"
BACKUP_FILE = DATA_DIR / "trading_state_backup.json"


def save_state(
    trade_history: list,
    open_positions: list,
    total_equity: float,
    available_capital: float,
    cycle_count: int,
    trade_count: int,
):
    state = {
        "version": 2,
        "saved_at": time.time(),
        "saved_at_human": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_equity": total_equity,
        "available_capital": available_capital,
        "cycle_count": cycle_count,
        "trade_count": trade_count,
        "trade_history": trade_history,
        "open_positions": open_positions,
    }

    try:
        if STATE_FILE.exists():
            shutil.copy2(STATE_FILE, BACKUP_FILE)

        tmp = STATE_FILE.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        tmp.replace(STATE_FILE)
    except Exception as e:
        print(f"[DATA STORE] Kayit hatasi: {e}")


def load_state() -> dict | None:
    for fpath in [STATE_FILE, BACKUP_FILE]:
        try:
            if fpath.exists():
                with open(fpath, "r", encoding="utf-8") as f:
                    state = json.load(f)
                print(f"[DATA STORE] {fpath.name} yuklendi - {len(state.get('trade_history', []))} islem, {len(state.get('open_positions', []))} pozisyon, ${state.get('total_equity', 0):.2f} bakiye")
                return state
        except Exception as e:
            print(f"[DATA STORE] {fpath.name} yukleme hatasi: {e}")

    print("[DATA STORE] Kayitli veri bulunamadi, sifirdan basliyor")
    return None


def get_data_dir() -> Path:
    return DATA_DIR
