import hashlib
import time
from pathlib import Path
from urllib.request import urlretrieve

DATA_URL = "https://raw.githubusercontent.com/pparshakov/data/main/hh_vacancies_2023_2026.parquet"
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
DATA_FILE = RAW_DIR / "vacancies.parquet"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rel_path = DATA_FILE.relative_to(ROOT)
    t0 = time.time()
    if DATA_FILE.exists():
        print(f"Файл уже существует, повторно не скачиваю: {rel_path}")
    else:
        print(f"Скачиваю {DATA_URL} -> {rel_path}")
        urlretrieve(DATA_URL, DATA_FILE)
        print(f"Скачано за {time.time() - t0:.1f} с")

    size_mb = DATA_FILE.stat().st_size / (1024 * 1024)
    checksum = sha256_of(DATA_FILE)
    print(f"Размер: {size_mb:.2f} МБ")
    print(f"SHA-256: {checksum}")


if __name__ == "__main__":
    main()
