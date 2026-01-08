# FL System (Bootstrap)

Repository bootstrap untuk sistem Federated Learning multi-PC berbasis Flower.

## Prasyarat
- Python 3.10+

## Setup (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Setup (Linux/macOS)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Setup (1 command, all OS)
```bash
python scripts/setup.py
```

## Menjalankan Server Pusat (M1–M2)
Terminal 1 (Control API + DB):
```bash
python -m control_api.main --config configs/control_api.yaml
```

Terminal 2 (Portal Web):
```bash
uvicorn portal_web.main:app --port 9000
```

Portal:
- http://127.0.0.1:9000
- http://127.0.0.1:9000/download

## Menjalankan Client UI (M4, nanti)
Jalankan UI lokal:
```bash
uvicorn client_app.client_ui.main:app --port 7000
```

Buka browser:
- http://localhost:7000

Alur cepat:
1) Isi Control API URL + FL Server Address.
2) Klik Register/Connect.
3) Pilih mode Manual/Auto dan klik Start Training.
4) Lihat log di panel bawah, Stop untuk menghentikan proses.

## Menjalankan Client Desktop (Flet, M4)
```bash
python -m client_app.client_desktop.app
```

## Demo 1 perintah (Linux/macOS/WSL)
```bash
./scripts/run_local_demo.sh
```

## Konfigurasi penting
- `configs/control_api.yaml`: host/port Control API + default FL server address.
- `configs/server.yaml`: address Flower server + jumlah round + DB path.
- `configs/client.yaml`: address server, control API URL, dan hyperparameter.

## Lihat isi SQLite (M2)
DB default tersimpan di `data/metrics.db`.

Contoh query:
```bash
sqlite3 data/metrics.db "SELECT round, num_clients, loss, accuracy FROM round_metrics ORDER BY id DESC LIMIT 5;"
sqlite3 data/metrics.db "SELECT round, client_id, loss, accuracy, num_examples FROM client_metrics ORDER BY id DESC LIMIT 5;"
```
