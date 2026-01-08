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

## Menjalankan Server & Client (M1–M5)
Buka 4 terminal (atau gunakan script demo di bawah).

Terminal 1 (Control API + Dashboard):
```bash
python -m control_api.main --config configs/control_api.yaml
```

Terminal 2 (Flower Server):
```bash
python -m fl_server.server --config configs/server.yaml
```

Terminal 3 (Client 1):
```bash
python -m fl_client.client --config configs/client.yaml --client-id client1
```

Terminal 4 (Client 2):
```bash
python -m fl_client.client --config configs/client.yaml --client-id client2
```

Dashboard:
- http://127.0.0.1:8000/dashboard

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
