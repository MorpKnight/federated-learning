Arsitektur:

Training Plane (Flower):
- fl_server: Flower Server menjalankan strategi FedAvg, mengatur round, sampling client, menerima updates.
- fl_client: Flower Client melakukan training lokal pada dataset lokal, kirim parameters+metrics ke server.

Control Plane (Web/API):
- control_api: FastAPI service untuk:
  - /register: membuat/menyimpan client_id + metadata (os, cpu/gpu, ram, dataset_size)
  - /config/{client_id}: mengembalikan config untuk client (server_address, default hyperparams, policy)
  - /heartbeat: update status client online
- metrics_store: SQLite untuk menyimpan:
  - rounds (round_id, timestamp, num_clients, global_metrics)
  - client_metrics (round_id, client_id, loss, acc, fit_time, num_examples)
- dashboard: halaman sederhana yang membaca DB dan menampilkan tabel + grafik sederhana (opsional tahap awal).

Requirement MVP:
- 1 perintah untuk start server
- 1 perintah untuk start client (bisa multiple di terminal berbeda)
- Setelah N round, tercetak metrik global dan tersimpan di SQLite.

Milestone Plan (urutan kerja yang aman):
- M0 — Bootstrap repo.
- M1 — Control API minimal + DB.
- M2 — Client UI lokal (agent + UI) yang bisa register & fetch config.
- M3 — Training Plane Flower server+client (MVP).
- M4 — Integrasi UI -> start/stop training + tampilkan metrics.
- M5 — Auto hyperparameter.
- M6 — Dashboard server (opsional).
