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
- M0 — Repo bootstrap: struktur folder, venv, requirements, konfigurasi dasar.
- M1 — Training Plane MVP (Flower): fl_server + fl_client jalan, dataset dummy lokal, bisa run 1 server + 2 client dari satu PC (terminal berbeda).
- M2 — Metrics logging ke SQLite: server simpan metrik global per round, client kirim metrik per round (atau server catat saat receive), schema DB + migrasi sederhana.
- M3 — Control API MVP (FastAPI): register client + config endpoint; client saat start: register → ambil config → connect FL server.
- M4 — Auto hyperparameter (rule-based): deteksi hardware sederhana (CPU count, RAM, GPU ada/tidak) dan rekomendasi batch_size/epochs/lr dari aturan statis.
- M5 — Dashboard minimal: tampilkan tabel round dan tabel per-client; opsional plot sederhana.
