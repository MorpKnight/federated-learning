const logsEl = document.getElementById("logs");

function appendLog(line) {
  if (!line) return;
  logsEl.textContent += line + "\n";
  logsEl.scrollTop = logsEl.scrollHeight;
}

async function loadStatus() {
  const res = await fetch("/status");
  const data = await res.json();

  document.getElementById("controlUrl").value = data.config.control_api.url || "";
  document.getElementById("serverAddr").value = data.config.fl_server.address || "";
  document.getElementById("clientId").value = data.config.client_id || "client1";
  document.getElementById("batchSize").value = data.config.train.batch_size || 32;
  document.getElementById("epochs").value = data.config.train.epochs || 1;
  document.getElementById("lr").value = data.config.train.lr || 0.01;
  document.getElementById("hpMode").value = data.config.train.auto ? "auto" : "manual";

  const connection = data.connection || {};
  let connectionLabel = "no";
  if (connection.connected) {
    connectionLabel = "Connected";
  } else if (connection.registered && connection.config_fetched) {
    connectionLabel = "Registered + Config fetched";
  }
  document.getElementById("connected").textContent = connectionLabel;
  document.getElementById("running").textContent = data.process.running ? "yes" : "no";
  document.getElementById("lastLoss").textContent = data.process.last_loss ?? "-";
  document.getElementById("lastAcc").textContent = data.process.last_acc ?? "-";

  const d = data.device;
  document.getElementById("deviceInfo").textContent =
    `${d.cpu_count} CPU, ${d.ram_gb} GB, GPU=${d.gpu_available}`;
  document.getElementById("datasetInfo").textContent =
    data.dataset.num_samples !== null ? data.dataset.num_samples : "-";
}

async function saveConfig() {
  const payload = {
    client_id: document.getElementById("clientId").value,
    control_api: { url: document.getElementById("controlUrl").value },
    fl_server: { address: document.getElementById("serverAddr").value },
    train: {
      batch_size: parseInt(document.getElementById("batchSize").value, 10),
      epochs: parseInt(document.getElementById("epochs").value, 10),
      lr: parseFloat(document.getElementById("lr").value),
      auto: document.getElementById("hpMode").value === "auto",
      device: "auto",
    },
  };
  await fetch("/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  await loadStatus();
}

async function register() {
  await fetch("/register", { method: "POST" });
  await loadStatus();
}

async function startTraining() {
  await fetch("/start", { method: "POST" });
  await loadStatus();
}

async function stopTraining() {
  await fetch("/stop", { method: "POST" });
  await loadStatus();
}

function startLogStream() {
  const es = new EventSource("/logs/stream");
  es.onmessage = (event) => appendLog(event.data);
  es.onerror = () => {
    es.close();
    appendLog("[log stream disconnected]");
  };
}

document.getElementById("btnSave").addEventListener("click", saveConfig);
document.getElementById("btnRegister").addEventListener("click", register);
document.getElementById("btnStart").addEventListener("click", startTraining);
document.getElementById("btnStop").addEventListener("click", stopTraining);

loadStatus();
startLogStream();
