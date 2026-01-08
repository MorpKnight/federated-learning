$control = "powershell -NoExit -Command \"python -m control.api --config config/control.yaml\""
$server = "powershell -NoExit -Command \"python -m fl_server.server --config config/server.yaml\""
$client1 = "powershell -NoExit -Command \"python -m fl_client.client --config config/client.yaml --client-id client1\""
$client2 = "powershell -NoExit -Command \"python -m fl_client.client --config config/client.yaml --client-id client2\""

Start-Process powershell -ArgumentList "-NoExit", "-Command", $control
Start-Process powershell -ArgumentList "-NoExit", "-Command", $server
Start-Process powershell -ArgumentList "-NoExit", "-Command", $client1
Start-Process powershell -ArgumentList "-NoExit", "-Command", $client2
