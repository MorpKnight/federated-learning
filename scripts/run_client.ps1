param(
    [string]$Config = "config/client.yaml",
    [string]$ClientId = "client1"
)
python -m fl_client.client --config $Config --client-id $ClientId
