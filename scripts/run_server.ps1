param(
    [string]$Config = "config/server.yaml"
)
python -m fl_server.server --config $Config
