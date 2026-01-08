param(
    [string]$Config = "config/control.yaml"
)
python -m control.api --config $Config
