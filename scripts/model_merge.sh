export PYTHONPATH="${PYTHONPATH:-}:$PWD/HTMR"

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir ... \
    --target_dir ...