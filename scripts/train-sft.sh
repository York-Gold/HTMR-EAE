# pip install swanlab
export SWANLAB_API_KEY="..."
export PYTHONPATH="${PYTHONPATH:-}:$PWD/HTMR"

torchrun --nproc_per_node=4 -m verl.trainer.fsdp_sft_trainer \
    data.train_files=... \
    data.val_files=... \
    data.prompt_key=prompt \
    data.response_key=completion \
    data.micro_batch_size_per_gpu=2 \
    optim.lr=1e-5 \
    data.max_length=8192 \
    data.train_batch_size=16 \
    model.partial_pretrain=... \
    trainer.total_epochs=3 \
    trainer.logger='["console","swanlab"]' \
    trainer.project_name="SFT-VERL" \
    trainer.experiment_name="..." \
