export VLLM_WORKER_MULTIPROC_METHOD=spawn

python infer.py \
    --model_path ... \
    --dataset_type ... \
    --result_path ... \
    --use_vllm True 