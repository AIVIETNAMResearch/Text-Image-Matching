name='clip_baseline_2'
config='configs/clip_baseline.yaml'

python3 main.py --name $name --config $config --use_wandb True \
--logs-dir logs/${name} \