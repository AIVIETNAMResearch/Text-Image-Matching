name='clip_multigrained'
config='configs/clip_multigrained.yaml'

python3 main.py --name $name --config $config --use_wandb True \
--logs-dir logs/${name} 
