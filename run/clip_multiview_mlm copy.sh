name='clip_multiview_mlm'
config='configs/clip_multiview_mlm.yaml'

python3 main.py --name $name --config $config --use_wandb True \
--logs-dir logs/${name} 
