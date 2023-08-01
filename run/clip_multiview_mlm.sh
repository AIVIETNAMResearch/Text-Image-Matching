name='clip_multiview_mlm_mae'
config='configs/clip_multiview_mlm.yaml'

python3 main.py --name $name --config $config --use_wandb True \
--logs-dir logs/${name} 
