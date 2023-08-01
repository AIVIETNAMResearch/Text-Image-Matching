name='clip_multiview_text_aug'
config='configs/clip_multiview.yaml'

python3 main.py --name $name --config $config --use_wandb True \
--logs-dir logs/${name} 
