name='clip_multiprompt'
config='configs/clip_multiprompt.yaml'

python3 main.py --name $name --config $config \
--logs-dir logs/${name} 
