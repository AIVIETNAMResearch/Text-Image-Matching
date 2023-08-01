name='clip_multiview_gnn'
config='configs/clip_multiview_gnn.yaml'

python3 main.py --name $name --config $config \
--logs-dir logs/${name} 
