name='baseline'
config='configs/baseline.yaml'

python3 main.py --name $name --config $config \
--logs-dir logs/${name} \