name='test_result'
config='configs/clip_multiview_mlm.yaml'

python3 test.py --name $name --config $config -r True --eval_only \
--logs-dir logs/${name} 