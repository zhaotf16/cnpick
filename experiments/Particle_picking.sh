python main.py ctdet --exp_id GspDvc_dla --batch_size 32 --master_batch 1 --lr 1.25e-4  --gpus 0,1,2,3

python main.py ctdet --exp_id TrpV1_dla --dataset TrpV1_512 --batch_size 32 --master_batch 1 --lr 5e-4  --gpus 0,1,2,3

python demo.py ctdet --demo ../images/GspDvc_512 --load_model ../exp/ctdet/GspDvc_dla/model_best.pth

python demo.py ctdet --demo ../images/GspDvc_512 --load_model ../exp/ctdet/GspDvc_dla/model_best.pth --K 1000 --vis_thresh 0.3