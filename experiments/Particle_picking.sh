python main.py ctdet --exp_id GspDvc_1024 --dataset GspDvc_1024 --batch_size 16 --master_batch 1 --lr 5e-4  --gpus 0,1,2,3

python main.py ctdet --exp_id TrpV1_1024 --dataset TrpV1_1024 --batch_size 16 --master_batch 1 --lr 5e-4  --gpus 0,1,2,3

python main.py ctdet --exp_id proteasome_1024 --dataset proteasome --batch_size 16 --master_batch 1 --lr 2.5e-4  --gpus 0,1,2,3

python main.py ctdet --exp_id proteasome_512 --dataset proteasome_512 --batch_size 32 --master_batch 1 --lr 5e-4 --gpus 0,1,2,3 --num_epochs 110

python main.py ctdet --exp_id proteasome_1024 --dataset proteasome --batch_size 16 --master_batch 1 --lr 5e-4 --gpus 0,1,2,3 --num_epochs 140     

python demo.py ctdet --demo ../images/GspDvc_512 --load_model ../exp/ctdet/GspDvc_dla/model_best.pth

python demo.py ctdet --demo ../images/GspDvc_512 --load_model ../exp/ctdet/GspDvc_dla/model_best.pth --K 1000 --vis_thresh 0.3

python demo.py ctdet --demo ../images/TrpV1_512 --load_model ../exp/ctdet/TrpV1_dla/model_best.pth --K 1000 --vis_thresh 0.3

python demo.py ctdet --demo ../images/TrpV1_1024 --load_model ../exp/ctdet/TrpV1_1024/model_best.pth --K 500 --vis_thresh 0.3

python demo.py ctdet --demo ../images/proteasome_demo --load_model ../exp/ctdet/coco_dla/model_last.pth --K 1500 --vis_thresh 0.3

python demo.py ctdet --demo ../images/proteasome_512 --load_model ../exp/ctdet/proteasome_512/model_best.pth --K 1500 --vis_thresh 0.3

python test.py --exp_id proteasome_1024 --not_prefetch_test ctdet --load_model ../exp/ctdet/proteasome_1024/model_best.pth --dataset proteasome --K 1500

python test.py --exp_id coco_dla --not_prefetch_test ctdet --load_model ../exp/ctdet/coco_dla/model_best.pth --K 700 --dataset proteasome_512

python test.py --exp_id proteasome_512_1 --not_prefetch_test ctdet --load_model ../exp/ctdet/proteasome_512_1/model_best.pth --dataset proteasome_512 --K 1500

python test.py --exp_id TrpV1_1024 --dataset TrpV1_1024 --not_prefetch_test ctdet --load_model ../exp/ctdet/TrpV1_1024/model_best.pth --K 200

python test.py --exp_id TrpV1_dla --not_prefetch_test ctdet --load_model ../exp/ctdet/TrpV1_dla/model_best.pth --K 500