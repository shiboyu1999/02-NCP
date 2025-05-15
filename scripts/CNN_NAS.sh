CUDA_VISIBLE_DEVICES=1 python /home/biye/code/02-NCP-1/LG-NAS/CNN-NAS.py \
        --batch_size 32 \
        --learning_rate 0.0005 \
        --epochs 20 \
        --log_path ./results/CNN-NAS/resnet-34/logs/ \
        --dataset_root /home/biye/dataset/ILSVRC2012/ \
        --arch resnet34
