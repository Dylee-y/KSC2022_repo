python ../train_imagenet.py \
			--ngpu 2 \
			--workers 8 \
			--arch resnet --depth 50 \
			--epochs 60 \
			--batch-size 64 --lr 0.1 \
			--att-type CBAM \
			--net-type ImageNet \
			--prefix RESNET50_CUB_CBAM \
			/home/percv-d0/dyne_ksc2022/datasets/CUB_200_2011
