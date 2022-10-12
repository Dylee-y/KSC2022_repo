python ../train_imagenet.py \
			--ngpu 1 \
			--workers 8 \
			--arch resnet --depth 50 \
			--epochs 80 \
			--batch-size 256 --lr 0.1 \
			--att-type CBAM \
			--net-type Tiny-ImageNet \
			--prefix RESNET18_TINYIMAGENET_CBAM \
			/home/percv-d0/dyne_ksc2022/datasets/tiny-imagenet-200
