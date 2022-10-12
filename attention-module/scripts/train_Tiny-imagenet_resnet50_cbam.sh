python ../train_imagenet.py \
			--ngpu 2 \
			--workers 8 \
			--arch resnet --depth 50 \
			--epochs 40 \
			--batch-size 64 --lr 0.1 \
			--att-type CBAM \
			--net-type Tiny-ImageNet \
			--prefix RESNET50_TINYIMAGENET_CBAM \
			/home/percv-d0/dyne_ksc2022/datasets/tiny-imagenet-200
