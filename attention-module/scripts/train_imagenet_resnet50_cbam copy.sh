python ../train_imagenet.py \
			--ngpu 2 \
			--workers 20 \
			--arch resnet --depth 50 \
			--epochs 80 \
			--batch-size 256 --lr 0.1 \
			--att-type CBAM \
			--prefix RESNET50_IMAGENET_CBAM \
			./data/ImageNet/
