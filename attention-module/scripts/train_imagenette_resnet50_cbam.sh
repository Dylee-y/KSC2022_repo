CUDA_VISIBLE_DEVICES=0,1
python ../train_imagenette.py \
			--ngpu 2 \
			--workers 8 \
			--arch resnet --depth 50 \
			--epochs 100 \
			--batch-size 8 --lr 0.1 \
			--resume ./scripts/checkpoints/ \
			--att-type CBAM \
			--net-type Tiny-ImageNet \
			--prefix RESNET50_IMGNTT_CBAM \
			/home/percv-d0/dyne_ksc2022/datasets/imagenette2
