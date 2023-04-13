### Evaluation code for imagenet under mobilenet_v2, resnet50 and resnet18
### logs and results are saved in folder "./eval_imagenet"


## PASNet-A (ResNet 18 with all_poly)
python3 eval_imagenet.py --arch resnet18 -e ckpts/resnet18/all_poly/best.pth.tar --gpus 4

## PASNet-B (ResNet 50 with all_poly)
python3 eval_imagenet.py --arch resnet50 -e ckpts/resnet50/all_poly/best.pth.tar --gpus 4

## PASNet-C (ResNet 50 with some ReLU layers)
python3 eval_imagenet.py --arch resnet50 -e ckpts/resnet50/lambda_1e-6/best.pth.tar --gpus 4

## PASNet-D (MobileNet v2 with all_poly)
python3 eval_imagenet.py --arch mobilenet_v2 -e ckpts/mobilenetv2/all_poly/best.pth.tar --gpus 4