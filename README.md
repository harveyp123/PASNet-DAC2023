# PASNet_DAC

Evaluation code on PASNet-A (ResNet18 as backbone), PASNet-B and PASNet-C (ResNet50 as backbone), PASNet-D (MobileNetV2 as backbone) on imagenet.

- Install requirements
    ```bash
    pip install -r requirements.txt
    ```

- Run evaluation
    ```bash
    bash run_eval.sh
    ```

Please find the checkpoints [here](https://drive.google.com/drive/folders/1gk7lL6tkG2rr8cAKHbwaOGiHhfDTdFrt?usp=share_link). Full PASNet framework training/secure inference pipeline, and pretrained models (PASNet-A to PASNet-D) will be released upon paper reception. 

Full code will be released soon...

Please cite our paper if you use the code ✔
```
@inproceedings{peng2023pasnet,
  title={PASNet: Polynomial Architecture Search Framework for Two-party Computation-based Secure Neural Network Deployment},
  author={Peng, Hongwu and Zhou, Shanglin and Luo, Yukui and others},
  booktitle={2023 60th ACM/IEEE Design Automation Conference (DAC)},
  pages={1--6},
  year={2023},
  organization={IEEE}
}
```
