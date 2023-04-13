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

Please cite our paper if you use the code ✔
```
@inproceedings{peng2023PASNet,
  title={PASNet: Polynomial Architecture Search Framework for Two-party Computation-based Secure Neural Network Deployment},
  author={Hongwu Peng, Shanglin Zhou, Yukui Luo, Nuo Xu, Shijin Duan, Ran Ran, Jiahui Zhao, Chenghong Wang, Tong Geng, Wujie Wen, Xiaolin Xu and Caiwen Ding},
  booktitle={Proceedings of the 60th ACM/IEEE Design Automation Conference},
  year={2023}
}
```
