# InfoMaxCutOut

This is a Data Augmentation method designed for Dermatology Classification. Our tech report describing it is under review at ISBI 2024.

To run our method, you must have access to the Fitzpatrick17k dataset. First build the Dockerfile:

    docker build --tag=equity:0.1 .
  
Next, edit run.sh to insert the path on your machine to this directory, to the Fitzpatrick17k image directory, and to the Fitzpatrick17k label CSV file. Then run

    bash run_experiments.sh
    
We trained each network on 4 NVIDIA Quadro RTX 5000s on a single server.

If you would like to cite us, the bibtex is as follows.

    @inproceedings{
    infomaxcutout,
    title={Classifier-Steered Background Suppression for Robust Dermatology Diagnosis},
    author={Miguel Dominguez},
    booktitle={Under review at the IEEE International Symposium of Biomedical Imaging 2024},
    year={2024}
    }

This code uses bits and pieces from the following repositories:

https://github.com/milesial/Pytorch-UNet

https://github.com/WangYueFt/dgcnn
