# InfoMaxCutOut

This is a Data Augmentation method designed for Dermatology Classification to improve robustness to distributional shifts. Our tech report describing it has been accepted at AMIA Symposium 2024.

To run our method, you must have access to the Fitzpatrick17k dataset. First build the Dockerfile:

    docker build --tag=equity:0.1 .
  
Next, edit run.sh to insert the path on your machine to this directory, to the Fitzpatrick17k image directory, and to the Fitzpatrick17k label CSV file. Then run

    bash run_experiments.sh
    
We trained each network on 4 NVIDIA Quadro RTX 5000s on a single server.

If you would like to cite us, the bibtex is as follows.

    @inproceedings{
    infomaxcutout,
    title={Robust Visual Identification of Under-resourced Dermatological Diagnoses with Classifier-Steered Background Masking},
    author={Miguel Dominguez, Julie Ryan Wolf, Paritosh Prasad, Wendemagegn Enbiale, Michael Gottlieb, Carl T. Berdahl, Art Papier},
    booktitle={American Medical Informatics Association Symposium 2024},
    year={2024}
    }

This code uses bits and pieces from the following repositories:

https://github.com/milesial/Pytorch-UNet (UNet Implementation)

https://github.com/WangYueFt/dgcnn (Basic training code)

https://github.com/conceptofmind/PaLM/ (StableAdamW)
