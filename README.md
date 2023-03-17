## Image Captioning Kazakh model (based on [ExpansioNet v2](https://github.com/jchenghu/expansionnet_v2))

#### Requirements
* python >= 3.7
* numpy
* Java 1.8.0
* pytorch 1.9.0
* h5py
* playsound
* scipy

### Model checkpoint

The checkpoint for the model is stored in [drive](https://drive.google.com/drive/folders/16PDZvoNs3P-O9Vr3zEb6bb-aaSDOiSY0?usp=sharing). Please, place the file into the `checkpoints` directory.

### Inference acceleration with NVIDIA's TensorRT deep learning library
* Convert Pytorch model to onnx using this [script](https://github.com/jchenghu/ExpansionNet_v2/blob/master/onnx4tensorrt/convert2onnx.py).
* Convert onnx to TensorRT format. The onnx model file can be converted to a TensorRT egnine using the trtexec tool.
```
trtexec --onnx=./model.onnx --saveEngine=./model_fp32.engine --workspace=200

```
* Inference using TensorRT engine
```
python3 infer_trt.py
```


### Benchmark

| № image       | Pytorch model(model size:2.7GB) |  TensorRT(FP32, model size: 986MB) |
| ------------- | -------------------------------- |-----------------------------------|
| 1             | 2.56                             |  0.53
| 2             | 1.14                             |  0.48
| 3             | 1.16                             |  0.47
| 4             | 1.12                             |  0.49
| 5             | 1.17                             |  0.46
| 6             | 1.21                             |  0.48
| 7             | 1.35                             |  0.5
| 8             | 1.5                              |  0.5
| 9             | 1.12                             |  0.46
| 10            | 1.1                              |  0.5

### Acknowledgements
The implementation of the model relies on https://github.com/jchenghu/expansionnet_v2. We thank the original authors for their open-sourcing.

### Preprint on TechRxiv
[Image Captioning for the Visually Impaired and Blind: A Recipe for Low-Resource Languages](https://www.techrxiv.org/articles/preprint/Image_Captioning_for_the_Visually_Impaired_and_Blind_A_Recipe_for_Low-Resource_Languages/22133894)

### BibTex
```
@article{Arystanbekov2023,
author = "Batyr Arystanbekov and Askat Kuzdeuov and Shakhizat Nurgaliyev and Hüseyin Atakan Varol",
title = "{Image Captioning for the Visually Impaired and Blind: A Recipe for Low-Resource Languages}",
year = "2023",
month = "2",
url = "https://www.techrxiv.org/articles/preprint/Image_Captioning_for_the_Visually_Impaired_and_Blind_A_Recipe_for_Low-Resource_Languages/22133894",
doi = "10.36227/techrxiv.22133894.v1"
}
```

