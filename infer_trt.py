import tensorrt as trt
import numpy as np
from collections import OrderedDict,namedtuple
import pickle
import torchvision
from PIL import Image as PIL_Image
from utils.language_utils import tokens2description
import time
import torch
import pycuda.autoinit
import pycuda.driver as cuda
img_size = 384

with open('./demo_material/demo_coco_tokens.pickle', 'rb') as f:
    coco_tokens = pickle.load(f)
    sos_idx = coco_tokens['word2idx_dict'][coco_tokens['sos_str']]
    eos_idx = coco_tokens['word2idx_dict'][coco_tokens['eos_str']]

class TRT_engine():
    def __init__(self, weight) -> None:
        self.imgsz = [318,318]
        #self.weight = weight
        self.device = torch.device('cuda:0')

        # Infer TensorRT Engine
        logger = trt.Logger(trt.Logger.WARNING)
        logger.min_severity = trt.Logger.Severity.ERROR
        runtime = trt.Runtime(logger)
        trt.init_libnvinfer_plugins(logger,'') # initialize TensorRT plugins
        with open(weight, "rb") as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.imgsz = engine.get_binding_shape(0)[2:]  # get the read shape of model, in case user input it wrong
        self.context = engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_tensor_shape(binding))
            dtype = trt.nptype(engine.get_tensor_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

    def preprocess_image(self, image_path=None, img=None):
        if img is not None:
            pil_image = PIL_Image.fromarray(img)
        else:
            pil_image = PIL_Image.open(image_path)
        transf_1 = torchvision.transforms.Compose([torchvision.transforms.Resize((img_size, img_size))])
        transf_2 = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                    std=[0.229, 0.224, 0.225])])
        if pil_image.mode != 'RGB':
            pil_image = PIL_Image.new("RGB", pil_image.size)
        preprocess_pil_image = transf_1(pil_image)
        image = torchvision.transforms.ToTensor()(preprocess_pil_image)
        image = transf_2(image)
        return image.unsqueeze(0)
    def predict(self,img_path):
        img = self.preprocess_image(img_path)
        self.inputs[0]['host'] = np.ravel(img).astype(np.float32)
        self.inputs[1]['host'] = np.array([0]).astype(np.int32)
        self.inputs[2]['host'] = np.array([sos_idx]).astype(np.int32)
        # transfer data to the gpu
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        # run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        # fetch outputs from gpu
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        # synchronize stream
        self.stream.synchronize()
        #data = [out['host'] for out in self.outputs]
        output_caption = tokens2description(self.outputs[0]['host'].tolist(), coco_tokens['idx2word_list'], sos_idx, eos_idx)
        return output_caption


if __name__ == "__main__":
    trt_engine = TRT_engine("./trt_fp32.engine")
    img_path = './demo_material/napoleon.jpg'
    result = trt_engine.predict(img_path)
    print(result)