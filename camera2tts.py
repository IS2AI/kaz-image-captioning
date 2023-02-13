import torch
import torchvision
import argparse
import pickle
import cv2
from argparse import Namespace
import os
from PIL import Image as PIL_Image
from models.End_ExpansionNet_v2 import End_ExpansionNet_v2
from utils.language_utils import convert_vector_idx2word
from espnet2.bin.tts_inference import Text2Speech
import scipy.io.wavfile as scipy_wavfile
import playsound
import threading
import RPi.GPIO as GPIO
from time import sleep, time

# Set the GPIO mode on the NVIDIA Jetson Xavier NX 
GPIO.setmode(GPIO.BCM)

# Set the GPIO pin for the button
btn = 26

# Disable GPIO warnings
GPIO.setwarnings(False)

# Set the button pin as an input pin
GPIO.setup(btn,GPIO.IN)

# Path to vocoder model file(https://github.com/IS2AI/Kazakh_TTS)
vocoder_path="./checkpoint-400000steps.pkl"

# Path to Tacotron 2 model file(https://github.com/IS2AI/Kazakh_TTS)
model_path = "./exp/tts_train_raw_char/train.loss.ave_5best.pth"

# Path to checkpoint and dictionary files of ExpansionNet v2: Kaz model
load_path = 'checkpoints/kaz_model.pth'
dict_path = 'vocabulary/vocab_kz.pickle'

#ESPnet2-TTS Model Setup
tts = Text2Speech.from_pretrained(
	model_file=model_path,
	vocoder_file=vocoder_path,
	device = "cuda",
	threshold=0.5,
	minlenratio=0.0,
	maxlenratio=10.0,
	use_att_constraint=False,
	backward_window=1,
	forward_window=3,
	prefer_normalized_feats=True,
)

#ExpansionNet v2: Kaz model Model Setup
drop_args = Namespace(enc=0.0,
                      dec=0.0,
                      enc_input=0.0,
                      dec_input=0.0,
                      other=0.0)
model_args = Namespace(model_dim=512,
                       N_enc=3,
                       N_dec=3,
                       dropout=0.0,
                       drop_args=drop_args)


with open(dict_path, 'rb') as f:
    coco_tokens = pickle.load(f)
print("Dictionary loaded ...")


img_size = 384
model = End_ExpansionNet_v2(swin_img_size=img_size, swin_patch_size=4, swin_in_chans=3,
                                swin_embed_dim=192, swin_depths=[2, 2, 18, 2], swin_num_heads=[6, 12, 24, 48],
                                swin_window_size=12, swin_mlp_ratio=4., swin_qkv_bias=True, swin_qk_scale=None,
                                swin_drop_rate=0.0, swin_attn_drop_rate=0.0, swin_drop_path_rate=0.0,
                                swin_norm_layer=torch.nn.LayerNorm, swin_ape=False, swin_patch_norm=True,
                                swin_use_checkpoint=False,
                                final_swin_dim=1536,

                                d_model=model_args.model_dim, N_enc=model_args.N_enc,
                                N_dec=model_args.N_dec, num_heads=8, ff=2048,
                                num_exp_enc_list=[32, 64, 128, 256, 512],
                                num_exp_dec=16,
                                output_word2idx=coco_tokens['word2idx_dict'],
                                output_idx2word=coco_tokens['idx2word_list'],
                                max_seq_len=63, drop_args=model_args.drop_args,
                                rank=0)

device = torch.device('cuda')
model.to(device)

checkpoint = torch.load(load_path)
model.load_state_dict(checkpoint['model_state_dict'])
print("Model loaded ...")

transf_1 = torchvision.transforms.Compose([torchvision.transforms.Resize((img_size, img_size))])
transf_2 = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225])])
beam_search_kwargs = {'beam_size': 5,
                      'beam_max_seq_len': 63,
                      'sample_or_max': 'max',
                      'how_many_outputs': 1,
                      'sos_idx': coco_tokens['word2idx_dict'][coco_tokens['sos_str']],
                      'eos_idx': coco_tokens['word2idx_dict'][coco_tokens['eos_str']]}

# Speech synthesis
def text_to_speech(text: str, export_wav_filepath):
	wav = tts(text)["wav"]
	scipy_wavfile.write(export_wav_filepath, tts.fs, wav.view(-1).cpu().numpy())
	return export_wav_filepath

# Convert opencv image format to PIL image format
def cv2_to_pil(img): 
    return PIL_Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# Play audio
def play_audio(file):
    playsound.playsound(file)

# Generate image captions
def generate_caption(img):
    start = time()
    pil_image = cv2_to_pil(img)
    
    if pil_image.mode != 'RGB':
        pil_image = PIL_Image.new("RGB", pil_image.size)
    preprocess_pil_image = transf_1(pil_image)
    tens_image_1 = torchvision.transforms.ToTensor()(preprocess_pil_image)
    tens_image_2 = transf_2(tens_image_1)

    image = tens_image_2.unsqueeze(0).cuda()
    with torch.no_grad():
        pred, _ = model(enc_x=image,
                        enc_x_num_pads=[0],
                        mode='beam_search', **beam_search_kwargs)
    pred = convert_vector_idx2word(pred[0][0], coco_tokens['idx2word_list'])[1:-1]
    pred[-1] = pred[-1] + '.'
    pred = ' '.join(pred).capitalize()
    stop = time()
    print('\nDescription: ' + pred)
    print('Time: {:.4f}s\n'.format(stop-start))
    return pred

if __name__ == "__main__":
    try:
        # initialize the Intel Realsense D455 camera
        cap = cv2.VideoCapture(4)
        while True:
            ret, frame = cap.read()
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            if ret is None:
                break
            cv2.imshow("frame", frame)
            k = cv2.waitKey(1)
            if k%256 == 27:
            # ESC pressed
                print("Escape hit, closing...")
                break
            if(GPIO.input(btn) == 1): #check button pressed
                start = time() #start timer
                sleep(0.02)
                while(GPIO.input(btn) == 1): #always loop if button pressed
                    sleep(0.01)
                length = time() - start #get long time button pressed
                long_press_length = 5 # seconds
                if (length > long_press_length): #if button greater 5 second exit from application 
                    print("Long time button pressed")
                    sleep(0.2)
                    break
                else:
                    print("Short time button pressed")
                    result = generate_caption(frame)
                    text_to_speech(result, './output.wav')
                    print(result)
                    audio_thread = threading.Thread(target=play_audio, args = ('output.wav',))
                    audio_thread.start()
                    sleep(0.2)
                print("press duration: "+str(length))
    except KeyboardInterrupt:
        GPIO.cleanup()
    cap.release()    
    cv2.destroyAllWindows()
