#!/usr/bin/env python
# coding: utf-8

# In[1]:
from IPython import get_ipython

# clone YOLOv5 repository
get_ipython().system('git clone https://github.com/ultralytics/yolov5  # clone repo')
get_ipython().run_line_magic('cd', 'yolov5')
get_ipython().system('git reset --hard 886f1c03d839575afecb059accf74296fad395b6')


# In[2]:


# install dependencies as necessary
get_ipython().system('pip install -qr requirements.txt  # install dependencies (ignore errors)')
import torch

from IPython.display import Image, clear_output  # to display images
#from utils.google_utils import gdrive_download  # to download models/datasets

# clear_output()
print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))


# In[3]:


#from google.colab import drive
#drive.mount('/content/drive')


# In[4]:


#follow the link below to get your download code from from Roboflow
get_ipython().system('pip install -U roboflow')
from roboflow import Roboflow
rf = Roboflow(model_format="yolov5", notebook="roboflow-yolov5")


# In[5]:


get_ipython().run_line_magic('cd', '/content/yolov5')
get_ipython().system('curl -L "https://universe.roboflow.com/ds/OzC5DYUPMt?key=CXXfMFYYMq" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip')
#after following the link above, recieve python code with these fields filled in
#from roboflow import Roboflow
#rf = Roboflow(api_key="YOUR API KEY HERE")
#project = rf.workspace().project("YOUR PROJECT")
#dataset = project.version("YOUR VERSION").download("yolov5")


# In[6]:


# this is the YAML file Roboflow wrote for us that we're loading into this notebook with our data
get_ipython().run_line_magic('cat', 'data.yaml')


# In[7]:


# define number of classes based on YAML
import yaml
with open("data.yaml", 'r') as stream:
    num_classes = str(yaml.safe_load(stream)['nc'])


# In[8]:


#this is the model configuration we will use for our tutorial 
get_ipython().run_line_magic('cat', '/content/yolov5/models/yolov5s.yaml')


# In[9]:


#customize iPython writefile so we can write variables
from IPython.core.magic import register_line_cell_magic

@register_line_cell_magic
def writetemplate(line, cell):
    with open(line, 'w') as f:
        f.write(cell.format(**globals()))


# In[10]:


get_ipython().run_cell_magic('writetemplate', '/content/yolov5/models/custom_yolov5s.yaml', "\n# parameters\nnc: {num_classes}  # number of classes\ndepth_multiple: 0.33  # model depth multiple\nwidth_multiple: 0.50  # layer channel multiple\n\n# anchors\nanchors:\n  - [10,13, 16,30, 33,23]  # P3/8\n  - [30,61, 62,45, 59,119]  # P4/16\n  - [116,90, 156,198, 373,326]  # P5/32\n\n# YOLOv5 backbone\nbackbone:\n  # [from, number, module, args]\n  [[-1, 1, Focus, [64, 3]],  # 0-P1/2\n   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4\n   [-1, 3, BottleneckCSP, [128]],\n   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8\n   [-1, 9, BottleneckCSP, [256]],\n   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16\n   [-1, 9, BottleneckCSP, [512]],\n   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32\n   [-1, 1, SPP, [1024, [5, 9, 13]]],\n   [-1, 3, BottleneckCSP, [1024, False]],  # 9\n  ]\n\n# YOLOv5 head\nhead:\n  [[-1, 1, Conv, [512, 1, 1]],\n   [-1, 1, nn.Upsample, [None, 2, 'nearest']],\n   [[-1, 6], 1, Concat, [1]],  # cat backbone P4\n   [-1, 3, BottleneckCSP, [512, False]],  # 13\n\n   [-1, 1, Conv, [256, 1, 1]],\n   [-1, 1, nn.Upsample, [None, 2, 'nearest']],\n   [[-1, 4], 1, Concat, [1]],  # cat backbone P3\n   [-1, 3, BottleneckCSP, [256, False]],  # 17 (P3/8-small)\n\n   [-1, 1, Conv, [256, 3, 2]],\n   [[-1, 14], 1, Concat, [1]],  # cat head P4\n   [-1, 3, BottleneckCSP, [512, False]],  # 20 (P4/16-medium)\n\n   [-1, 1, Conv, [512, 3, 2]],\n   [[-1, 10], 1, Concat, [1]],  # cat head P5\n   [-1, 3, BottleneckCSP, [1024, False]],  # 23 (P5/32-large)\n\n   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)\n  ]")


# In[13]:


# train yolov5s on custom data for 100 epochs
# time its performance

get_ipython().system("python train.py --img 416 --batch 16 --epochs 100 --data '/content/yolov5/data.yaml' --cfg '/content/yolov5/models/custom_yolov5s.yaml' --weights '' --name yolov5s_results  --cache")


# In[14]:


# Start tensorboard
# Launch after you have started training
# logs save in the folder "runs"
get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir runs')


# In[15]:


# we can also output some older school graphs if the tensor board isn't working for whatever reason...  # plot results.txt as results.png
Image(filename='/content/yolov5/runs/train/yolov5s_results2/results.png', width=1000)  # view results.png


# In[16]:


# first, display our ground truth data
print("GROUND TRUTH TRAINING DATA:")
Image(filename='/content/yolov5/runs/train/yolov5s_results2/test_batch0_labels.jpg', width=900)


# In[17]:


# print out an augmented training example
print("GROUND TRUTH AUGMENTED TRAINING DATA:")
Image(filename='/content/yolov5/runs/train/yolov5s_results2/train_batch0.jpg', width=900)


# #Run Inference  With Trained Weights
# Run inference with a pretrained checkpoint on contents of `test/images` folder downloaded from Roboflow.

# In[ ]:


# trained weights are saved by default in our weights folder
get_ipython().run_line_magic('ls', 'runs/')


# In[ ]:


get_ipython().run_line_magic('ls', 'runs/train/yolov5s_results2/weights')


# In[ ]:


# when we ran this, we saw .007 second inference time. That is 140 FPS on a TESLA P100!
# use the best weights!
get_ipython().run_line_magic('cd', '/content/yolov5/')
get_ipython().system('python detect.py --weights runs/train/yolov5s_results7/weights/best.pt --img 416 --conf 0.4 --source /content/yolov5/try.mp4')


# In[ ]:


get_ipython().system('python detect.py  --weights runs/train/yolov5s_results7/weights/best.pt --img 416 --conf 0.4 --source 0')


# In[ ]:


#display inference on ALL test images
#this looks much better with longer training above

import glob
from IPython.display import Image, display

for imageName in glob.glob('/content/yolov5/runs/detect/exp6/0_41_jpg.rf.c20e4f5e95c8e725f203a87c13ec8460.jpg'): #assuming JPG
    display(Image(filename=imageName))
    print("\n")


# In[ ]:


#from google.colab import drive
#drive.mount('/content/gdrive')


# In[ ]:


get_ipython().run_line_magic('cp', '/content/yolov5/runs/train/yolov5s_results2/weights/best.pt /content/gdrive/My\\ Drive')


# ## Congrats!
# 
# Hope you enjoyed this!
# 
# --Team [Roboflow](https://roboflow.ai)
