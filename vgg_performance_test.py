# -*- encoding: utf-8 -*-
'''
@create_time: 2021/12/03 10:12:21
@author: lichunyu
'''
import time

import torchvision.models as models
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor



device = 'cuda'

with torch.no_grad():
    vgg16 = models.vgg16(pretrained=True)
    vgg16.to(device)
    vgg16.eval()
    # train_nodes, eval_nodes = get_graph_node_names(vgg16)
    return_nodes = {
        'features.30': 'features30',
        'avgpool': 'avgpool',
        'flatten': 'flatten'
    }
    model = create_feature_extractor(vgg16, return_nodes=return_nodes)
    model.to(device)
    model.eval()

    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    img = Image.open('img.jpg')
    img = transform(img).unsqueeze(0).to(device)
    # warm up
    for _ in range(5):
        start = time.time()
        outputs = model(img)
        torch.cuda.synchronize()
        end = time.time()
        # print('Time:{}ms'.format((end-start)*1000))
    fps_time_cost = []
    for _ in range(10):
        start = time.time()
        transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        img = Image.open('img.jpg')
        img = transform(img).unsqueeze(0).to(device)
        # start = time.time()
        outputs = model(img)
        torch.cuda.synchronize()
        end = time.time()
        fps_time_cost.append(end-start)

        print('Time:{}ms'.format((end-start)*1000))
    total_time_cost = 0
    for t in fps_time_cost:
        total_time_cost += t
    print("FPS: %f"%(1.0/(total_time_cost/10)))

    with torch.profiler.profile(
        activities=[
            # torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        # use_cuda=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs'),
        profile_memory=True,
        record_shapes=True,
        with_stack=True
    ) as prof:
    # with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False) as prof:
        outputs = model(img)
        prof.step()
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage"))


# pip install torch_tb_profiler
# tensorboard --logdir ./logs --port 6006

pass