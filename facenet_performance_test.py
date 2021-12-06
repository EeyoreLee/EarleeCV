from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from torchvision import transforms as T
import torch
import time


device = 'cuda'

resnet = InceptionResnetV1(pretrained='vggface2', device=torch.device(device)).eval()
mtcnn = MTCNN(device=torch.device(device))

with torch.no_grad():

        # warm up
    for _ in range(5):
        start = time.time()
        # best image size is 160x160 px
        img = Image.open('face.jpeg')

        # Get cropped and prewhitened image tensor
        img_cropped = mtcnn(img, save_path='face_only.jpg')
        img_cropped = img_cropped.to(device)


        # Calculate embedding (unsqueeze to add batch dimension)
        img_embedding = resnet(img_cropped.unsqueeze(0))
    # img_embedding = img_embedding.detach().clone().cpu().numpy()
        torch.cuda.synchronize()
        end = time.time()
        # print('Time:{}ms'.format((end-start)*1000))
    fps_time_cost = []
    for _ in range(10):
        start = time.time()
        img = Image.open('face.jpeg')

        # Get cropped and prewhitened image tensor
        img_cropped = mtcnn(img, save_path='face_only.jpg')
        img_cropped = img_cropped.to(device)


        # Calculate embedding (unsqueeze to add batch dimension)
        img_embedding = resnet(img_cropped.unsqueeze(0))
        torch.cuda.synchronize()
        end = time.time()
        fps_time_cost.append(end-start)

        print('Time:{}ms'.format((end-start)*1000))
    total_time_cost = 0
    for t in fps_time_cost:
        total_time_cost += t
    print("FPS: %f"%(1.0/(total_time_cost/10)))

    # best image size is 160x160 px
    # img = Image.open('face.jpeg')

    # # Get cropped and prewhitened image tensor
    # img_cropped = mtcnn(img, save_path='face_only.jpg')
    # img_cropped = img_cropped.to(device)


    # # Calculate embedding (unsqueeze to add batch dimension)
    # img_embedding = resnet(img_cropped.unsqueeze(0))
    # img_embedding = img_embedding.detach().clone().cpu().numpy()
pass