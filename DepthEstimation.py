import cv2
import torch
import os
import numpy as np

# load MiDaS model for depth estimation
model_type = "DPT_Large"

midas = torch.hub.load("intel-isl/MiDaS", model_type)

# move model to gpu if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# load transforms to resize and normalize the image
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

# set the directory containing the training image
train_dir = "/home/junaid/alifya/Bean/3DBEAN/images/"

# set the output directory
output_dir = "/home/junaid/alifya/Bean/3DBEAN/depth_hybrid/"

for image in os.listdir(train_dir):
    image_path = os.path.join(train_dir, image)

    # read the input image
    img = cv2.imread(image_path)

    # apply transforms
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # apply input transforms
    input_batch = transform(img).to(device)

    # predict and resize to original resolution
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size = img.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    #print(depth_map)

    depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    #print(depth_map)

    depth_map = (depth_map*255).astype(np.uint8)
    #print(depth_map)
    depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)

    # save the depth map as an image in the output folder
    depth_map_output = os.path.join(output_dir, os.path.splitext(image)[0] + '.png')
    cv2.imwrite(depth_map_output, depth_map)

    print(f"processed {image} success and saved depth map to {depth_map_output}")

