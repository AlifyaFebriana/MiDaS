import numpy as np

def compute_camera_intrinsic(focal_length, sensor_size, original_image_size, cropped_image_size):
    fx = (focal_length / sensor_size[0] * original_image_size[0])
    fy = (focal_length / sensor_size[1] * original_image_size[1])

    original_cx = original_image_size[0] / 2
    original_cy = original_image_size[1] / 2

    crop_offset_x = (original_image_size[0] - cropped_image_size[0]) / 2
    crop_offset_y = (original_image_size[1] - cropped_image_size[1]) / 2

    cropped_cx = original_cx - crop_offset_x
    cropped_cy = original_cy - crop_offset_y

    #return fx, cropped_cx, fy, cropped_cy

    K = np.array([[fx, 0, cropped_cx, 0],
                [0, fy, cropped_cy, 0],
                [0, 0, 1, 0]])


    return K.flatten()
    #return K


focal_length = 74.5
sensor_size = (22.3, 14.9)
original_image_size = (5184, 3456)
cropped_image_size = (256, 256)
K = compute_camera_intrinsic(focal_length, sensor_size, original_image_size, cropped_image_size)
print(K)
'''
fx, cx, fy, cy = compute_camera_intrinsic(focal_length, sensor_size, original_image_size, cropped_image_size)
print("This is the Camera Intrinsic Parameters:\n")
print("\nfx (focal length) in the X directions: \n", fx)
print("\nfy (focal length) in the Y direction: \n", fy)
print("\ns (axis skew): 0\n")
print("\ncx (optical center): \n", cx)
print("\ncy (optical center): \n", cy)
'''
# save the camera intrinsic to a txt file in the kitti format
# kitti format:
# [fu  0  cx  -fu*bx]  
# [0  fv  cy -fv*by ]  
# [0   0   1     0  ]

#with open("calib.txt", "w") as file:
   # file.write("{} 0 {} 0 0 {} {} 0 0 0 1 0\n". format(fx, cx, fy, cy))

with open('calib.txt', 'w') as f:
    f.write(' '.join(['%.12e' % x for x in K]))
