import numpy as np
from scipy.ndimage import zoom
from PIL import Image
import random


# convert image to (3, h, w) tensor
def img_to_tensor(img):
    img_tensor = np.moveaxis(np.array(img)/255, [0, 1, 2], [1, 2, 0])
    print("shape", img_tensor.shape)
    sizes = (img_tensor.shape[1], img_tensor.shape[2])

    # print(img_tensor)
    if img_tensor.shape[0] == 3:
        print("no transparency")
        alpha_array = np.ones((1,img_tensor.shape[1], img_tensor.shape[2]))
        # alpha_array = np.array(np.random.rand(1,img_tensor.shape[1], img_tensor.shape[2]))
        # print("alpha_array", alpha_array)
        # print("alpha_array.shape", alpha_array.shape)
        img_tensor = np.vstack((img_tensor, alpha_array))
        # print(img_tensor.shape)
        # img_tensor[img_tensor.shape[0]] = np.ones(sizes)
    return img_tensor

def tensor_to_img(tensor):
    # print("raw:")
    # print(tensor)
    # print("first three")
    reconst = (tensor[0:3] * tensor[3] * 255).astype(np.uint8)
    h, w = np.shape(reconst)[1:]
    # print("reconst shape:", reconst.shape)
    # stack, h, w  ---> w, h, 3
    # it just fucking works I don't know why
    reconst = np.moveaxis(reconst, [0, 1, 2], [2, 0, 1])
    # reconst = reconst.reshape((w, h, 3))
    # print("reconst shape:", reconst.shape)
    # print("{} by {}".format(w, h))
    # print(reconst)
    return Image.fromarray(reconst)


for i in range(1,20):

  # Assuming your 3D array is called 'data' with shape (height, width, channels)
  data = img_to_tensor(Image.open("eye.png"))
  print(data)
  # Define the scale factors for resizing along each dimension
  scale_factors = (1,random.randrange(1,6)/2,random.randrange(1,6)/2)  # Example: reduce height and width by half, keep channels unchanged

  # Perform interpolation-based resizing
  resized_data = zoom(data, scale_factors, order=3) 
  print(resized_data)

  tensor_to_img(resized_data).save("resized"+str(i)+".png")