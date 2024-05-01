import numpy as np
from PIL import Image
import random
import copy
from scipy.ndimage import zoom
import os


def create_tensor(w, h):
    new_tensor = np.ones((4, h, w))
    return new_tensor

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


def load_images_from_directory(directory):
    image_objects = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            # Load the image using PIL
            image_path = os.path.join(directory, filename)
            image = Image.open(image_path)
            image_objects.append(img_to_tensor(image))
    return image_objects

# Example usage:
image_objects_array = load_images_from_directory("ingredients/")
target_objects_array = load_images_from_directory("targets/")


def make_gene():
    return np.random.random(10)

gene = [0.4, # image index
        0, # x origin 0
        0, # y origin 0 
        1, # x origin 1
        1, # y origin 1
        0, # x destination 0
        0, # y destination 0
        0.8, # x destination 1
        0.8, # y destination 1
        0]   # z index

def apply(gene, canvas):
    # image = gene[0]
    ingredient = image_objects_array[int(gene[0]*len(image_objects_array))]

    # print("ingredient shape:", ingredient.shape)
    # print("canvas shape:", canvas.shape)
    # select image
    x_ori_0 = min(gene[1], gene[3])
    y_ori_0 = min(gene[2], gene[4])
    x_ori_1 = max(gene[1], gene[3])
    y_ori_1 = max(gene[2], gene[4])
    x_dest_0 = min(gene[5], gene[7])
    y_dest_0 = min(gene[6], gene[8])
    x_dest_1 = max(gene[5], gene[7])
    y_dest_1 = max(gene[6], gene[8])

    canvas_width = canvas.shape[2]
    canvas_height = canvas.shape[1]
    ingredient_width = ingredient.shape[2]
    ingredient_height = ingredient.shape[1]

    origin_x0 = int(ingredient_width * x_ori_0)
    origin_y0 = int(ingredient_height * y_ori_0)
    origin_x1 = int(ingredient_width * x_ori_1)
    origin_y1 = int(ingredient_height * y_ori_1)

    splice = ingredient[:,origin_y0:origin_y1, origin_x0:origin_x1]
    

    dest_x0 = int(canvas_width * x_dest_0)
    dest_y0 = int(canvas_height * y_dest_0)
    dest_x1 = int(canvas_width * x_dest_1)
    dest_y1 = int(canvas_height * y_dest_1)
    try:
        # resize the splice
        splice_width = (origin_x1-origin_x0)
        splice_height = (origin_y1-origin_y0)
        
        destination_width = (dest_x1-dest_x0)
        destination_height = (dest_y1-dest_y0)

        width_factor =  destination_width / splice_width
        height_factor =  destination_height / splice_height

        splice = zoom(splice, (1, height_factor, width_factor), order=3)

        if width_factor == 0 or height_factor == 0:
            # print("slice too thin")
            pass
        else:
            # try:
            canvas[:,dest_y0:dest_y1, dest_x0:dest_x1] = splice
            # except:
                # print("fuck this, moveon")
    except ZeroDivisionError:
        # print("slice too thin and you divided by 0")
        pass

def distance(canvas, target):
    return np.sum(np.abs(canvas - target)) ##/ (canvas.shape[1] * canvas.shape[2] * 3)
    # return np.sum(np.power(np.abs(canvas - target),2)) ##/ (canvas.shape[1] * canvas.shape[2] * 3)

best = np.ones_like(target_objects_array[0])
attempt = np.ones_like(target_objects_array[0])

genes = [make_gene() for x in range(10)]
for gene in genes:
    apply(gene, attempt)
    apply(gene, best)

last_distance = distance(attempt[0:3,:,:], target_objects_array[0][0:3,:,:])
print("initial distance:", last_distance)

for ctr in range(10000):
    # retain old genes
    old_genome = copy.deepcopy(genes)

    if random.random() < 0.9:
        for _ in range(len(old_genome)//10):
            rand_gene_i = np.random.randint(0, len(genes))
            rand_chroma_i = np.random.randint(0, len(genes[rand_gene_i]))
            value = genes[rand_gene_i][rand_chroma_i]
            genes[rand_gene_i][rand_chroma_i] = np.random.rand()
    else:
        # if random.random() < 0.66:
        genes.append(make_gene())
        # else:
            # random_index = random.randint(0, len(genes) - 1)
            # removed_gene = genes.pop(random_index)
    attempt = np.ones_like(target_objects_array[0])

    genes = sorted(genes, key=lambda elem: elem[9])
    for gene in genes:
        apply(gene, attempt)

    new_distance = distance(attempt[0:3,:,:], target_objects_array[0][0:3,:,:])
    if new_distance <= last_distance:
        print("ctr", ctr, str(len(genes)), "genes,", "dist", min(new_distance,last_distance), " changed", new_distance - last_distance)
        if new_distance < last_distance:
            tensor_to_img(attempt).save("progress"+"{:05d}".format(ctr)+".png")
        last_distance = new_distance
    else:
        print("ctr", ctr, str(len(genes)), "genes,", "dist", min(new_distance,last_distance), " worse", new_distance - last_distance)
        genes = old_genome


tensor_to_img(attempt).save("final.png")
# eye = img_to_tensor(Image.open("eye.png"))
# print(eye)
# print(eye.shape)
# backeye = tensor_to_img(eye)
# backeye.save("eye2.png")
# print(backeye)

# re_eye = img_to_tensor(Image.open("eye2.png"))
# print("eye2",re_eye.shape)
# tensor_to_img(re_eye).save("eye3.png")