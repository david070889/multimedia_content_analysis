import os
def get_video_number(image_name):
    # Extract video number from the image name
    # Assuming the image name contains the video number as a prefix
    return int(image_name.split('_')[0])

path1 = 'dataloader_c23/train/real'
path2 = 'dataloader_c23/train/fake'
train_real_images = os.listdir(path1)
train_fake_images = os.listdir(path2)
a, b = [], []
for i in train_real_images:
    x = get_video_number(i)
    if x not in a:
        a.append(x)
for i in train_fake_images:
    x = get_video_number(i)
    if x not in b:
        b.append(x)

def equal(a, b):
    for i in range(len(a)):
        if a[i] != b[i]:
            return print("not equal")
    return print('equal')

equal(a, b)
print(a[:10])
print(b[:10])
