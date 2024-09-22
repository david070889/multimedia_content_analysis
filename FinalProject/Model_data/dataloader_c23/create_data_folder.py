import argparse
import csv
import os
import random
import shutil

def retrieve_images(compression="c23"):
    fake_types = [
        "Deepfakes", "FaceSwap",
        "NeuralTextures", "Face2Face", "FaceShifter"
    ]
    real_types = ["youtube"]
    fake_path = lambda t, c: f"manipulated_sequences/{t}/{c}/images/"
    real_path = lambda t, c: f"original_sequences/{t}/{c}/images/"
    cases = [(fake_types, fake_path), (real_types, real_path)]
    
    folder_abbr = {
        "Deepfakes": "DF",
        "FaceSwap": "FS",
        "NeuralTextures": "NT",
        "Face2Face": "F2F",
        "FaceShifter": "FSH",
        "youtube": "YT"
    }
    
    # Generate video indices
    video_indices = list(range(1000))
    random.shuffle(video_indices)
    
    train_videos = set(video_indices[:720])
    val_videos = set(video_indices[720:860])
    test_videos = set(video_indices[860:])
    
    def get_video_number(image_name):
        # Extract video number from the image name
        # Assuming the image name contains the video number as a prefix
        return int(image_name.split('_')[0])
    
    real_images = []
    fake_images = []
    
    # Retrieve the paths and labels
    for label, case in enumerate(cases):
        video_types = case[0]
        path_f = case[1]
        for t in video_types:
            path = path_f(t, compression)
            images = os.listdir(path)
            paths = [path]*len(images)
            labels = [label]*len(images)
            abbr = folder_abbr[t]
            concat = [(path, image, label) for image in images]
            if label == 0:
                fake_images += concat
            else:
                real_images += concat
    
    random.shuffle(real_images)
    random.shuffle(fake_images)
    
    new_path = f"dataloader_{compression}/"
    new_tr_path_real = new_path + "train/real/"
    new_tr_path_fake = new_path + "train/fake/"
    new_va_path_real = new_path + "validation/real/"
    new_va_path_fake = new_path + "validation/fake/"
    new_te_path_real = new_path + "test/real/"
    new_te_path_fake = new_path + "test/fake/"
    
    os.makedirs(new_tr_path_real, exist_ok=True)
    os.makedirs(new_tr_path_fake, exist_ok=True)
    os.makedirs(new_va_path_real, exist_ok=True)
    os.makedirs(new_va_path_fake, exist_ok=True)
    os.makedirs(new_te_path_real, exist_ok=True)
    os.makedirs(new_te_path_fake, exist_ok=True)
    
    def move_images(image_list, video_set, target_path_real, target_path_fake, folder_abbr):
        for image in image_list:
            video_number = get_video_number(image[1])
            if video_number in video_set:
                src = image[0] + image[1]
                abbr = folder_abbr[image[0].split('/')[1]]  # 根據資料夾名稱獲取縮寫
                new_image_name = image[1].replace(".jpg", f"_{abbr}.jpg")
                if image[2] == 0:  # fake images
                    tar = target_path_fake + new_image_name
                else:  # real images
                    tar = target_path_real + new_image_name
                shutil.copyfile(src, tar)
    
    # Move images to their respective folders
    move_images(real_images, train_videos, new_tr_path_real, new_tr_path_fake, folder_abbr)
    move_images(fake_images, train_videos, new_tr_path_real, new_tr_path_fake, folder_abbr)
    move_images(real_images, val_videos, new_va_path_real, new_va_path_fake, folder_abbr)
    move_images(fake_images, val_videos, new_va_path_real, new_va_path_fake, folder_abbr)
    move_images(real_images, test_videos, new_te_path_real, new_te_path_fake, folder_abbr)
    move_images(fake_images, test_videos, new_te_path_real, new_te_path_fake, folder_abbr)
    
    # Create CSV files
    def create_csv(image_list, video_set, filename):
        csv_list = [image_metadata[1:] for image_metadata in image_list if get_video_number(image_metadata[1]) in video_set]
        with open(filename, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "label"])
            for row in csv_list:
                writer.writerow(row)
    
    create_csv(real_images + fake_images, train_videos, "train_metadata.csv")
    create_csv(real_images + fake_images, val_videos, "validation_metadata.csv")
    create_csv(real_images + fake_images, test_videos, "test_metadata.csv")

if __name__ == "__main__":
    # Parse the command line input
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--compression", "-c", type=str)
    args = p.parse_args()
    # Check if the compression input is valid
    if args.compression is None:
        print("Please enter a compression argument.")
    elif args.compression not in ["raw", "c23", "c40"]:
        print(f"Compression argument '{args.compression}', not valid")
    else:
        retrieve_images(args.compression)
