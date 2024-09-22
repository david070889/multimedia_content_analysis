import os

def parse_video_list(file_path):
    video_entries = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 2:
                label = int(parts[0])
                video_path = parts[1]
                video_entries.append((label, video_path))
            else:
                print(f"Invalid line format: {line}")
    return video_entries



txt = 'List_of_testing_videos.txt'
video_entries = parse_video_list(txt)
for label, video_path in video_entries:
    print(f"Label: {label}, Video Path: {video_path}")
    print(len(video_entries))