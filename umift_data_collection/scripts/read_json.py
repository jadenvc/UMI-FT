import json
import pickle
import os
import glob

folder_path = "/shared_local/data/raw/umift/test/WBW-iph-b0/processed_data/iphone/2025-04-18/"
# loop through folders under the path
for folder in os.listdir(folder_path):
    full_folder_path = os.path.join(folder_path, folder)
    if os.path.isdir(full_folder_path):
        print("Folder:", folder)

        json_path = os.path.join(full_folder_path, "left.json")

        # read the json file, print the keys and values
        with open(json_path, 'r') as f:
            data = json.load(f)
            # print("Keys:", data.keys())
            print("poseTimes:", len(data["poseTimes"]))
            print("rgbTimes:", len(data["rgbTimes"]))

        mp4_path = os.path.join(full_folder_path, "left_rgb.mp4")


        import cv2

        

        def count_frames(video_path):
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print("Error: Cannot open video file.")
                return

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Frame size: {width} x {height}")
            
            
            
            

            cap.release()

            print(f"Total number of frames: {frame_count}")

        # Example usage
        count_frames(mp4_path)








