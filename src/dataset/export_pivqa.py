import json
import os
from tqdm import tqdm
import numpy as np

from constants import CLEVRER_PATH, PIVQA_DATASET_DIR

source_directory = CLEVRER_PATH
destination_directory = PIVQA_DATASET_DIR

def export_annotation():
    for root, dirs, files in os.walk(source_directory):
        dest_root = os.path.join(destination_directory, os.path.relpath(root, source_directory))

        if not "annotations" in dest_root:
            continue

        os.makedirs(dest_root, exist_ok=True)

        print(f"loading from {root} to {dest_root}...")

        for file in tqdm(files):
            if file.endswith('.json'):
                source_file = os.path.join(root, file)
                dest_file = os.path.join(dest_root, file)

                with open(source_file, 'r') as f_source:
                    js = json.load(f_source)
                
                # js["hello"] = "world"

                with open(dest_file, 'w') as f_dest:
                    json.dump(js, f_dest)

def load_annotation()->list:
    """
    returns train, valid set of annotations
    annotations contains an annotation of each frame.

    each annotation is a dictionary with the following keys:
        "scene_index": scene_index (e.g. 00300)
        "video_filename": video_filename (e.g. video_00300.mp4)
        "locations": (objects * frames * 3) np.array
        "velocities": (objects * frames * 3) np.array
    """
    train_annotations = []
    valid_annotaitons = []

    for root, dirs, files in os.walk(source_directory):
        dest_root = os.path.join(destination_directory, os.path.relpath(root, source_directory))

        if not "annotations" in dest_root:
            continue

        os.makedirs(dest_root, exist_ok=True)

        print(f"loading from {root} to {dest_root}...")

        # read each annotation per a scene
        for file in tqdm(files):
            if file.endswith('.json'):

                source_file = os.path.join(root, file)
                dest_file = os.path.join(dest_root, file)

                with open(source_file, 'r') as f_source:
                    annotation_js = json.load(f_source)
                
                # load metadata
                scene_index = annotation_js["scene_index"]
                video_filename = annotation_js["video_filename"]

                num_objects = len(annotation_js["object_property"])
                num_frames = len(annotation_js["motion_trajectory"])

                # load object properties
                locations = np.zeros((num_objects, num_frames, 3))
                velocities = np.zeros((num_objects, num_frames, 3))

                for i in range(num_objects):
                    for j in range(num_frames):
                        locations[i, j] = annotation_js["motion_trajectory"][j]["objects"][i]["location"]
                        velocities[i, j] = annotation_js["motion_trajectory"][j]["objects"][i]["velocity"]
                
                annotation = {
                    "scene_index": scene_index,
                    "video_filename": video_filename,
                    "locations": locations,
                    "velocities": velocities
                
                }

                if "train" in root:
                    train_annotations.append(annotation)
                elif "valid" in root:
                    valid_annotaitons.append(annotation)

    return train_annotations, valid_annotaitons

def main():
    train_annotations, valid_annotaitons = load_annotation()
    print(train_annotations[0])

if __name__ == "__main__":
    main()