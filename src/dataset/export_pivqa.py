import json
import os
from tqdm import tqdm
import numpy as np
from typing import Tuple, List, Literal, Union
import random

from dataset.question import (
    LocationQuestion, 
    VelocityQuestion, 
    SimpleCumulativeQuestion, 
    ElaborativeCumulativeQuestion
    )
from dataset.annotation import ObjectProperty, Collision, Annotation
from constants import (
    CLEVRER_PATH, 
    PIVQA_DATASET_DIR, 
    SEED, 
    FRAME_RATE, 
    NUM_QUESTIONS_PER_A_SCENE
    )

source_directory = CLEVRER_PATH
destination_directory = PIVQA_DATASET_DIR

def load_annotation()->Tuple[List[Annotation], List[Annotation]]:
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

        if not "annotations" in root:
            continue

        print(f"loading from {root}...")

        # read each annotation per a scene
        for file in tqdm(files):
            if file.endswith('.json'):

                source_file = os.path.join(root, file)

                with open(source_file, 'r') as f_source:
                    annotation_js = json.load(f_source)
                
                # load metadata
                scene_index = annotation_js["scene_index"]
                video_filename = annotation_js["video_filename"]

                num_objects = len(annotation_js["object_property"])
                num_frames = len(annotation_js["motion_trajectory"])

                # load object properties
                object_properties = []

                for prop in annotation_js["object_property"]:
                    object_id = prop["object_id"]
                    color = prop["color"]
                    material = prop["material"]
                    shape = prop["shape"]

                    object_properties.append(
                        ObjectProperty(
                            object_id, 
                            color, 
                            material, 
                            shape
                            )
                        )

                # load motions
                locations = np.zeros((num_objects, num_frames, 3))
                velocities = np.zeros((num_objects, num_frames, 3))

                for i in range(num_objects):
                    for j in range(num_frames):
                        locations[i, j] = annotation_js["motion_trajectory"][j]["objects"][i]["location"]
                        velocities[i, j] = annotation_js["motion_trajectory"][j]["objects"][i]["velocity"]

                assert isinstance(scene_index, int)

                # load collisions
                collisions:List[Collision] = []
                for c in annotation_js["collision"]:
                    collision = Collision(c["object_ids"], c["frame_id"], c["location"])
                    collisions.append(collision)
                
                # create single annotation
                annotation = Annotation(scene_index, video_filename, object_properties, locations, velocities, collisions)

                if "train" in root:
                    train_annotations.append(annotation)
                elif "valid" in root:
                    valid_annotaitons.append(annotation)

    train_annotations = sorted(train_annotations, key=lambda x: x.scene_index)
    valid_annotaitons = sorted(valid_annotaitons, key=lambda x: x.scene_index)

    return train_annotations, valid_annotaitons

def export_qa(
        annotations:List[Annotation], 
        dataset_type:Literal["train", "validation"],
        num_questions_per_a_scene:int=NUM_QUESTIONS_PER_A_SCENE,
        )->None:
    """
    output: qa dataset (list of {questions, scene_index, video_filename})
    [{"questions", "scene_index", "video_filename"}, {}, ...]

    questions: list of dict {
        "question_id", 
        "question", 
        "question_type", 
        "question_subtype", 
        "program", 
        "answer"
        }
    """
    random.seed(SEED)
    print(f"exporting {dataset_type} QA dataset...")

    qa_dataset = []
    for annotation in tqdm(annotations):
        questions = []

        # create {num_questions_per_a_scene} questions per a video
        for i in range(num_questions_per_a_scene):

            # choose template of a question
            question_cls = random.choice([
                LocationQuestion, 
                VelocityQuestion, 
                SimpleCumulativeQuestion,
                ElaborativeCumulativeQuestion
                ])
            qa_generator = question_cls(annotation.locations, annotation.velocities)

            # choose objects and retrieve attributes 
            object_id = random.randint(0, annotation.locations.shape[0]-1)
            frame_id = random.randint(0, annotation.locations.shape[1]-1)

            color = annotation.object_properties[object_id].color
            material = annotation.object_properties[object_id].material
            shape = annotation.object_properties[object_id].shape

            square_error = None

            # generate QA text
            if isinstance(qa_generator, Union[LocationQuestion, VelocityQuestion]):
                question, answer = qa_generator.generate_qa(object_id, color, material, shape, frame_id)
            elif isinstance(qa_generator, Union[SimpleCumulativeQuestion, ElaborativeCumulativeQuestion]):
                interval = FRAME_RATE // 2
                start_frame_idx = random.randint(0, (annotation.locations.shape[1]-1-FRAME_RATE) // interval) * interval
                end_frame_idx = random.randint((start_frame_idx+FRAME_RATE) // interval, (annotation.locations.shape[1]-1) // interval) * interval
                related_collisions = [] 
                for c in annotation.collisions:
                    if (c.frame_id>=start_frame_idx and c.frame_id<end_frame_idx and object_id in c.object_ids):
                        related_collisions.append(c)
                related_collisions = sorted(related_collisions, key=lambda x: x.frame_id)

                question, answer, square_error = qa_generator.generate_qa(
                    object_id, 
                    annotation.object_properties, 
                    related_collisions, 
                    color, 
                    material,
                    shape, 
                    start_frame_idx, 
                    end_frame_idx
                    )
                
            question_dict = {
                "question_id": i,
                "question": question,
                "question_type": qa_generator.question_type,
                "question_subtype": qa_generator.question_subtype,
                "program": [],
                "square_error": square_error,
                "answer": answer
            }
            questions.append(question_dict)

        scene_index = annotation.scene_index
        video_filename = annotation.video_filename

        qa_dataset.append({
            "questions": questions,
            "scene_index": scene_index,
            "video_filename": video_filename
        })

    # export VQ dataset
    save_dir = os.path.join(destination_directory, "questions", dataset_type)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, dataset_type+".json")
    with open(save_path, 'w') as f:
        json.dump(qa_dataset, f)

def main()->None:
    train_annotations, valid_annotaitons = load_annotation()
    export_qa(train_annotations, "train")
    export_qa(valid_annotaitons, "validation")

if __name__ == "__main__":
    main()