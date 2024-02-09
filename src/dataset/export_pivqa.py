import json
import os
from tqdm import tqdm
import numpy as np
from typing import Tuple, List, Literal, Dict, Any
import random

from constants import CLEVRER_PATH, PIVQA_DATASET_DIR, SEED, FRAME_RATE

source_directory = CLEVRER_PATH
destination_directory = PIVQA_DATASET_DIR

class ObjectProperty:
    def __init__(self, object_id:int, color:str, material:str, shape:str):
        self.object_id = object_id
        self.color = color
        self.material = material
        self.shape = shape

    def __repr__(self)->str:
        return f"ObjectProperty(object_id={self.object_id}, color={self.color}, material={self.material}, shape={self.shape})"

class Collision:
    def __init__(self, object_ids:List[int], frame_id:int, location:List[int]):
        assert len(object_ids)==2, "Please specify two objects to collide."
        assert len(location)==3, "Please specify the 3D location."
        self.object_ids = object_ids
        self.frame_id = frame_id
        self.location = location

class Annotation:
    def __init__(
            self, 
            scene_index:int, 
            video_filename:str, 
            object_properties:List[ObjectProperty], 
            locations:np.ndarray, 
            velocities:np.ndarray,
            collisions:List[Collision]
            ):
        self.scene_index = scene_index
        self.video_filename = video_filename
        self.object_properties = object_properties
        self.locations = locations
        self.velocities = velocities
        self.collisions = collisions

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
        dest_root = os.path.join(destination_directory, os.path.relpath(root, source_directory))

        if not "annotations" in dest_root:
            continue

        os.makedirs(dest_root, exist_ok=True)

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

class Question:
    def __init__(
            self, 
            locations:np.ndarray,
            velocities:np.ndarray,
            question_template:str, 
            question_type:str, 
            question_subtype:str, 
            answer_template:str
            ):
        self.locations = locations
        self.velocities = velocities
        self.question_template = question_template
        self.question_type = question_type
        self.question_subtype = question_subtype
        self.answer_template = answer_template

    def calc_location(
            self, 
            object_id:int, 
            frame_idx:int
            )->Tuple[float, float, float]:
        x, y, z = self.locations[object_id, frame_idx]
        return x, y, z
    
    def calc_velocity(
            self, 
            object_id:int, 
            frame_idx:int
            )->Tuple[float, float, float]:
        x, y, z = self.velocities[object_id, frame_idx]
        return x, y, z

    def generate_qa(self, *args: Any, **kwargs: Any)->Tuple[str, str]:
        raise NotImplementedError
    
class LocationQuestion(Question):
    def __init__(
            self, 
            locations:np.ndarray,
            velocities:np.ndarray,
            question_template:str="""Where is the {color} {shape} at {time}s? Please provide the coordinates in three dimensions.""", 
            question_type:str="physical", 
            question_subtype:str="location", 
            answer_template:str="[{x}, {y}, {z}]"
            ):
        super().__init__(
            locations,
            velocities, 
            question_template, 
            question_type, 
            question_subtype, 
            answer_template
            )

    def generate_qa(
            self,
            object_id:int,
            color:str,
            shape:str,
            frame_idx:int
            )->Tuple[str, str]:
        time = frame_idx / FRAME_RATE
        question = self.question_template.format(color=color, shape=shape, time=time)
        x, y, z = self.calc_location(object_id, frame_idx)
        answer = self.answer_template.format(x=x, y=y, z=z)
        return question, answer

class VelocityQuestion(Question):
    def __init__(
            self, 
            locations:np.ndarray,
            velocities:np.ndarray,
            question_template:str="""Where is the {color} {shape} at {time}s? Please provide the coordinates in three dimensions.""", 
            question_type:str="physical", 
            question_subtype:str="location", 
            answer_template:str="[{x}, {y}, {z}]"
            ):
        super().__init__(
            locations,
            velocities, 
            question_template, 
            question_type, 
            question_subtype, 
            answer_template
            )

    def generate_qa(
            self,
            object_id:int,
            color:str,
            shape:str,
            frame_idx:int
            )->Tuple[str, str]:
        time = frame_idx / FRAME_RATE
        question = self.question_template.format(color=color, shape=shape, time=time)
        x, y, z = self.calc_velocity(object_id, frame_idx)
        answer = self.answer_template.format(x=x, y=y, z=z)
        return question, answer

def export_qa(
        annotations:List[Annotation], 
        dataset_type:Literal["train", "validation"],
        num_questions_per_a_scene:int=15,
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
            question_cls = random.choice([LocationQuestion, VelocityQuestion])
            qa_generator = question_cls(annotation.locations, annotation.velocities)

            # choose objects and retrieve attributes 
            object_id = random.randint(0, len(annotation.object_properties)-1)
            frame_id = random.randint(0, annotation.locations.shape[0]-1)

            color = annotation.object_properties[object_id].color
            shape = annotation.object_properties[object_id].shape

            # generate QA text
            question, answer = qa_generator.generate_qa(object_id, color, shape, frame_id)

            question_dict = {
                "question_id": i,
                "question": question,
                "question_type": qa_generator.question_type,
                "question_subtype": qa_generator.question_subtype,
                "program": [],
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
    export_qa(valid_annotaitons, "validation")

if __name__ == "__main__":
    main()