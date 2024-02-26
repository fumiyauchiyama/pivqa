from typing import Tuple, List, Literal
import numpy as np

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