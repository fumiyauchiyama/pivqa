import numpy as np
from typing import Tuple, Any, List

from dataset.annotation import ObjectProperty, Collision
from constants import FRAME_RATE, ERROR_LIMIT

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

    def fetch_location(
            self, 
            object_id:int, 
            frame_idx:int
            )->Tuple[float, float, float]:
        x, y, z = self.locations[object_id, frame_idx]
        return x, y, z
    
    def fetch_velocity(
            self, 
            object_id:int, 
            frame_idx:int
            )->Tuple[float, float, float]:
        vx, vy, vz = self.velocities[object_id, frame_idx]
        return vx, vy, vz

    def generate_qa(self, *args: Any, **kwargs: Any)->Tuple[str, str]:
        raise NotImplementedError
    
    def calc_interval(
            self,
            object_id:int,
            former_frame_idx:int,
            latter_frame_idx:int
            )->Tuple[float, float, float, float, float, float]:
        middle_frame_id = int((former_frame_idx + latter_frame_idx) / 2 )
        vx, vy, vz = self.fetch_velocity(object_id, middle_frame_id)
        dx_interval, dy_interval, dz_interval = ((latter_frame_idx-former_frame_idx)/FRAME_RATE) * (vx, vy, vz)
        return vx, vy, vz, dx_interval, dy_interval, dz_interval
    
class LocationQuestion(Question):
    def __init__(
            self, 
            locations:np.ndarray,
            velocities:np.ndarray,
            question_template:str=(
                "Where is the {color} {shape} at {time}s? "
                "Please provide the coordinates in three dimensions."
                ), 
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
        x, y, z = self.fetch_location(object_id, frame_idx)
        answer = self.answer_template.format(x=x, y=y, z=z)
        return question, answer

class VelocityQuestion(Question):
    def __init__(
            self, 
            locations:np.ndarray,
            velocities:np.ndarray,
            question_template:str=(
                "What is the velocity of the {color} {shape} at {time}s? "
                "Please provide the coordinates in three dimensions."
                ), 
            question_type:str="physical", 
            question_subtype:str="velocity", 
            answer_template:str="[{vx}, {vy}, {vz}]"
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
        vx, vy, vz = self.fetch_velocity(object_id, frame_idx)
        answer = self.answer_template.format(vx=vx, vy=vy, vz=vz)
        return question, answer

class SimpleCumulativeQuestion(Question):
    def __init__(
            self, 
            locations:np.ndarray,
            velocities:np.ndarray,
            question_template:str=(
                "How much has the {color} {shape} moved between {start_time}s and {end_time}s? "
                "Please provide the vector components in three dimensions."
                ), 
            question_type:str="cumulative", 
            question_subtype:str="simple", 
            answer_template:str="[{dx}, {dy}, {dz}]"
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
            objects:List[ObjectProperty],
            related_collisions:List[Collision],
            color:str,
            shape:str,
            start_frame_idx:int,
            end_frame_idx:int
            )->Tuple[str, str]:
        start_time = start_frame_idx / FRAME_RATE
        end_time = end_frame_idx / FRAME_RATE
        question = self.question_template.format(color=color, shape=shape, start_time=start_time, end_time=end_time)
        
        assert len(related_collisions)>=0

        interval_list:List[Any] = []
        
        vx: float
        vy: float
        vz: float
        dx_interval: float
        dy_interval: float
        dz_interval: float

        # Explain first interval
        if len(related_collisions)==0:
            answer = f"There are no collisions related to the {color} {shape} in between {start_time}s and {end_time}s. "
            vx, vy, vz, dx_interval, dy_interval, dz_interval = self.calc_interval(
                object_id, 
                start_frame_idx, 
                end_frame_idx
                )
            answer += (
                f"from {start_time}s to {end_time}s, "
                f"the motion pattern continued with velocity vectors [{vx}, {vy}, {vz}]. "
                f"During this time, the {color} {shape} moves [{dx_interval}, {dy_interval}, {dz_interval}"
            )
            interval_list.append([dx_interval, dy_interval, dz_interval])
        elif len(related_collisions)>0:
            if len(related_collisions)==1:
                answer = f"There are one collision related to the {color} {shape} in the following times: {', '.join([f'{t.frame_id / FRAME_RATE}s' for t in related_collisions])}. "
            else:
                answer = f"There are {len(related_collisions)} collisions related to the {color} {shape} in the following times: {', '.join([f'{t.frame_id / FRAME_RATE}s' for t in related_collisions])}. "
            
            latter_frame_idx = related_collisions[0].frame_id
            vx, vy, vz, dx_interval, dy_interval, dz_interval = self.calc_interval(
                object_id, 
                start_frame_idx, 
                latter_frame_idx)
            answer += (
                f"from {start_time}s to {(related_collisions[0].frame_id/FRAME_RATE)}s, "
                f"the motion pattern continued with velocity vectors [{vx}, {vy}, {vz}]. "
                f"During this time, the {color} {shape} moves [{dx_interval}, {dy_interval}, {dz_interval}"
            )
            interval_list.append([dx_interval, dy_interval, dz_interval])

        for idx, collision in enumerate(related_collisions):
            another_objects = collision.object_ids[0] if collision.object_ids[0]!=object_id else collision.object_ids[1]
            ax, ay, az = self.fetch_velocity(object_id, collision.frame_id + int(FRAME_RATE/2))
            timestamp = collision.frame_id/FRAME_RATE
            collision_info =  (
                f"At timestamp {timestamp}, the {color} {shape} collided with the {objects[another_objects].color} {objects[another_objects].shape}, "
                f"altering its velocity vector to [{ax}, {ay}, {az}]. "
            )

            latter_frame_idx = end_frame_idx if idx==len(related_collisions)-1 else related_collisions[idx+1].frame_id
            latter_time = latter_frame_idx / FRAME_RATE
            time_interval = (latter_frame_idx - collision.frame_id) / FRAME_RATE
            vx_middle: float
            vy_middle: float
            vz_middle: float
            dx_interval_middle: float
            dy_interval_middle: float
            dz_interval_middle: float
            vx_middle, vy_middle, vz_middle, dx_interval_middle, dy_interval_middle, dz_interval_middle = self.calc_interval(
                object_id, 
                collision.frame_id, 
                latter_frame_idx)
            motion_pattern = (
                f"The motion pattern continued for {latter_time} - {timestamp} = {time_interval}[s] with velocity vectors [{vx_middle}, {vy_middle}, {vz_middle}]. "
                f"During this time, the {color} {shape} moves [{dx_interval_middle}, {dy_interval_middle}, {dz_interval_middle}]."
                )
            answer += collision_info + motion_pattern
            interval_list.append([dx_interval_middle, dy_interval_middle, dz_interval_middle])

        total_repr_x = str(interval_list[0][0])
        total_repr_y = str(interval_list[0][1])
        total_repr_z = str(interval_list[0][2])

        total_x, total_y, total_z = 0, 0, 0

        for idx, interval in enumerate(interval_list):
            total_x += interval[0]
            total_y += interval[1]
            total_z += interval[2]
            if idx!=0:
                total_repr_x += f" + {interval[0]}"
                total_repr_y += f" + {interval[1]}"
                total_repr_z += f" + {interval[2]}"
            
        answer += (
            f"Therefore, the cumulative movement is "
            f"[{total_repr_x}, {total_repr_y}, {total_repr_z}] = [{total_x}, {total_y}, {total_z}]."
            )

        location_end = self.fetch_location(object_id, end_frame_idx)
        location_start = self.fetch_location(object_id, start_frame_idx)

        delta_x = location_end[0] - location_start[0]
        delta_y = location_end[1] - location_start[1]
        delta_z = location_end[2] - location_start[2]

        square_error:float = abs((delta_x**2 + delta_y**2 + delta_z**2) - (total_x**2 + total_y**2 + total_z**2))
        assert  square_error<=ERROR_LIMIT, f"square error is more than {ERROR_LIMIT}: {square_error}"

        simple_answer = self.answer_template.format(x=total_x, y=total_y, z=total_z)
        return question, simple_answer