"""
MPC Control Inference Agent for Racing
Predicts acceleration and curvature using trained PlanT model
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import carla

from pathlib import Path
from omegaconf import OmegaConf

from model_race import HFLM
from dataset_race import PlanTDataset


class MPCControlAgent:
    """
    Agent that predicts MPC controls (acceleration, curvature) using PlanT
    """
    
    def __init__(self, checkpoint_path, config_path=None):
        """
        Args:
            checkpoint_path: Path to .ckpt file
            config_path: Path to config.yaml (optional)
        """
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Load config
        if config_path is None:
            config_path = "/workspace/plant2/PlanT/config/config.yaml"
        
        with open(config_path, 'r') as f:
            cfg = OmegaConf.load(f)
        
        # Load model config
        model_config_path = "/workspace/plant2/PlanT/config/model/PlanT.yaml"
        with open(model_config_path, 'r') as f:
            model_cfg = OmegaConf.load(f)
        
        cfg.model = model_cfg
        self.cfg = cfg
        
        # Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        # checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Initialize model
        from lit_module_race import LitHFLM
        self.model = LitHFLM.load_from_checkpoint(
            checkpoint_path,
            cfg=cfg,
            map_location=self.device
        )
        self.model.eval()
        self.model.to(self.device)
        
        print(f"Model loaded successfully on {self.device}")
        print(f"MPC controls enabled: {self.model.model.use_mpc_controls}")
        
        # Initialize dataset helper (for preprocessing)
        self.dataset = PlanTDataset(
            root="/workspace/data/dummy",  # Not used for inference
            cfg=cfg,
            shared_dict=None
        )
    
    @torch.inference_mode()
    def predict(self, 
                ego_speed,           # float: m/s
                route,               # np.array: (20, 2) - local frame
                bounding_boxes=None, # list of dicts
                speed_limit=50.0):   # float: km/h
        """
        Predict MPC controls from current state
        
        Args:
            ego_speed: Current speed in m/s
            route: Route waypoints in ego frame (20, 2)
            bounding_boxes: Optional list of bounding boxes
            speed_limit: Speed limit in km/h
        
        Returns:
            controls: dict with keys:
                - accelerations: np.array (N,)
                - curvatures: np.array (N,)
                - waypoints: np.array (M, 2) - predicted trajectory
        """
        
        # ==================== Preprocess inputs ====================
        
        # 1. Route (20 points)
        route_tensor = torch.tensor(route[:20], dtype=torch.float32).unsqueeze(0)
        route_tensor = route_tensor.to(self.device)
        
        # 2. Speed limit category
        speed_limit_kmh = round(speed_limit)
        speed_cat = self.dataset.speed_cats.get(speed_limit_kmh, 1)
        speed_limit_tensor = torch.tensor([speed_cat], dtype=torch.int).to(self.device)
        
        # 3. Ego speed
        ego_speed_tensor = torch.tensor([ego_speed], dtype=torch.float32).to(self.device)
        
        # 4. Bounding boxes (objects)
        if bounding_boxes is None or len(bounding_boxes) == 0:
            # Empty scene
            x_objs = torch.tensor([[0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32)
            batch_idxs = torch.zeros((1, 0), dtype=torch.int32)
        else:
            x_objs = [[0, 0, 0, 0, 0, 0, 0]]  # Padding
            input_objects = []
            
            for bb in bounding_boxes:
                # Parse bounding box
                obj = [
                    bb.get('class_id', 1),      # type
                    bb['position'][0],          # x
                    bb['position'][1],          # y
                    bb['yaw'],                  # yaw (degrees)
                    bb.get('speed', 0.0) * 3.6, # speed (km/h)
                    bb['extent'][1] * 2,        # width
                    bb['extent'][0] * 2,        # length
                ]
                input_objects.append(obj)
            
            x_objs.extend(input_objects)
            x_objs = torch.tensor(x_objs, dtype=torch.float32)
            
            batch_idxs = torch.zeros((1, len(input_objects)), dtype=torch.int32)
            batch_idxs[0, :] = torch.arange(1, len(input_objects) + 1)
        
        x_objs = x_objs.to(self.device)
        batch_idxs = batch_idxs.to(self.device)
        
        # 5. Dummy outputs (not used in inference)
        # y_objs = torch.tensor([[-999, -999, -999, -999]], dtype=torch.long).to(self.device)
        

        # 6. Dummy waypoints (for shape compatibility)
        waypoints = torch.zeros((1, self.cfg.model.waypoints.wps_len, 2), dtype=torch.float32)
        waypoints = waypoints.to(self.device)
        
        # ==================== Build batch ====================
        batch = {
            'idxs': batch_idxs,
            'x_objs': x_objs,
            'y_objs': None,
            'route_original': route_tensor,
            'route': route_tensor,
            'speed_limit': speed_limit_tensor,
            'waypoints': waypoints,
            'target_speed': ego_speed_tensor,
            'ego_speed': ego_speed_tensor,
        }
        
        # ==================== Forward pass ====================
        logits, targets, pred_plan, attn_map = self.model.model(batch)
        
        pred_path, pred_wps, pred_speed, pred_controls = pred_plan
        
        # ==================== Extract results ====================
        results = {}
        
        # 1. Waypoints (trajectory)
        if pred_wps is not None:
            waypoints_pred = pred_wps[0].detach().cpu().numpy()  # (N, 2)
            results['waypoints'] = waypoints_pred
        
        # 2. MPC Controls
        if pred_controls is not None:
            controls = pred_controls[0].detach().cpu().numpy()  # (N, 2)
            results['accelerations'] = controls[:, 0]  # (N,)
            results['curvatures'] = controls[:, 1]     # (N,)
            
            # First step control (immediate action)
            results['acceleration'] = float(controls[0, 0])
            results['curvature'] = float(controls[0, 1])
        else:
            print("WARNING: Model did not output controls!")
            results['accelerations'] = np.zeros(self.cfg.model.waypoints.wps_len)
            results['curvatures'] = np.zeros(self.cfg.model.waypoints.wps_len)
            results['acceleration'] = 0.0
            results['curvature'] = 0.0
        
        # 3. Path (optional)
        if pred_path is not None:
            path_pred = pred_path[0].detach().cpu().numpy()
            results['path'] = path_pred
        
        return results
    
    def control_to_carla(self, acceleration, curvature, current_speed, 
                         wheelbase=2.875, max_steer=1.22, dt=0.05):
        """
        Convert MPC controls to CARLA VehicleControl
        
        Args:
            acceleration: m/s^2
            curvature: 1/m
            current_speed: m/s
            wheelbase: vehicle wheelbase (m)
            max_steer: max steering angle (rad)
            dt: control timestep (s)
        
        Returns:
            carla.VehicleControl
        """
        
        # Curvature → Steering angle
        steering_angle = np.arctan(curvature * wheelbase)
        steering_angle = np.clip(steering_angle, -max_steer, max_steer)
        steer = float(steering_angle / max_steer)  # Normalize to [-1, 1]
        
        # Acceleration → Throttle/Brake
        if acceleration > 0:
            throttle = float(np.clip(acceleration / 8.0, 0.0, 1.0))
            brake = 0.0
        else:
            throttle = 0.0
            brake = float(np.clip(-acceleration / 8.0, 0.0, 1.0))
        
        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = throttle
        control.brake = brake
        
        return control


# ==================== Usage Example ====================

def main():
    """Example usage"""
    
    # 1. Initialize agent
    checkpoint_path = "/workspace/plant2/PlanT/checkpoints/last_42.ckpt"
    agent = MPCControlAgent(checkpoint_path)
    
    # 2. Prepare inputs (example)
    ego_speed = 10.0  # m/s
    
    # Route in ego frame (20 waypoints)
    route = np.array([
        [2.0 * i, 0.1 * i**2] for i in range(20)
    ])
    
    # Bounding boxes (optional)
    bounding_boxes = [
        {
            'class_id': 1,  # vehicle
            'position': [10.0, 2.0, 0.0],
            'yaw': 5.0,  # degrees
            'speed': 8.0,  # m/s
            'extent': [2.0, 1.0, 0.8],  # length, width, height
        }
    ]
    
    # 3. Predict
    results = agent.predict(
        ego_speed=ego_speed,
        route=route,
        bounding_boxes=bounding_boxes,
        speed_limit=50.0
    )
    
    # 4. Results
    print("=" * 60)
    print("MPC Control Prediction Results:")
    print("=" * 60)
    print(f"Immediate acceleration: {results['acceleration']:.3f} m/s²")
    print(f"Immediate curvature: {results['curvature']:.4f} 1/m")
    print(f"\nAcceleration sequence: {results['accelerations']}")
    print(f"Curvature sequence: {results['curvatures']}")
    print(f"\nPredicted waypoints:\n{results['waypoints']}")
    
    # 5. Convert to CARLA control
    control = agent.control_to_carla(
        results['acceleration'],
        results['curvature'],
        ego_speed
    )
    
    print("\n" + "=" * 60)
    print("CARLA Control:")
    print("=" * 60)
    print(f"Steer: {control.steer:.3f}")
    print(f"Throttle: {control.throttle:.3f}")
    print(f"Brake: {control.brake:.3f}")


if __name__ == '__main__':
    main()