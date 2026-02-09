"""
MPC Control Inference Agent for Racing
Predicts acceleration and curvature using trained PlanT model
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import carla
import pickle

from pathlib import Path
from omegaconf import OmegaConf

from model_race import HFLM
from dataset_race import PlanTDataset


class MPCControlAgent:
    """
    Agent that predicts MPC controls (acceleration, curvature) using PlanT
    """
    
    def __init__(self, checkpoint_path, raceline_file, config_path=None):
        """
        Args:
            checkpoint_path: Path to .ckpt file
            raceline_file: Path to raceline .pkl file
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
        
        # ✅ Load raceline (for feedforward)
        self.raceline = self._load_raceline(raceline_file)
        print(f"Loaded raceline: {len(self.raceline)} waypoints")
        
        # Cache
        self.prev_closest_idx = None
        
        # ✅ 디버깅 통계 초기화 (맨 마지막에 추가!)
        self.debug_stats = {
            'predictions': {'accel': [], 'kappa_fb': []},
            'applied': {'accel': [], 'kappa_total': []},
            'overrides': {'count': 0, 'events': []},
        }
        print("  Debug stats initialized" )
    
    def _load_raceline(self, filename):
        """Load and correct raceline yaw"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        raceline = data['raceline']
        
        # YAW offset correction (same as data collection)
        first_wp = raceline[0]
        second_wp = raceline[1]
        
        dx = second_wp['x'] - first_wp['x']
        dy = second_wp['y'] - first_wp['y']
        actual_yaw = np.arctan2(dy, dx)
        
        yaw_offset = actual_yaw - first_wp['yaw']
        yaw_offset = np.arctan2(np.sin(yaw_offset), np.cos(yaw_offset))
        
        for wp in raceline:
            wp['yaw_original'] = wp['yaw']
            wp['yaw'] = wp['yaw'] + yaw_offset
            wp['yaw'] = np.arctan2(np.sin(wp['yaw']), np.cos(wp['yaw']))
        
        print(f"  YAW offset applied: {np.rad2deg(yaw_offset):.2f}°")
        
        return raceline
    
    def get_lookahead_waypoints(self, ego_transform, lookahead=20):
        """
        Extract lookahead waypoints from raceline
        ✅ Same as data collection
        
        Returns:
            lookahead_wps: list of dicts with local frame coords
            min_dist: distance to closest waypoint
            closest_idx: index of closest waypoint
        """
        ego_x = ego_transform.location.x
        ego_y = ego_transform.location.y
        ego_yaw = np.deg2rad(ego_transform.rotation.yaw)
        
        # Find closest waypoint
        min_dist = float('inf')
        closest_idx = 0
        
        for i in range(len(self.raceline)):
            wp = self.raceline[i]
            dx = wp['x'] - ego_x
            dy = wp['y'] - ego_y
            dist = np.sqrt(dx**2 + dy**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # Extract lookahead waypoints (transform to ego frame)
        lookahead_wps = []
        cos_yaw = np.cos(-ego_yaw)
        sin_yaw = np.sin(-ego_yaw)
        
        for i in range(lookahead):
            idx = (closest_idx + i) % len(self.raceline)
            wp = self.raceline[idx]
            
            # Global → Ego frame
            dx = wp['x'] - ego_x
            dy = wp['y'] - ego_y
            
            local_x = cos_yaw * dx - sin_yaw * dy
            local_y = sin_yaw * dx + cos_yaw * dy
            local_yaw = wp['yaw'] - ego_yaw
            
            lookahead_wps.append({
                'x': local_x,
                'y': local_y,
                'yaw': local_yaw,
                'velocity': wp['velocity'],      # ✅ From raceline
                'curvature': wp['curvature']     # ✅ From raceline (feedforward!)
            })
        
        self.prev_closest_idx = closest_idx
        
        return lookahead_wps, min_dist, closest_idx
    
    @torch.inference_mode()
    def predict(self, 
                ego_speed,              # float: m/s
                route,                  # np.array: (20, 2) - LOCAL FRAME
                target_speed=None,      # float: m/s (from raceline)
                bounding_boxes=None,    # list of dicts
                speed_limit=50.0):      # float: km/h
        """
        Predict MPC controls from current state
        
        Args:
            ego_speed: Current speed in m/s
            route: Route waypoints in LOCAL ego frame (20, 2)
            target_speed: Target speed from raceline (m/s)
            bounding_boxes: Optional list of bounding boxes
            speed_limit: Speed limit in km/h
        
        Returns:
            controls: dict with keys:
                - acceleration: float
                - curvature_fb: float (feedback only)
                - accelerations: np.array (N,)
                - curvatures_fb: np.array (N,)
                - waypoints: np.array (M, 2)
        """
        
        # 1. Route (20 points in LOCAL frame)
        if len(route) < 20:
            route_padded = np.zeros((20, 2))
            route_padded[:len(route)] = route
            route = route_padded
        
        route_tensor = torch.tensor(route[:20], dtype=torch.float32).unsqueeze(0)
        route_tensor = route_tensor.to(self.device)
        
        # 2. Speed limit category
        speed_limit_kmh = round(speed_limit)
        speed_cat = self.dataset.speed_cats.get(speed_limit_kmh, 1)
        speed_limit_tensor = torch.tensor([speed_cat], dtype=torch.int).to(self.device)
        
        # 3. Ego speed
        ego_speed_tensor = torch.tensor([ego_speed], dtype=torch.float32).to(self.device)
        
        # 4. Target speed (from raceline)
        if target_speed is None:
            target_speed = ego_speed
        target_speed_tensor = torch.tensor([target_speed], dtype=torch.float32).to(self.device)
        
        # 5. Bounding boxes
        if bounding_boxes is None or len(bounding_boxes) == 0:
            x_objs = torch.tensor([[0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32)
            batch_idxs = torch.zeros((1, 0), dtype=torch.int32)
        else:
            x_objs = [[0, 0, 0, 0, 0, 0, 0]]
            input_objects = []
            
            for bb in bounding_boxes:
                obj = [
                    bb.get('class_id', 1),
                    bb['position'][0],
                    bb['position'][1],
                    bb['yaw'],
                    bb.get('speed', 0.0) * 3.6,
                    bb['extent'][1] * 2,
                    bb['extent'][0] * 2,
                ]
                input_objects.append(obj)
            
            x_objs.extend(input_objects)
            x_objs = torch.tensor(x_objs, dtype=torch.float32)
            
            batch_idxs = torch.zeros((1, len(input_objects)), dtype=torch.int32)
            batch_idxs[0, :] = torch.arange(1, len(input_objects) + 1)
        
        x_objs = x_objs.to(self.device)
        batch_idxs = batch_idxs.to(self.device)
        
        # 6. Dummy waypoints
        waypoints = torch.zeros((1, self.cfg.model.waypoints.wps_len, 2), dtype=torch.float32)
        waypoints = waypoints.to(self.device)
        
        # Build batch
        batch = {
            'idxs': batch_idxs,
            'x_objs': x_objs,
            'y_objs': None,
            'route_original': route_tensor,
            'route': route_tensor,
            'speed_limit': speed_limit_tensor,
            'waypoints': waypoints,
            'target_speed': target_speed_tensor,
            'ego_speed': ego_speed_tensor,
        }
        
        # Forward pass
        logits, targets, pred_plan, attn_map = self.model.model(batch)
        pred_path, pred_wps, pred_speed, pred_controls = pred_plan
        
        # Extract results
        results = {}
        
        if pred_wps is not None:
            waypoints_pred = pred_wps[0].detach().cpu().numpy()
            results['waypoints'] = waypoints_pred
        
        if pred_controls is not None:
            controls = pred_controls[0].detach().cpu().numpy()  # (N, 2)
            
            # ✅ Model outputs [a, κ_fb]
            results['accelerations'] = controls[:, 0] 
            results['curvatures_fb'] = controls[:, 1] 
            
            results['acceleration'] = float(controls[0, 0])
            results['curvature_fb'] = float(controls[0, 1])
        else:
            print("WARNING: Model did not output controls!")
            results['accelerations'] = np.zeros(self.cfg.model.waypoints.wps_len)
            results['curvatures_fb'] = np.zeros(self.cfg.model.waypoints.wps_len)
            results['acceleration'] = 0.0
            results['curvature_fb'] = 0.0
        
        if pred_path is not None:
            path_pred = pred_path[0].detach().cpu().numpy()
            results['path'] = path_pred
        
        return results
    
    # def control_to_carla(self,
    #                     acceleration,
    #                     curvature_fb,
    #                     curvature_ff,
    #                     current_speed,
    #                     target_speed,          # ✅ 추가
    #                     wheelbase=2.875,
    #                     max_steer=1.22,
    #                     kappa_min=-0.2,
    #                     kappa_max=0.2,
    #                     a_max=12.0,
    #                     a_min=-12.0):

    #     # --- curvature는 그대로 ---
    #     kappa_cmd = np.clip(curvature_fb + curvature_ff, kappa_min, kappa_max)

    #     steering_angle = np.arctan(kappa_cmd * wheelbase)
    #     steering_angle = np.clip(steering_angle, -max_steer, max_steer)
    #     steer = float(steering_angle / max_steer)

    #     # ================== 🔴 핵심 ==================
    #     # 속도 초과 시 가속 금지
    #     if current_speed > target_speed:
    #         acceleration = min(acceleration, 0.0)
    #         print("→ CLAMPED")
    #     else:
    #         print("→ OK")

    #     # (선택) 차량 물리 한계
    #     # acceleration = np.clip(acceleration, -6.0, 4.0)
    #     # ============================================

    #     if acceleration > 0:
    #         throttle = float(acceleration / a_max)
    #         brake = 0.0
    #     else:
    #         throttle = 0.0
    #         brake = float(-acceleration / abs(a_min))

    #     control = carla.VehicleControl()
    #     control.steer = steer
    #     control.throttle = np.clip(throttle, 0.0, 1.0)
    #     control.brake = np.clip(brake, 0.0, 1.0)

    #     return control
  
    def control_to_carla(self,
                        acceleration,
                        curvature_fb,
                        curvature_ff,
                        current_speed,
                        target_speed,
                        wheelbase=2.875,
                        max_steer=1.22,
                        kappa_min=-0.2,
                        kappa_max=0.2,
                        a_max=12.0,
                        a_min=-12.0):
        
        # ✅ Save original predictions
        accel_before = acceleration
        
        # # Curvature
        kappa_cmd = curvature_fb + curvature_ff
        # kappa_cmd = 0.7 * curvature_ff + 1.0 * curvature_fb

        
        steering_angle = np.arctan(kappa_cmd * wheelbase)
        steering_angle = np.clip(steering_angle, -max_steer, max_steer)
        steer = float(steering_angle / max_steer)
        
        # ✅ Speed override
        override_applied = False
        if current_speed > target_speed:
            acceleration = min(acceleration, 0.0)
            override_applied = True
            self.debug_stats['overrides']['count'] += 1
            
            # Record override event
            self.debug_stats['overrides']['events'].append({
                'speed': current_speed,
                'target': target_speed,
                'accel_before': accel_before,
                'accel_after': acceleration,
            })
        
        # ✅ Statistics
        self.debug_stats['predictions']['accel'].append(accel_before)
        self.debug_stats['predictions']['kappa_fb'].append(curvature_fb)
        self.debug_stats['applied']['accel'].append(acceleration)
        self.debug_stats['applied']['kappa_total'].append(kappa_cmd)
        
        # Throttle/Brake
        if acceleration > 0:
            throttle = float(acceleration / a_max)
            brake = 0.0
        else:
            throttle = 0.0
            brake = float(-acceleration / abs(a_min))
        
        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = np.clip(throttle, 0.0, 1.0)
        control.brake = np.clip(brake, 0.0, 1.0)
        
        # ✅ Add debug info to control object
        control.debug_info = {
            'accel_model': accel_before,
            'accel_applied': acceleration,
            'kappa_fb': curvature_fb,
            'kappa_ff': curvature_ff,
            'kappa_total': kappa_cmd,
            'override': override_applied,
        }
        
        return control
    
    def print_debug_summary(self):
        """디버깅 통계 출력"""
        import numpy as np
        
        pred_accels = np.array(self.debug_stats['predictions']['accel'])
        pred_kappas = np.array(self.debug_stats['predictions']['kappa_fb'])
        applied_accels = np.array(self.debug_stats['applied']['accel'])
        applied_kappas = np.array(self.debug_stats['applied']['kappa_total'])
        
        print(f"\n{'='*80}")
        print(f"DEBUG SUMMARY")
        print(f"{'='*80}")
        
        print(f"\n📊 Model Predictions:")
        print(f"  Acceleration: mean={np.mean(pred_accels):+.2f}, "
              f"std={np.std(pred_accels):.2f}, "
              f"range=[{np.min(pred_accels):+.2f}, {np.max(pred_accels):+.2f}]")
        print(f"  κ_fb:         mean={np.mean(pred_kappas):+.6f}, "
              f"std={np.std(pred_kappas):.6f}, "
              f"range=[{np.min(pred_kappas):+.6f}, {np.max(pred_kappas):+.6f}]")
        
        print(f"\n🎮 Applied Controls:")
        print(f"  Acceleration: mean={np.mean(applied_accels):+.2f}, "
              f"std={np.std(applied_accels):.2f}, "
              f"range=[{np.min(applied_accels):+.2f}, {np.max(applied_accels):+.2f}]")
        print(f"  κ_total:      mean={np.mean(applied_kappas):+.6f}, "
              f"std={np.std(applied_kappas):.6f}, "
              f"range=[{np.min(applied_kappas):+.6f}, {np.max(applied_kappas):+.6f}]")
        
        print(f"\n⚠️  Override Events:")
        print(f"  Total overrides: {self.debug_stats['overrides']['count']} / {len(pred_accels)} "
              f"({100*self.debug_stats['overrides']['count']/max(len(pred_accels),1):.1f}%)")
        
        if len(self.debug_stats['overrides']['events']) > 0:
            recent_events = self.debug_stats['overrides']['events'][-5:]
            print(f"  Last 5 overrides:")
            for i, event in enumerate(recent_events):
                print(f"    {i+1}. speed={event['speed']*3.6:.1f} > target={event['target']*3.6:.1f} km/h | "
                      f"accel: {event['accel_before']:+.2f} → {event['accel_after']:+.2f}")
        
        print(f"{'='*80}\n")