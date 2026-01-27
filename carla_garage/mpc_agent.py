"""
MPC-based driving agent for data collection.
Collects training data in PlanT format using MPC trajectory planning.
"""

import os
import sys
import torch
import numpy as np
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import carla
import cv2
import math

from data_agent import DataAgent
from data import CARLA_Data
from config import GlobalConfig

# MPC controller import
sys.path.append(os.path.join(os.path.dirname(__file__), '../mpc'))
from mpc_controller import MPCController  # 기존 MPC 코드

SAVE_PATH = os.environ.get('SAVE_PATH', None)
jsonpickle_numpy.register_handlers()
jsonpickle.set_encoder_options('json', sort_keys=True, indent=4)


def get_entry_point():
    return 'MPCAgent'


class MPCAgent(DataAgent):
    """
    MPC-based driving agent for data collection.
    Uses Model Predictive Control to generate expert trajectories.
    """

    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        super().setup(path_to_conf_file, route_index, traffic_manager)

        torch.cuda.empty_cache()

        # Load config
        with open(os.path.join(path_to_conf_file, 'config.json'), 'rt', encoding='utf-8') as f:
            json_config = f.read()

        loaded_config = jsonpickle.decode(json_config)

        # Generate config
        self.config = GlobalConfig()
        self.config.__dict__.update(loaded_config.__dict__)

        self.config.debug = int(os.environ.get('VISU_PLANT', 0)) == 1
        self.device = torch.device('cuda:0')

        self.data = CARLA_Data(root=[], config=self.config, shared_dict=None)

        # ==================== MPC Setup ====================
        mpc_config = {
            'wheelbase': 2.875,
            'horizon': self.config.mpc_horizon if hasattr(self.config, 'mpc_horizon') else 15,
            'dt': self.config.mpc_dt if hasattr(self.config, 'mpc_dt') else 0.1,
            
            'Q': self.config.mpc_Q if hasattr(self.config, 'mpc_Q') else [100.0, 100.0, 50.0, 10.0],
            'R': self.config.mpc_R if hasattr(self.config, 'mpc_R') else [1.0, 5.0],
            'Qf': self.config.mpc_Qf if hasattr(self.config, 'mpc_Qf') else [200.0, 200.0, 100.0, 100.0],
            
            'a_min': -5.0,
            'a_max': 5.0,
            'kappa_min': -0.5,
            'kappa_max': 0.5,
            'ay_max': 5.0,
            
            'v_min': 0.0,
            'v_max': 8.0,  # m/s (~30 km/h, CARLA 도심 주행)
            'discount_rate': 0.95,
            
            'velocity_profile': {
                # ==================== 곡률 기준 ====================
                # κ (1/m) | 반경 (m) | 속도 비율
                # -----------------------------------------------
                'kappa_breakpoints': [
                    0.00,   # R = ∞ (직선)
                    0.01,   # R = 100m
                    0.02,   # R = 50m
                ],
                'velocity_ratios': [
                    1.0,    # 100% (직선)
                    0.65,   # 65%
                    0.25,   # 25% (급커브)
                ],
                # ==================================================
            }
        }
        
        self.mpc = MPCController(vehicle=None, global_path=None, config=mpc_config)
        print("MPC Controller initialized for data collection")
        # ===================================================

        if self.config.debug:
            self.init_map = False

        # Data collection stats
        self.total_steps = 0
        self.mpc_failures = 0

    def sensors(self):
        result = super().sensors()
        if self.config.debug:
            result += [{
                'type': 'sensor.camera.rgb',
                'x': self.config.camera_pos[0],
                'y': self.config.camera_pos[1],
                'z': self.config.camera_pos[2],
                'roll': self.config.camera_rot_0[0],
                'pitch': self.config.camera_rot_0[1],
                'yaw': self.config.camera_rot_0[2],
                'width': self.config.camera_width,
                'height': self.config.camera_height,
                'fov': self.config.camera_fov,
                'id': 'rgb_debug'
            }]
        return result

    def run_step(self, input_data, timestamp, sensors=None):
        """Main control loop with MPC"""
        
        if self.config.debug and not self.init_map:
            self.init_map = True

        # Get tick data (same as PlanT)
        tick_data = super().run_step(input_data, timestamp, plant=True)

        if self.config.debug:
            camera = input_data['rgb_debug'][1][:, :, :3]
            rgb_debug = cv2.cvtColor(camera, cv2.COLOR_BGR2RGB)
            rgb_debug = np.transpose(rgb_debug, (2, 0, 1))

        # ==================== Prepare MPC Input ====================
        # Ego state: [x, y, θ, v]
        # (ego 좌표계이므로 현재 위치는 항상 (0, 0, 0))
        current_state = np.array([0.0, 0.0, 0.0, tick_data['speed']])

        # Route preprocessing (PlanT와 동일)
        route = tick_data['route']
        if len(route) < self.config.num_route_points:
            num_missing = self.config.num_route_points - len(route)
            route = np.array(route)
            route = np.vstack((route, np.tile(route[-1], (num_missing, 1))))
        else:
            route = np.array(route[:self.config.num_route_points])

        if self.config.smooth_route:
            route = self.data.smooth_path(route)

        # Convert route to waypoints format for MPC
        waypoints = self._route_to_waypoints(route)

        # Bounding boxes (obstacles)
        bounding_boxes, _ = self.data.parse_bounding_boxes(tick_data['bounding_boxes'])
        # ===========================================================

        # ==================== MPC Solve ====================
        # Reference trajectory 생성
        ref_traj = self.mpc.get_reference_trajectory(waypoints)
        
        # MPC 최적화
        acceleration, curvature, success = self.mpc.solve(current_state, ref_traj)
        
        # Optimal trajectory 추출 (PlanT GT로 저장할 데이터)
        if success and self.mpc.optimal_trajectory is not None:
            opt_x, opt_y = self.mpc.optimal_trajectory
            mpc_trajectory = np.stack([opt_x, opt_y], axis=1)  # (horizon+1, 2)
            
            # 속도 프로파일도 저장
            if hasattr(self.mpc, 'prev_solution') and self.mpc.prev_solution is not None:
                X_opt = self.mpc.prev_solution['X']
                U_opt = self.mpc.prev_solution['U']
                mpc_velocities = X_opt[3, :]  # (horizon+1,)
                mpc_controls = U_opt  # (2, horizon)
            else:
                mpc_velocities = None
                mpc_controls = None
        else:
            mpc_trajectory = None
            mpc_velocities = None
            mpc_controls = None
            self.mpc_failures += 1
        # ===================================================

        # ==================== Control Execution ====================
        control = carla.VehicleControl()
        
        if success:
            # Curvature → Steering
            steering_angle = np.arctan(curvature * self.mpc.wheel_base)
            steering_angle = np.clip(steering_angle, -self.mpc.max_steer_angle, self.mpc.max_steer_angle)
            control.steer = float(steering_angle / self.mpc.max_steer_angle)

            # Acceleration → Throttle/Brake
            if acceleration > 0:
                control.throttle = float(np.clip(acceleration / 3.0, 0.0, 1.0))
                control.brake = 0.0
            else:
                control.throttle = 0.0
                control.brake = float(np.clip(-acceleration / 3.0, 0.0, 1.0))
        else:
            # MPC failed → Emergency brake
            control.throttle = 0.0
            control.brake = 1.0
            control.steer = 0.0
            print(f"MPC failed at step {self.total_steps}")
        # ===========================================================

        # ==================== Save MPC Output for Training ====================
        # PlanT 학습에 사용될 GT 데이터
        tick_data['mpc_output'] = {
            'trajectory': mpc_trajectory.tolist() if mpc_trajectory is not None else None,
            'velocities': mpc_velocities.tolist() if mpc_velocities is not None else None,
            'controls': mpc_controls.tolist() if mpc_controls is not None else None,
            'target_speed': ref_traj[3, 0] if ref_traj is not None else tick_data['speed'],
            'feasible': success,
            'acceleration': acceleration,
            'curvature': curvature,
        }
        # ======================================================================

        # ==================== Logging ====================
        self.total_steps += 1
        
        if self.total_steps % 100 == 0:
            success_rate = (1.0 - self.mpc_failures / self.total_steps) * 100
            print(f"📊 MPC Stats: Steps={self.total_steps}, "
                  f"Success={success_rate:.1f}%, "
                  f"Speed={tick_data['speed']*3.6:.1f} km/h")
        # ================================================

        # ==================== Visualization (Optional) ====================
        if self.config.debug and (not self.save_path is None):
            self._visualize_mpc(
                save_path=self.save_path,
                step=self.step,
                rgb=torch.tensor(rgb_debug) if self.config.debug else None,
                route=route,
                mpc_trajectory=mpc_trajectory,
                bounding_boxes=bounding_boxes,
                tick_data=tick_data
            )
        # ==================================================================

        return control

    def _route_to_waypoints(self, route):
        """
        Convert route to waypoints format for MPC
        
        Args:
            route: (N, 2) array of [x, y] in ego frame
            
        Returns:
            waypoints: list of dicts with keys ['x', 'y', 'yaw']
        """
        waypoints = []
        
        for i in range(len(route)):
            wp = {
                'x': route[i, 0],
                'y': route[i, 1],
                'z': 0.0,
            }
            
            # Yaw 계산 (다음 포인트 방향)
            if i < len(route) - 1:
                dx = route[i+1, 0] - route[i, 0]
                dy = route[i+1, 1] - route[i, 1]
                wp['yaw'] = np.arctan2(dy, dx)
            else:
                # 마지막 포인트는 이전 yaw 유지
                wp['yaw'] = waypoints[-1]['yaw'] if waypoints else 0.0
            
            waypoints.append(wp)
        
        return waypoints

    def _visualize_mpc(self, save_path, step, rgb, route, mpc_trajectory, 
                       bounding_boxes, tick_data):
        """Visualize MPC output (optional)"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left: BEV
        ax = axes[0]
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Route
        if route is not None:
            ax.plot(route[:, 0], route[:, 1], 'g*-', 
                   label='Route', markersize=8, alpha=0.6)
        
        # MPC trajectory
        if mpc_trajectory is not None:
            ax.plot(mpc_trajectory[:, 0], mpc_trajectory[:, 1], 'r--', 
                   linewidth=3, label='MPC Trajectory', alpha=0.8)
        
        # Ego vehicle
        ax.plot(0, 0, 'bo', markersize=15, label='Ego')
        
        # Bounding boxes
        if len(bounding_boxes) > 0:
            for bb in bounding_boxes:
                # bb: [class, x, y, yaw, extent_x, extent_y, speed, brake]
                x, y = bb[1], bb[2]
                ax.plot(x, y, 'rx', markersize=10)
        
        ax.legend()
        ax.set_title(f'Step {step}')
        
        # Right: RGB (if available)
        if rgb is not None:
            axes[1].imshow(rgb.permute(1, 2, 0).cpu().numpy())
            axes[1].set_title('Camera View')
            axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/mpc_vis_{step:06d}.png', dpi=100)
        plt.close()

    def destroy(self, results=None):
        """Cleanup"""
        print(f"\n📊 Final MPC Stats:")
        print(f"  Total steps: {self.total_steps}")
        print(f"  MPC failures: {self.mpc_failures}")
        print(f"  Success rate: {(1.0 - self.mpc_failures / max(self.total_steps, 1)) * 100:.2f}%")
        
        super().destroy(results)