"""
This file is used to compare the differences between the collision body and visual body of Shadowhand, 
so a simple SAPIEN environment is used instead of being loaded into the MANISKILL environment.
"""

import sys
from pathlib import Path
cur_dir, project_dir = Path(__file__).parent, Path(__file__).parent.parent
sys.path.extend([p for p in [str(cur_dir), str(project_dir)] if p not in sys.path])
import numpy as np
import sapien.core as sapien
from sapien.core import Pose
from typing import Dict
from urdf_loader import URDFLoader
from typing import Union, List
from sapien.utils import Viewer
import gymnasium as gym
from gymnasium.utils import seeding
from typing import Union


class SapienEnv(gym.Env):
    """Superclass for Sapien environments."""

    def __init__(self, control_freq, timestep, **kwargs):
        self.control_freq = control_freq    # alias: frame_skip in mujoco_py
        self.timestep = timestep

        self._scene = sapien.Scene()
        self._scene.set_timestep(timestep)  

        self._build_world(**kwargs)
        self.viewer: Union[Viewer, None] = None
        self.seed()

    def _build_world(self):
        raise NotImplementedError()

    def _setup_viewer(self):
        raise NotImplementedError()

    # ---------------------------------------------------------------------------- #
    # Override gym functions
    # ---------------------------------------------------------------------------- #
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        if self.viewer is not None:
            pass  # release viewer

    def render(self, mode='human'):
        if mode == 'human':
            if self.viewer is None:
                self._setup_viewer()
            self._scene.update_render()
            self.viewer.render()
        else:
            raise NotImplementedError('Unsupported render mode {}.'.format(mode))


    def get_articulation(self, name) -> sapien.physx.PhysxArticulation:
        all_articulations = self._scene.get_all_articulations()
        articulation = [x for x in all_articulations if x.name == name]
        if len(articulation) > 1:
            raise RuntimeError(f'Not a unique name for articulation: {name}')
        elif len(articulation) == 0:
            raise RuntimeError(f'Articulation not found: {name}')
        return articulation[0]

    @property
    def dt(self):
        return self.timestep * self.control_freq

# Shadowhand environment
class ShadowhandEnv(SapienEnv):
    def __init__(self, urdf_path: str, srdf_path: str = None, init_pose: Pose = None, init_qpos: np.ndarray = None, disable_gravity: bool = True):
        self.urdf_path = urdf_path
        self.srdf_path = srdf_path
        super().__init__(control_freq=20, timestep=1 / 100)  
        self.robot = self.get_articulation("shadowhand")
        self.robot_no_collision = self.get_articulation("shadowhand_no_collision")
        self.setup_robot(init_qpos=init_qpos, disable_gravity=disable_gravity, stiffness=1000, damping=200)
        # print(self.robot.get_qpos())
        self.dof = self.robot.dof
        self.robot_no_collision_joints = self.robot_no_collision.get_active_joints()
        self.robot_joints = self.robot.get_active_joints()
        self.robot_joints_names = [joint.name for joint in self.robot_joints]
        self.robot_joints_ids = {name: i for i, name in enumerate(self.robot_joints_names)}
        self.robot_qliimits = self.robot.get_qlimits() 



    def _build_world(self, **kwargs):

        physical_material = sapien.physx.PhysxMaterial(1.0, 1.0, 0.0)
        self._scene.add_ground(0.0, material=physical_material)

        loader = URDFLoader(self._scene, fix_root_link=True, load_collision=False)
        robot_no_collision = loader.load(self.urdf_path, self.srdf_path)
        robot_no_collision.set_name("shadowhand_no_collision")
        robot_no_collision.set_root_pose(Pose([0.0, 0.5, 1]))

        loader = URDFLoader(self._scene, fix_root_link=True, load_collision=True)
        robot = loader.load(self.urdf_path, self.srdf_path)
        robot.set_name("shadowhand")
        robot.set_root_pose(Pose([0, 0.5, 1]))


    def setup_robot(self, init_pose=None, init_qpos=None, disable_gravity=False, **kwargs):

        # set initial qpos
        init_qpos = np.zeros(self.robot.dof) if init_qpos is None else init_qpos
        self.robot.set_qpos(init_qpos)
        self.robot_no_collision.set_qpos(init_qpos)

        # set gravity
        if disable_gravity:
            for link in self.robot.get_links():
                link.disable_gravity = True
            for link in self.robot_no_collision.get_links():
                link.disable_gravity = True

        # set joint stiffness and damping
        stiffness, damping = kwargs.get("stiffness", 1000), kwargs.get("damping", 200)
        for joint in self.robot.get_joints():
            joint.set_drive_property(stiffness=stiffness, damping=damping)
        for joint in self.robot_no_collision.get_joints():
            joint.set_drive_property(stiffness=stiffness, damping=damping)


    def demo(self):
        while True:
            qpos = np.array(self.robot.get_qpos())
            self.robot.set_qpos(qpos)
            self.robot_no_collision.set_qpos(qpos)
            self._scene.step()
            self.render()


    def _setup_lighting(self):
        self._scene.set_ambient_light([0.4, 0.4, 0.4])
        self._scene.add_directional_light([1, -1, -1], [0.3, 0.3, 0.3])
        self._scene.add_directional_light([0, 0, -1], [1, 1, 1])

    def _setup_viewer(self):
        self._setup_lighting()
        self.viewer = self._scene.create_viewer()
        self.viewer.set_camera_xyz(x=1.5, y=0.0, z=2.0)
        self.viewer.set_camera_rpy(y=3.14, p=-0.5, r=0)


def main():
    urdf_path = str(cur_dir / "assets/shadowhand/shadowhand_simple.urdf")
    srdf_path = str(cur_dir / "assets/shadowhand/shadowhand.srdf")
    env = ShadowhandEnv(urdf_path, srdf_path)
    env.reset()
    env.demo()
    


if __name__ == "__main__":
    main()
