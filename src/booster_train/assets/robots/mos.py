"""MOS simulation scaffold. Gains and limits below are NOT motor specifications."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


from wbmr.paths import ROOT as PROJECT_ROOT
MOS_JOINT_NAMES = [
    "b_n", "n_h",
    "b_Rs", "Rs_Ra", "Ra_Rh", "b_Ls", "Ls_La", "La_Lh",
    "b_Rh", "Rh_Rl", "Rl_Rl1", "Rl1_Rl2", "Rl2_Ra", "Ra_Rf",
    "b_Lh", "Lh_Ll", "Ll_Ll1", "Ll1_Ll2", "Ll2_La", "La_Lf",
]
MOS_BODY_NAMES = [
    "body", "neck", "head",
    "Rshoulder", "Rarm", "Rhand", "Lshoulder", "Larm", "Lhand",
    "Rhip", "Rlap", "Rleg1", "Rleg2", "Rankle", "Rfoot",
    "Lhip", "Llap", "Lleg1", "Lleg2", "Lankle", "Lfoot",
]
MOS_FOOT_NAMES = ("Lfoot", "Rfoot")

MOS_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(PROJECT_ROOT / "robots/mos/training/mos_training.urdf"),
        fix_base=False,
        root_link_name="body",
        merge_fixed_joints=False,
        collider_type="convex_hull",
        self_collision=False,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False, linear_damping=0.0, angular_damping=0.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8, solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65), joint_pos={".*": 0.0}, joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=MOS_JOINT_NAMES[8:], effort_limit_sim=60.0,
            velocity_limit_sim=20.0, stiffness=100.0, damping=5.0,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=MOS_JOINT_NAMES[2:8], effort_limit_sim=20.0,
            velocity_limit_sim=20.0, stiffness=40.0, damping=2.0,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=MOS_JOINT_NAMES[:2], effort_limit_sim=5.0,
            velocity_limit_sim=20.0, stiffness=10.0, damping=0.5,
        ),
    },
)
MOS_ACTION_SCALE = {
    name: 0.25 * actuator.effort_limit_sim / actuator.stiffness
    for actuator in MOS_CFG.actuators.values() for name in actuator.joint_names_expr
}
