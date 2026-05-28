"""CALVIN benchmark data config, embodiment tags, and mixtures.

CALVIN demonstrations are expected in LeRobot format.  The dataset stores
normalized 7-D continuous actions in the layout used by the existing eval
client:

    dx, dy, dz, droll, dpitch, dyaw, gripper

Camera order is static RGB first, wrist/gripper RGB second.  Keep this order in
sync with ``examples/calvin/train_files/modality.json`` and eval preprocessing.
"""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)


class CalvinFrankaDataConfig:
    """Data config for CALVIN task_D_D LeRobot demonstrations."""

    embodiment_tag = EmbodimentTag.FRANKA

    video_keys = [
        "video.primary_image",
        "video.wrist_image",
    ]

    # Current CALVIN modality.json contains an extra state.pad channel.  This
    # config intentionally omits it so optional proprio input is 7-D and matches
    # framework.action_model.state_dim when include_state is enabled.
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.roll",
        "state.pitch",
        "state.yaw",
        "state.gripper",
    ]

    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]

    language_keys = ["annotation.human.action.task_description"]

    observation_indices = [0]
    action_indices = list(range(8))
    state_indices = [0]

    def modality_config(self):
        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=self.state_indices,
                modality_keys=self.state_keys,
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices,
                modality_keys=self.action_keys,
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.language_keys,
            ),
        }

    def transform(self):
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes={
                        "state.x": "min_max",
                        "state.y": "min_max",
                        "state.z": "min_max",
                        "state.roll": "min_max",
                        "state.pitch": "min_max",
                        "state.yaw": "min_max",
                        "state.gripper": "binary",
                    },
                ),
                StateActionToTensor(apply_to=self.action_keys),
            ]
        )


ROBOT_TYPE_CONFIG_MAP = {
    "calvin_franka": CalvinFrankaDataConfig(),
}


ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    # Kept for registry compatibility; embodiment_tag lives on the DataConfig.
}


DATASET_NAMED_MIXTURES = {
    # Matches examples/calvin/train_files/starvla_train_calvin.yaml.
    # Expected path with default data_root_dir:
    #   playground/Datasets/calvin/task_D_D
    "calvin_task_D_D": [
        ("task_D_D", 1.0, "calvin_franka"),
    ],
    # Compatibility with the pre-existing central mixture name.
    # Expected path:
    #   playground/Datasets/calvin/calvin_task_D_D_v3.0
    "calvin_task_D_D_v3.0": [
        ("calvin_task_D_D_v3.0", 1.0, "calvin_franka"),
    ],
}
