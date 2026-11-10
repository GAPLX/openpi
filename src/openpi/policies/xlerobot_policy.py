import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


# Must match the joint order used by the bridge XLerobot environment and dataset recorder.
JOINTS = (
    "right_arm_shoulder_pan.pos",
    "right_arm_shoulder_lift.pos",
    "right_arm_elbow_flex.pos",
    "right_arm_wrist_flex.pos",
    "right_arm_wrist_roll.pos",
    "right_arm_gripper.pos",
    "left_arm_shoulder_pan.pos",
    "left_arm_shoulder_lift.pos",
    "left_arm_elbow_flex.pos",
    "left_arm_wrist_flex.pos",
    "left_arm_wrist_roll.pos",
    "left_arm_gripper.pos",
)
CAMERAS = ("head", "left_wrist", "right_wrist")
ACTION_DIM = len(JOINTS)


def make_xlerobot_example() -> dict:
    """Creates a random raw XLerobot observation, as sent by the bridge `ChunkClient`."""
    return {
        **{joint: np.float32(np.random.uniform(-90, 90)) for joint in JOINTS},
        **{
            cam: np.random.randint(256, size=(480, 640, 3), dtype=np.uint8)
            for cam in CAMERAS
        },
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class XlerobotRawInputsRepack(transforms.DataTransformFn):
    """Converts a raw XLerobot observation recieved from ChunkClient into the format produced by the training repack transform.

    Inference only. Pass it as `repack_transforms` to `create_trained_policy`. `actions` (the RTC `prev_actions`)
    and `prompt` are kept as is.
    """

    def __call__(self, data: dict) -> dict:
        inputs = {
            "observation/state": np.asarray(
                [data[joint] for joint in JOINTS], dtype=np.float32
            ),
            **{f"observation/images/{cam}": data[cam] for cam in CAMERAS},
        }
        for key in ("actions", "prompt"):
            if key in data:
                inputs[key] = data[key]
        return inputs


@dataclasses.dataclass(frozen=True)
class XlerobotInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right).
        base_image = _parse_image(data["observation/images/head"])
        left_wrist_image = _parse_image(data["observation/images/left_wrist"])
        right_wrist_image = _parse_image(data["observation/images/right_wrist"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": _decode_state(np.asarray(data["observation/state"])),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are available during training and, as RTC `prev_actions`, during inference.
        if "actions" in data:
            inputs["actions"] = _decode_action(np.asarray(data["actions"]))

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class XlerobotOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        action = np.asarray(data["actions"][..., :14])
        return {"actions": _encode_action(action)}


def _decode_state(state: np.ndarray) -> np.ndarray:
    # padding to 2×(6+1) dof
    state = np.insert(state, [3, 9], 0)
    # degree to radian
    mask = np.r_[0:6, 7:13]
    state[mask] = np.deg2rad(state[mask])
    # reverse gripper
    mask = np.array([6, 13])
    state[mask] = 1 - state[mask] / 100

    return state


def _decode_action(action: np.ndarray) -> np.ndarray:
    # padding to 2×(6+1) dof
    action = np.insert(action, [3, 9], 0, axis=1)
    # degree to radian
    mask = np.r_[0:6, 7:13]
    action[:, mask] = np.deg2rad(action[:, mask])
    # reverse gripper
    mask = np.array([6, 13])
    action[:, mask] = 1 - action[:, mask] / 100

    return action


def _encode_action(action: np.ndarray) -> np.ndarray:
    # recover gripper
    mask = np.array([6, 13])
    action[:, mask] = 100 - 100 * action[:, mask]
    # radian to degree
    mask = np.r_[0:6, 7:13]
    action[:, mask] = np.rad2deg(action[:, mask])
    # recover to 2×(5+1) dof
    action = np.delete(action, [3, 10], axis=1)

    return action
