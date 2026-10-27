import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_xlerobot_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(12),
        "observation/images/head": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/images/left_wrist": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/images/right_wrist": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
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
class XlerobotInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType
    adapt_to_aloha: bool = False

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
            "state": _decode_state(
                np.asarray(data["observation/state"]), adapt_to_aloha=self.adapt_to_aloha
            ),
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
        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = _decode_action(
                np.asarray(data["actions"]), adapt_to_aloha=self.adapt_to_aloha
            )

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

    adapt_to_aloha: bool = False

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        action = np.asarray(data["actions"][..., :14])
        return {"actions": _encode_action(action, adapt_to_aloha=self.adapt_to_aloha)}


def _decode_state(state: np.ndarray, adapt_to_aloha: bool = False) -> np.ndarray:
    if not adapt_to_aloha:
        # padding to 2×(6+1) dof
        state = np.insert(state, [3, 9], 0)
        # degree to radian
        mask = np.r_[0:6, 7:13]
        state[mask] = np.deg2rad(state[mask])
        # reverse gripper
        mask = np.array([6, 13])
        state[mask] = 1 - state[mask] / 100
    else:
        # padding to 2×(6+1) dof
        state = np.insert(state, [3, 9], 0)
        # degree to radian
        mask = np.r_[0:6, 7:13]
        state[mask] = np.deg2rad(state[mask])
        # flip joints
        mask = np.array([0, 1, 2, 5, 7, 8, 9, 12])
        state[mask] = -state[mask]
        
    return state


def _decode_action(action: np.ndarray, adapt_to_aloha: bool = False) -> np.ndarray:
    if not adapt_to_aloha:
        # padding to 2×(6+1) dof
        action = np.insert(action, [3, 9], 0, axis=1)
        # degree to radian
        mask = np.r_[0:6, 7:13]
        action[:, mask] = np.deg2rad(action[:, mask])
        # reverse gripper
        mask = np.array([6, 13])
        action[:, mask] = 1 - action[:, mask] / 100
    else:
        # padding to 2×(6+1) dof
        action = np.insert(action, [3, 9], 0, axis=1)
        # degree to radian
        mask = np.r_[0:6, 7:13]
        action[:, mask] = np.deg2rad(action[:, mask])
        # flip joints
        mask = np.array([0, 1, 2, 5, 7, 8, 9, 12])
        action[:, mask] = -action[:, mask]
        
    return action


def _encode_action(action: np.ndarray, adapt_to_aloha: bool = False) -> np.ndarray:
    if not adapt_to_aloha:
        # recover gripper
        mask = np.array([6, 13])
        action[:, mask] = 100 - 100 * action[:, mask]
        # radian to degree
        mask = np.r_[0:6, 7:13]
        action[:, mask] = np.rad2deg(action[:, mask])
        # recover to 2×(5+1) dof
        action = np.delete(action, [3, 10], axis=1)
    else:
        # recover flip joints
        mask = np.array([0, 1, 2, 5, 7, 8, 9, 12])
        action[:, mask] = -action[:, mask]
        # radian to degree
        mask = np.r_[0:6, 7:13]
        action[:, mask] = np.rad2deg(action[:, mask])
        # recover to 2×(5+1) dof
        action = np.delete(action, [3, 10], axis=1)

    return action
