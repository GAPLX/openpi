"""Serves an RTC-enabled pi0 / pi05 XLerobot policy for the bridge `ChunkClient`."""

import dataclasses
import logging
import socket

from bridge.runtime.server import ChunkServer
import tyro

from openpi import transforms as _transforms
from openpi.models import rtc as _rtc
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.policies import xlerobot_policy
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    """Arguments for the serve_xlerobot script."""

    # Training config name, e.g. "pi0_xlerobot" or "pi05_xlerobot".
    config: str
    # Checkpoint directory, e.g. "checkpoints/pi05_xlerobot/exp/20000".
    dir: str

    # Used if the "prompt" key is not present in the observation.
    default_prompt: str | None = None

    host: str = "0.0.0.0"
    # Must match the port of the bridge client.
    port: int = 8445
    # Record the policy's behavior for debugging.
    record: bool = False
    # Log the inference latency of every request.
    print_mode: bool = False

    rtc_config: _rtc.RTCConfig = dataclasses.field(default_factory=_rtc.RTCConfig)


def create_policy(args: Args) -> _policy.Policy:
    return _policy_config.create_trained_policy(
        _config.get_config(args.config),
        args.dir,
        default_prompt=args.default_prompt,
        rtc_config=args.rtc_config,
        repack_transforms=_transforms.Group(inputs=[xlerobot_policy.XlerobotRawInputs()]),
    )


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = ChunkServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy_metadata,
        print_mode=args.print_mode,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
