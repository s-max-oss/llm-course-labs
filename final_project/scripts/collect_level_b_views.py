#!/usr/bin/env python3
"""Collect a fixed grid of Gazebo RGB development views with pose truth.

This helper is intentionally an acquisition utility, not an inference component.
Every Gazebo pose reset starts a new segment and the manifest records the actual
Gazebo pose odometry observed after the reset.  It never reads scene files,
danger-target coordinates, or labels, but /Odometry_gazebo is pose ground truth.
"""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stamp_seconds(stamp: Any) -> float:
    return float(stamp.secs) + float(stamp.nsecs) / 1_000_000_000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--x", type=float, nargs="+", required=True)
    parser.add_argument("--y", type=float, nargs="+", required=True)
    parser.add_argument("--yaw-deg", type=float, nargs="+", default=[0, 90, 180, 270])
    parser.add_argument("--z", type=float, default=0.258)
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument("--message-timeout", type=float, default=5.0)
    parser.add_argument("--max-image-odom-delta-ms", type=float, default=20.0)
    parser.add_argument("--model-name", default="a1_gazebo")
    parser.add_argument("--rgb-topic", default="/real_sense/rgb/image_raw")
    parser.add_argument("--odom-topic", default="/Odometry_gazebo")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_dir}")
    if (
        args.settle_seconds < 0
        or args.message_timeout <= 0
        or args.max_image_odom_delta_ms <= 0
    ):
        raise ValueError("settle time must be non-negative and timeout must be positive")
    if not all(math.isfinite(value) for value in [*args.x, *args.y, *args.yaw_deg, args.z]):
        raise ValueError("all poses must be finite")

    import cv2
    import numpy as np
    import rospy
    from cv_bridge import CvBridge
    from gazebo_msgs.msg import ModelState
    from gazebo_msgs.srv import GetModelState, SetModelState
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Image
    from tf.transformations import quaternion_from_euler

    rospy.init_node("level_b_fixed_view_collector", anonymous=True, disable_signals=True)
    published = {name for name, _ in rospy.get_published_topics()}
    leaked = sorted(name for name in published if name.startswith("/ground_truth"))
    if leaked:
        raise RuntimeError(f"ground-truth topics are present; refusing capture: {leaked}")
    for required in (args.rgb_topic, args.odom_topic):
        if required not in published:
            raise RuntimeError(f"required topic is absent: {required}")

    rospy.wait_for_service("/gazebo/set_model_state", timeout=args.message_timeout)
    rospy.wait_for_service("/gazebo/get_model_state", timeout=args.message_timeout)
    set_model_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
    get_model_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
    bridge = CvBridge()
    args.output_dir.mkdir(parents=True)

    latest_image = [None]
    odometry_buffer = deque(maxlen=512)

    def receive_image(message: Image) -> None:
        latest_image[0] = message

    def receive_odometry(message: Odometry) -> None:
        odometry_buffer.append(message)

    image_subscriber = rospy.Subscriber(
        args.rgb_topic, Image, receive_image, queue_size=1
    )
    odometry_subscriber = rospy.Subscriber(
        args.odom_topic, Odometry, receive_odometry, queue_size=10
    )

    startup_deadline = time.monotonic() + args.message_timeout
    while latest_image[0] is None or not odometry_buffer:
        if time.monotonic() >= startup_deadline:
            raise RuntimeError("timed out waiting for initial RGB and odometry messages")
        rospy.sleep(0.01)

    records: list[dict[str, Any]] = []
    sequence_index = 0
    for x_value in args.x:
        for y_value in args.y:
            for yaw_deg in args.yaw_deg:
                quaternion = quaternion_from_euler(0.0, 0.0, math.radians(yaw_deg))
                state = ModelState()
                state.model_name = args.model_name
                state.reference_frame = "world"
                state.pose.position.x = x_value
                state.pose.position.y = y_value
                state.pose.position.z = args.z
                state.pose.orientation.x = quaternion[0]
                state.pose.orientation.y = quaternion[1]
                state.pose.orientation.z = quaternion[2]
                state.pose.orientation.w = quaternion[3]
                response = set_model_state(state)
                if not response.success:
                    raise RuntimeError(f"set_model_state failed: {response.status_message}")
                rospy.sleep(args.settle_seconds)
                minimum_image_stamp = rospy.Time.now().to_sec()
                message_deadline = time.monotonic() + args.message_timeout
                while True:
                    image_message = latest_image[0]
                    if (
                        image_message is not None
                        and stamp_seconds(image_message.header.stamp) >= minimum_image_stamp
                    ):
                        break
                    if time.monotonic() >= message_deadline:
                        raise RuntimeError("timed out waiting for a fresh post-settle RGB frame")
                    rospy.sleep(0.005)

                image_stamp = stamp_seconds(image_message.header.stamp)
                while True:
                    buffered_odometry = list(odometry_buffer)
                    if buffered_odometry and stamp_seconds(
                        buffered_odometry[-1].header.stamp
                    ) >= image_stamp:
                        break
                    if time.monotonic() >= message_deadline:
                        raise RuntimeError("timed out pairing odometry to the RGB frame")
                    rospy.sleep(0.002)
                odometry = min(
                    buffered_odometry,
                    key=lambda item: abs(stamp_seconds(item.header.stamp) - image_stamp),
                )
                actual = get_model_state(args.model_name, "world")
                if not actual.success:
                    raise RuntimeError(f"get_model_state failed: {actual.status_message}")

                image = bridge.imgmsg_to_cv2(image_message, desired_encoding="bgr8")
                if image.shape != (480, 640, 3):
                    raise RuntimeError(f"unexpected RGB shape: {image.shape}")
                view_id = f"view_{sequence_index:03d}"
                image_path = args.output_dir / f"{view_id}.png"
                if not cv2.imwrite(str(image_path), image):
                    raise RuntimeError(f"failed to write {image_path}")
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                colorful = np.logical_and(hsv[:, :, 1] > 80, hsv[:, :, 2] > 50)

                odom_stamp = stamp_seconds(odometry.header.stamp)
                image_odom_delta_ms = abs(image_stamp - odom_stamp) * 1000.0
                if image_odom_delta_ms > args.max_image_odom_delta_ms:
                    raise RuntimeError(
                        "RGB/odometry synchronization gate failed: "
                        f"{image_odom_delta_ms:.3f} ms"
                    )
                records.append(
                    {
                        "schema_version": 1,
                        "view_id": view_id,
                        "sequence_index": sequence_index,
                        "sequence_semantics": "fixed_grid_discrete_pose_segment",
                        "requested_pose": {
                            "x": x_value,
                            "y": y_value,
                            "z": args.z,
                            "yaw_deg": yaw_deg,
                        },
                        "actual_model_pose": {
                            "position": {
                                "x": actual.pose.position.x,
                                "y": actual.pose.position.y,
                                "z": actual.pose.position.z,
                            },
                            "orientation_xyzw": [
                                actual.pose.orientation.x,
                                actual.pose.orientation.y,
                                actual.pose.orientation.z,
                                actual.pose.orientation.w,
                            ],
                        },
                        "permitted_odometry": {
                            "stamp": odom_stamp,
                            "frame_id": odometry.header.frame_id,
                            "child_frame_id": odometry.child_frame_id,
                            "position": {
                                "x": odometry.pose.pose.position.x,
                                "y": odometry.pose.pose.position.y,
                                "z": odometry.pose.pose.position.z,
                            },
                            "orientation_xyzw": [
                                odometry.pose.pose.orientation.x,
                                odometry.pose.pose.orientation.y,
                                odometry.pose.pose.orientation.z,
                                odometry.pose.pose.orientation.w,
                            ],
                        },
                        "rgb": {
                            "path": image_path.name,
                            "sha256": sha256_path(image_path),
                            "stamp": image_stamp,
                            "frame_id": image_message.header.frame_id,
                            "encoding": "bgr8_export_from_rgb8",
                            "width": int(image.shape[1]),
                            "height": int(image.shape[0]),
                        },
                        "image_odom_delta_ms": image_odom_delta_ms,
                        "colorful_pixel_count": int(colorful.sum()),
                        "colorful_pixel_fraction": float(colorful.mean()),
                        "selection_uses_ground_truth": False,
                    }
                )
                sequence_index += 1

    manifest = {
        "schema_version": 1,
        "collector": "fixed_grid_truth_free_v1",
        "model_name": args.model_name,
        "rgb_topic": args.rgb_topic,
        "odom_topic": args.odom_topic,
        "ground_truth_topics_present": False,
        "discrete_pose_resets_break_tracks": True,
        "records": records,
    }
    manifest_path = args.output_dir / "scan_manifest.json"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        stream.write("\n")
    image_subscriber.unregister()
    odometry_subscriber.unregister()
    print(manifest_path)
    for record in sorted(records, key=lambda item: item["colorful_pixel_count"], reverse=True):
        pose = record["requested_pose"]
        print(
            f"{record['colorful_pixel_count']:7d} {record['view_id']} "
            f"x={pose['x']:.1f} y={pose['y']:.1f} yaw={pose['yaw_deg']:.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
