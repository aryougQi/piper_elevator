#!/usr/bin/env python3
"""Audit saved hand-eye samples offline; never initialize ROS or touch hardware."""

import argparse
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def pose_error(reference, actual):
    return {
        "translation_mm": float(np.linalg.norm(actual[:3, 3] - reference[:3, 3]) * 1000),
        "rotation_deg": float(Rotation.from_matrix(reference[:3, :3].T @ actual[:3, :3]).magnitude() * 180 / np.pi),
    }


def distribution(values):
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def main():
    script_path = Path(__file__).resolve()
    workspace = script_path.parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-root", type=Path, default=workspace / "handeye_calibration")
    parser.add_argument("--output", type=Path, default=script_path.parents[1] / "data" / "handeye_sample_integrity_20260908.json")
    args = parser.parse_args()
    root = args.calibration_root.resolve()
    solver_path = root / "scripts" / "solve_handeye.py"
    spec = importlib.util.spec_from_file_location("solve_handeye_offline", solver_path)
    solver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solver)
    urdf_path = root / "calibration" / "piper_handeye_input.urdf"
    kinematics = solver.UrdfKinematics(urdf_path)
    paths = sorted((root / "calibration" / "samples_20260902").glob("sample_*.json"))
    samples = [json.loads(path.read_text()) for path in paths]
    by_id = {sample["id"]: sample for sample in samples}
    stamp_ns = lambda sample: sample["image_stamp"]["sec"] * 1_000_000_000 + sample["image_stamp"]["nanosec"]
    ordered = sorted(samples, key=stamp_ns)
    order_ids = [sample["id"] for sample in ordered]

    def fk(values):
        return kinematics.forward("base_link", "tcp_link", {f"joint{i+1}": value for i, value in enumerate(values)})

    poses = {sample["id"]: fk(sample["joints"]) for sample in samples}
    results = []
    for index, sample in enumerate(ordered):
        sid = sample["id"]
        stored = sample["base_to_tcp"]
        saved_pose = solver.transform(stored["translation"], Rotation.from_quat(stored["quaternion_xyzw"]).as_matrix())
        errors = {other_id: pose_error(poses[other_id], saved_pose) for other_id in order_ids}
        joint_error = np.asarray(sample["joints"]) - np.asarray(sample["target"])
        previous = order_ids[index - 1] if index else None
        following = order_ids[index + 1] if index + 1 < len(ordered) else None
        closest_translation = min(errors, key=lambda value: errors[value]["translation_mm"])
        closest_rotation = min(errors, key=lambda value: errors[value]["rotation_deg"])
        results.append({
            "id": sid,
            "image_stamp": sample["image_stamp"],
            "image_time_utc": datetime.fromtimestamp(stamp_ns(sample) / 1e9, timezone.utc).isoformat(),
            "seconds_from_first_image": (stamp_ns(sample) - stamp_ns(ordered[0])) / 1e9,
            "seconds_from_previous_image": None if previous is None else (stamp_ns(sample) - stamp_ns(by_id[previous])) / 1e9,
            "target_rad": sample["target"],
            "joints_rad": sample["joints"],
            "joints_minus_target_rad": joint_error.tolist(),
            "max_abs_joint_error_rad": float(np.max(np.abs(joint_error))),
            "max_abs_joint_error_deg": float(np.max(np.abs(joint_error)) * 180 / np.pi),
            "target_fk_vs_joints_fk": pose_error(fk(sample["target"]), poses[sid]),
            "saved_tf_vs_own_fk": errors[sid],
            "previous_by_image_stamp": None if previous is None else {"id": previous, **errors[previous]},
            "next_by_image_stamp": None if following is None else {"id": following, **errors[following]},
            "nearest_translation_fk": {"id": closest_translation, **errors[closest_translation]},
            "nearest_rotation_fk": {"id": closest_rotation, **errors[closest_rotation]},
            "saved_tf_vs_all_sample_fk": errors,
            "corner_count": len(sample["corners"]),
            "paired_image_exists": (root / "calibration" / "samples_20260902" / f"sample_{sid}.png").exists(),
        })

    def summarize(rows):
        return {
            "count": len(rows),
            "max_abs_joint_error_rad": distribution([row["max_abs_joint_error_rad"] for row in rows]),
            "max_abs_joint_error_deg": distribution([row["max_abs_joint_error_deg"] for row in rows]),
            "target_fk_vs_joints_fk_translation_mm": distribution([row["target_fk_vs_joints_fk"]["translation_mm"] for row in rows]),
            "target_fk_vs_joints_fk_rotation_deg": distribution([row["target_fk_vs_joints_fk"]["rotation_deg"] for row in rows]),
            "saved_tf_vs_own_fk_translation_mm": distribution([row["saved_tf_vs_own_fk"]["translation_mm"] for row in rows]),
            "saved_tf_vs_own_fk_rotation_deg": distribution([row["saved_tf_vs_own_fk"]["rotation_deg"] for row in rows]),
        }

    previous_close = [row["id"] for row in results if row["previous_by_image_stamp"] is not None
                      and row["previous_by_image_stamp"]["translation_mm"] <= 2
                      and row["previous_by_image_stamp"]["rotation_deg"] <= 0.5]
    previous_closer = [row["id"] for row in results if row["previous_by_image_stamp"] is not None
                       and row["previous_by_image_stamp"]["translation_mm"] < row["saved_tf_vs_own_fk"]["translation_mm"]
                       and row["previous_by_image_stamp"]["rotation_deg"] < row["saved_tf_vs_own_fk"]["rotation_deg"]]
    report = {
        "method": "Read JSON and run the supplied URDF FK only. No optimization, ROS, or hardware access.",
        "sample_directory": str(paths[0].parent),
        "urdf": str(urdf_path),
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in [solver_path, urdf_path, *paths]},
        "sample_ids_by_image_stamp": order_ids,
        "image_time_span_seconds": (stamp_ns(ordered[-1]) - stamp_ns(ordered[0])) / 1e9,
        "all_samples": summarize(results),
        "production_fit_samples_excluding_07": summarize([row for row in results if row["id"] != "07"]),
        "per_joint_abs_error_max_rad": np.max(np.abs([row["joints_minus_target_rad"] for row in results]), axis=0).tolist(),
        "per_joint_abs_error_median_rad": np.median(np.abs([row["joints_minus_target_rad"] for row in results]), axis=0).tolist(),
        "all_within_move_script_default_0_006_rad_tolerance": all(row["max_abs_joint_error_rad"] <= 0.006 for row in results),
        "all_have_88_corners_and_paired_image": all(row["corner_count"] == 88 and row["paired_image_exists"] for row in results),
        "camera_metadata_identical": all(sample["camera"] == samples[0]["camera"] for sample in samples),
        "sample_keys": sorted(set(key for sample in samples for key in sample)),
        "previous_image_sample_fk_closer_than_own_in_both_metrics_ids": previous_closer,
        "previous_image_sample_fk_within_2mm_0_5deg_ids": previous_close,
        "nearest_saved_tf_to_any_fk_translation_mm": distribution([row["nearest_translation_fk"]["translation_mm"] for row in results]),
        "nearest_saved_tf_to_any_fk_rotation_deg": distribution([row["nearest_rotation_fk"]["rotation_deg"] for row in results]),
        "timing_evidence": {
            "known": [
                "JSON persists image_stamp but no joint-state stamp or saved TF stamp.",
                "read_joint_positions returns positions only; it discards the JointState header.",
                "move_and_check saves the last converged joint sample, then waits 0.5s before requesting an image.",
                "TF lookup uses Time() (latest available), after image and CameraInfo acquisition; no image-time lookup/interpolation occurs.",
                "TransformListener uses spin_thread=False; time.sleep and OpenCV work do not explicitly spin the node.",
                "The solver uses FK of persisted joints for the fit; saved TF is a diagnostic comparison only.",
            ],
            "not_proven": [
                "No actual joint/image/TF time offset can be recovered because joint and TF header stamps are absent.",
                "The nearest other sample pose is a geometric match, not proof of its exact TF timestamp.",
                "Unrecorded intermediate motions, earlier overwritten captures, and the publisher's buffering behavior cannot be reconstructed from these JSON files.",
                "The 0.5-second wall wait is known from code; exact sensor timestamp separation from the joint observation is unknown.",
            ],
        },
        "checkerboard": {
            "inner_corners": [8, 11],
            "square_size_m": 0.019,
            "inner_corner_extent_m": [0.133, 0.19],
            "inner_corner_half_extent_m": [0.0665, 0.095],
            "configuration_consistency": "capture.yaml, README, move_and_check detection and solve_handeye agree. calibrate.yaml comments use matching half spans but do not define corner count or square size.",
            "physical_scale_limitation": "19mm is a consistent assumed/configured size; no measurement record in these sample JSON files independently verifies physical square size or board flatness.",
        },
        "samples_in_image_time_order": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("source_sha256", "samples_in_image_time_order")}, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
