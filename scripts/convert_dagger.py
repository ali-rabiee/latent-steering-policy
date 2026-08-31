"""A4: DAgger episode dirs -> a zarr in boxes_v0's schema, optionally merged
with an existing zarr.

Each episode dir holds episode.npz (state, action, meta) written by
rollout_sim --dagger-out, plus images/<cam>/image_NNNNNN.png captured at every
tick. The action is the STATELESS EXPERT's, evaluated at the state the POLICY
reached -- so unlike a demonstration set, action[t] is deliberately NOT
pose[t+1] - pose[t]. That divergence is what DAgger is.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from lsteer.data import schema  # noqa: E402


def load_episode(d, cams):
    z = np.load(os.path.join(d, "episode.npz"), allow_pickle=True)
    state, action = np.asarray(z["state"]), np.asarray(z["action"])
    n = len(state)
    imgs = {}
    for cam in cams:
        files = sorted(glob.glob(os.path.join(d, "images", cam, "*.png")))
        if len(files) < n:
            return None  # truncated episode (wall clock); drop it whole
        arr = np.zeros((n, schema.IMG_STORE_SIZE, schema.IMG_STORE_SIZE, 3), dtype=np.uint8)
        for i in range(n):
            im = Image.open(files[i]).convert("RGB")
            if im.size != (schema.IMG_STORE_SIZE, schema.IMG_STORE_SIZE):
                im = im.resize((schema.IMG_STORE_SIZE, schema.IMG_STORE_SIZE), Image.BILINEAR)
            arr[i] = np.asarray(im, dtype=np.uint8)
        imgs[cam] = arr
    return dict(state=state.astype(np.float32), action=action.astype(np.float32), imgs=imgs,
                target_pos=np.asarray(z["target_pos_b"], dtype=np.float32),
                target_color=int(z["target_color"]),
                box_positions=np.asarray(z["box_positions"], dtype=np.float32),
                box_colors=np.asarray(z["box_colors"], dtype=np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dagger-dirs", nargs="+", required=True)
    ap.add_argument("--merge-zarr", default="", help="existing zarr to prepend (e.g. boxes_v0)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-episodes", type=int, default=0)
    ap.add_argument("--grip-supervise-below", type=float, default=0.0, metavar="METRES",
                    help="A4c: supervise the gripper only on DAgger frames whose EE height is below this, "
                         "masking it above. Overrides --mask-dagger-gripper.")
    ap.add_argument("--mask-dagger-gripper", action="store_true",
                    help="A4b: zero the gripper loss on DAgger frames, keeping their position supervision")
    args = ap.parse_args()
    import zarr

    cams = list(schema.CAMERA_NAMES)
    eps = []
    for root in args.dagger_dirs:
        for d in sorted(glob.glob(os.path.join(root, "episode_*"))):
            e = load_episode(d, cams)
            if e is None:
                print(f"  DROP (truncated): {d}")
                continue
            eps.append(e)
            if args.max_episodes and len(eps) >= args.max_episodes:
                break
    print(f"loaded {len(eps)} DAgger episodes, {sum(len(e['state']) for e in eps)} frames")

    states, actions, ends, gmask = [], [], [], []
    tpos, tcol, bpos, bcol, succ = [], [], [], [], []
    img_stacks = {c: [] for c in cams}
    total = 0

    if args.merge_zarr:
        src = zarr.open(args.merge_zarr, mode="r")
        s0 = np.asarray(src[schema.DATA_STATE]); a0 = np.asarray(src[schema.DATA_ACTION])
        e0 = np.asarray(src[schema.META_EPISODE_ENDS])
        states.append(s0); actions.append(a0)
        gmask.append(np.ones(len(s0), dtype=np.float32))  # demos supervise the gripper
        ends.extend(e0.tolist()); total = int(e0[-1])
        for c in cams:
            img_stacks[c].append(np.asarray(src[schema.camera_img_key(c)]))
        tpos.append(np.asarray(src[schema.META_TARGET_POS]))
        tcol.append(np.asarray(src[schema.META_TARGET_COLOR]))
        bpos.append(np.asarray(src[schema.META_BOX_POSITIONS]))
        bcol.append(np.asarray(src[schema.META_BOX_COLORS]))
        succ.append(np.asarray(src[schema.META_EPISODE_SUCCESS]))
        print(f"merged base {args.merge_zarr}: {len(e0)} episodes, {total} frames")

    for e in eps:
        states.append(e["state"]); actions.append(e["action"])
        # A4b: DAgger frames supervise POSITION only. Their gripper labels are
        # sparse (~5% closed against the demos' 41%) and sit right at the close
        # threshold, so including them collapsed the policy to never closing:
        # A4 scored 3.75% with a best-in-project 1.50 mm reach and a max finger
        # closure of 0.005. Where to go and when to close are learned from
        # different sources.
        if args.grip_supervise_below > 0.0:
            # A4c: supervise the gripper on the DAgger frames NEAR THE GRASP and
            # mask it elsewhere. Measured on dagger_v3: the close rate among
            # DAgger frames is 4.8% overall but 9.3% below 0.12 m and 12.8% below
            # 0.06 m, because almost every high frame is trivially "open". The
            # near-grasp band is where the label carries information -- and it
            # covers 0.093-0.102 m, exactly where A4b's FAILURES wrongly close and
            # where the expert says "still open, descend". A4b's successes close
            # at 0.071-0.075 m against the demos' 0.047, so the residual really is
            # timing, and this is the only signal that addresses it in the states
            # the policy actually reaches.
            gm = (np.asarray(e["state"])[:, 2] < args.grip_supervise_below).astype(np.float32)
        else:
            gm = np.full(len(e["state"]), 0.0 if args.mask_dagger_gripper else 1.0, dtype=np.float32)
        gmask.append(gm)
        total += len(e["state"]); ends.append(total)
        for c in cams:
            img_stacks[c].append(e["imgs"][c])
        tpos.append(e["target_pos"][None]); tcol.append(np.array([e["target_color"]], dtype=np.int64))
        bpos.append(e["box_positions"][None]); bcol.append(e["box_colors"][None])
        succ.append(np.array([1], dtype=np.int64))

    root = zarr.open(args.out, mode="w")
    root.create_dataset(schema.DATA_STATE, data=np.concatenate(states), dtype="f4")
    root.create_dataset(schema.DATA_ACTION, data=np.concatenate(actions), dtype="f4")
    for c in cams:
        stack = np.concatenate(img_stacks[c])
        root.create_dataset(schema.camera_img_key(c), data=stack, dtype="u1",
                            chunks=(1,) + stack.shape[1:])
    root.create_dataset(schema.META_EPISODE_ENDS, data=np.asarray(ends, dtype=np.int64), dtype="i8")
    root.create_dataset(schema.META_TARGET_POS, data=np.concatenate(tpos), dtype="f4")
    root.create_dataset(schema.META_TARGET_COLOR, data=np.concatenate(tcol), dtype="i8")
    root.create_dataset(schema.META_BOX_POSITIONS, data=np.concatenate(bpos), dtype="f4")
    root.create_dataset(schema.META_BOX_COLORS, data=np.concatenate(bcol), dtype="i8")
    root.create_dataset(schema.META_EPISODE_SUCCESS, data=np.concatenate(succ), dtype="i8")
    root.create_dataset(schema.DATA_GRIP_MASK, data=np.concatenate(gmask), dtype="f4")
    root.attrs["camera_names"] = cams
    root.attrs["dagger_episodes"] = len(eps)
    print(f"WROTE {args.out}: {len(ends)} episodes, {total} frames "
          f"({len(eps)} of them DAgger = {sum(len(e['state']) for e in eps)} frames)")


if __name__ == "__main__":
    main()
