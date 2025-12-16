#!/usr/bin/env python3
import argparse
import os
import re
import shutil
import sys
from pathlib import Path

QUOTED_PATH_RE = re.compile(r'"([^"]+)"')

def die(msg: str, code: int = 2) -> None:
    print(f"cp_tf: error: {msg}", file=sys.stderr)
    sys.exit(code)

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def write_text(path: Path, s: str) -> None:
    path.write_text(s, encoding="utf-8")

def fix_checkpoint_file(dst_dir: Path) -> None:
    """
    Adjust quoted paths in dst_dir/checkpoint so that any absolute checkpoint paths
    now point to dst_dir/<checkpoint_basename>.
    Example:
      model_checkpoint_path: "/old/.../pnn_X/199"
    becomes
      model_checkpoint_path: "/new/.../pnn_X/199"
    """
    ckpt_file = dst_dir / "checkpoint"
    if not ckpt_file.exists():
        die(f"destination directory has no 'checkpoint' file: {ckpt_file}")

    content = read_text(ckpt_file)

    def repl(m: re.Match) -> str:
        old_path = m.group(1)
        # Keep relative paths untouched (they already work after moving).
        if not os.path.isabs(old_path):
            return f"\"{old_path}\""
        ckpt_name = os.path.basename(old_path.rstrip("/"))
        new_path = str((dst_dir / ckpt_name).resolve())
        return f"\"{new_path}\""

    new_content = QUOTED_PATH_RE.sub(repl, content)
    write_text(ckpt_file, new_content)

def copy_dir(src: Path, dst: Path) -> None:
    # Python >=3.8: shutil.copytree supports dirs_exist_ok, but we prefer to fail loudly.
    if dst.exists():
        die(f"destination already exists: {dst}")
    shutil.copytree(src, dst, symlinks=True, copy_function=shutil.copy2)

def main() -> None:
    p = argparse.ArgumentParser(
        prog="cp_tf",
        description="Copy (or move) a TensorFlow checkpoint directory and fix absolute paths in 'checkpoint'.",
    )
    p.add_argument("src_dir", help="Source TF checkpoint directory (contains 'checkpoint', *.index, *.data-*)")
    p.add_argument("dest_parent", help="Destination parent directory (the source basename will be created inside)")
    p.add_argument("--mv", action="store_true", help="Remove the original directory after successful copy")
    args = p.parse_args()

    src_dir = Path(args.src_dir).expanduser().resolve()
    if not src_dir.is_dir():
        die(f"source is not a directory: {src_dir}")

    if not (src_dir / "checkpoint").exists():
        die(f"source directory has no 'checkpoint' file: {src_dir / 'checkpoint'}")

    dest_parent = Path(args.dest_parent).expanduser().resolve()
    if dest_parent.exists() and not dest_parent.is_dir():
        die(f"dest_parent exists but is not a directory: {dest_parent}")
    dest_parent.mkdir(parents=True, exist_ok=True)

    dst_dir = dest_parent / src_dir.name

    # Copy then fix checkpoint in the destination.
    copy_dir(src_dir, dst_dir)
    fix_checkpoint_file(dst_dir)

    # Optionally remove original.
    if args.mv:
        shutil.rmtree(src_dir)

    print(str(dst_dir))

if __name__ == "__main__":
    main()

