from __future__ import annotations
import argparse
import shutil
from pathlib import Path
import sys


def copy_tree(src: Path, dst: Path, clean: bool = False):
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")
    if clean and dst.exists():
        # remove dst contents but keep the folder itself if it's the svn working copy root
        for p in dst.iterdir():
            if p.name == '.svn':
                continue
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
    dst.mkdir(parents=True, exist_ok=True)

    if src.is_dir():
        for item in src.rglob('*'):
            rel = item.relative_to(src)
            target = dst / rel
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)
    else:
        # src is a file
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(description="Package current code into the SVN working copy at 'wangziwen', mirroring the 'test' folder layout (Interface.py + Parameters/ at repo root).")
    parser.add_argument('--dest', type=str, default=str(Path('wangziwen').resolve()), help="Destination SVN working copy root (default: ./wangziwen)")
    parser.add_argument('--src_interface', type=str, default=str(Path('test/Interface.py').resolve()), help="Source Interface.py (default: test/Interface.py)")
    parser.add_argument('--src_parameters', type=str, default=str(Path('test/Parameters').resolve()), help="Source Parameters directory (default: test/Parameters)")
    parser.add_argument('--clean', action='store_true', help="Clean destination (except .svn) before copying")
    args = parser.parse_args()

    dest_root = Path(args.dest).resolve()
    src_interface = Path(args.src_interface).resolve()
    src_params = Path(args.src_parameters).resolve()

    # Basic checks
    if not dest_root.exists():
        print(f"Destination does not exist, creating: {dest_root}")
        dest_root.mkdir(parents=True, exist_ok=True)
    if not (dest_root / '.svn').exists():
        print("Warning: destination does not appear to be an SVN working copy ('.svn' missing). Proceeding anyway.")

    if not src_interface.exists():
        print(f"ERROR: Interface.py not found at {src_interface}")
        sys.exit(1)
    if not src_params.exists():
        print(f"ERROR: Parameters source not found at {src_params}")
        sys.exit(1)

    # Clean destination root if requested (preserve .svn)
    if args.clean:
        for p in dest_root.iterdir():
            if p.name == '.svn':
                continue
            try:
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
            except Exception as e:
                print(f"Skip removing {p}: {e}")

    # Copy Interface.py to dest root
    target_interface = dest_root / 'Interface.py'
    target_interface.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_interface, target_interface)

    # Copy Parameters folder (entire tree)
    target_params_root = dest_root / 'Parameters'
    if target_params_root.exists():
        shutil.rmtree(target_params_root)
    copy_tree(src_params, target_params_root, clean=False)

    print('\nPackaged to:')
    print(f"  {dest_root}")
    print('Included:')
    print(f"  - Interface.py <- {src_interface}")
    print(f"  - Parameters/  <- {src_params}")
    print('\nYou can now `svn status` and `svn commit` inside the destination working copy.')


if __name__ == '__main__':
    main()
