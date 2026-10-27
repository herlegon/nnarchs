from argparse import (
    ArgumentParser,
    RawTextHelpFormatter
)
from pathlib import Path
from pprint import pprint
from hytils import (
    lightcyan,
    lightgreen,
    red,
)
import logging
import os
import signal
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# logging.config.fileConfig('config.ini')
# logger = logging.getLogger("nnarchs")
# logging.basicConfig(filename="logs.log", filemode="w", format="%(name)s → %(levelname)s: %(message)s")
# logging.config.fileConfig('config.ini')
start_time = time.time()

from nnarchs import (
    nnlogger,
    nnlib,
    NnModel,
    is_tensorrt_available,
    get_supported_model_extensions,
    NnFrameworkType,
    print_metadata,
)
print(f"nnarchs loaded in {(time.time() - start_time):.02f}s")


def main():
    parser = ArgumentParser(
        description="Walk through a directory and parse models",
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="",
        required=False,
        help="Path to a specific model file."
    )

    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default="",
        required=False,
        help="Directory containing model files."
    )

    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        required=False,
        help="Recursively search through subdirectories."
    )

    parser.add_argument(
        "-f",
        "--filter",
        choices=[
            *get_supported_model_extensions(),
            'pytorch',
            'onnx',
            'trt',
            'trtzip',
        ],
        default="",
        required=False,
        help="Filter models by file extension or framework (e.g., pytorch, onnx, trtzip)."
    )

    parser.add_argument(
        "--metadata",
        action="store_true",
        required=False,
        help="Display model metadata."
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        help="Display detailed model information."
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        required=False,
        help="Enable debug mode for additional logging."
    )

    parser.add_argument(
        "--dev",
        action="store_true",
        required=False,
        help="Development."
    )

    args = parser.parse_args()

    debug: bool = args.debug
    if debug:
        # FileOutputHandler = logging.FileHandler('logs.log', mode='w')
        # nnlogger.addHandler(FileOutputHandler)
        nnlogger.addHandler(logging.StreamHandler(sys.stdout))
        nnlogger.setLevel("DEBUG")


    filtered_exts: tuple[str] = get_supported_model_extensions()
    if args.filter != '':
        if args.filter == 'onnx':
            filtered_exts = get_supported_model_extensions(NnFrameworkType.ONNX)
        elif args.filter == 'pytorch':
            filtered_exts = get_supported_model_extensions(NnFrameworkType.PYTORCH)
        elif args.filter in  ('trt', 'trtzip'):
            filtered_exts = get_supported_model_extensions(NnFrameworkType.TENSORRT)
        else:
            filtered_exts = (args.filter)
    trt_extensions: tuple[int] = get_supported_model_extensions(NnFrameworkType.TENSORRT)

    # Parse a single model
    model_fp: Path = Path(args.model).resolve()
    if model_fp.is_file():
        ext = model_fp.suffix
        if ext not in filtered_exts:
            print(f"[E] Unsupported model: {args.model} cannot be loaded")
        if ext in trt_extensions and not is_tensorrt_available():
            print(f"[E] Unsupported hardware: {args.model} cannot be loaded")
        device = 'cuda' if ext in trt_extensions else 'cpu'

        try:
            start_time= time.time()
            model: NnModel = nnlib.open(model_fp, device)
            elapsed = time.time() - start_time
            print(
                f"\tarch:", lightcyan(model.arch_name),
                f"scale:", lightcyan(model.scale),
                f"\t\t({1000 * elapsed:.1f}ms)"
            )
            # Display metadata
            if args.metadata:
                print_metadata(model.metadata)


            if args.verbose:
                print("Model:")
                print(model)
                print("\nArchitecture:")
                print(model.arch)
        except Exception as e:
            # For debug:
            # model: NnModel = nnlib.open(model_fp, device)
            print(red(e))
        return

    # Parse a directory
    ml_models_path: Path = Path(args.directory).resolve()
    print(lightcyan(f"Parse a directory: ") + f"{ml_models_path}")
    if args.recursive:
        paths = ml_models_path.rglob("*")
    else:
        paths = ml_models_path.iterdir()

    model_paths: list[Path] = []
    for path in sorted(paths):
        if not path.is_file():
            continue

        ext = path.suffix
        if ext not in filtered_exts:
            continue

        if ext in trt_extensions and not is_tensorrt_available():
            print(f"[E] Unsupported hardware: {path} cannot be loaded")
        model_paths.append(path)

    print(lightcyan(f"Found candidates: ") + f"{len(model_paths)}")
    parsed: int = 0
    failed: list[Path] = []
    for path in model_paths:
        device = 'cuda' if ext in trt_extensions else 'cpu'

        print(lightgreen(f"{path}"))
        try:
            start_time= time.time()
            model: NnModel = nnlib.open(path, device)
            if model is None:
                failed.append(path)
                continue

            elapsed = time.time() - start_time
            dtype: str = (
                ", ".join(model.arch.dtypes)
                if model.framework.type == NnFrameworkType.PYTORCH
                else model.io_dtypes['input']
            )
            print(
                f"    arch:", lightcyan(model.arch_name),
                f"scale:", lightcyan(model.scale),
                f"\n    in_dtype:", lightcyan(dtype),
                f"\n    ({1000 * elapsed:.1f}ms)\n"
            )

            # Display metadata
            if args.metadata:
                print_metadata(model.metadata)

            if args.verbose or args.debug:
                print("Model:")
                print(model)

            if args.debug:
                print("Arch:")
                print(model.arch)

            parsed += 1

        except Exception as e:
            failed.append(path)
            if args.dev:
                print(red(f"Exception:") + f"{e} (retry for debug)")
                model: NnModel = nnlib.open(path, device)
            else:
                continue

    print(lightcyan(f"Successfully parsed: ") + f"{parsed}/{len(model_paths)}")
    if failed:
        print(red(f"Failed to parse:\n    ") + f"{'\n    '.join([str(f) for f in failed])}")



if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    main()
