import argparse
import logging
from cryosiam.apps.dense_simsiam_regression.predict import main as denoise_predict_main
from cryosiam.apps.dense_simsiam_semantic.predict import main as semantic_predict_main
from cryosiam.apps.dense_simsiam_instance.predict import main as instance_predict_main

__version__ = "0.1.0"


def configure_logging(verbosity):
    level = logging.WARNING  # default
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(format="%(levelname)s: %(message)s", level=level)


def main():
    parser = argparse.ArgumentParser(prog="cryosiam", description="CryoSiam Command Line Interface")
    parser.add_argument("--version", action="version", version=f"CryoSiam {__version__}")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity level (-v, -vv)")
    parser.add_argument("-q", "--quiet", action="store_true", help="Silence all logging")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # denoise_predict subcommand
    sp_semantic = subparsers.add_parser("denoise_predict", help="Run denoising prediction")
    sp_semantic.add_argument('--config_file', type=str, required=True)
    sp_semantic.add_argument('--filename', type=str, required=False, default=None)
    sp_semantic.set_defaults(func=lambda args: denoise_predict_main(args.config_file, args.filename))

    # semantic_predict subcommand
    sp_semantic = subparsers.add_parser("semantic_predict", help="Run semantic segmentation prediction")
    sp_semantic.add_argument('--config_file', type=str, required=True)
    sp_semantic.add_argument('--filename', type=str, required=False, default=None)
    sp_semantic.set_defaults(func=lambda args: semantic_predict_main(args.config_file, args.filename))

    # instance_predict subcommand
    sp_semantic = subparsers.add_parser("instance_predict", help="Run instance segmentation prediction")
    sp_semantic.add_argument('--config_file', type=str, required=True)
    sp_semantic.add_argument('--filename', type=str, required=False, default=None)
    sp_semantic.set_defaults(func=lambda args: semantic_predict_main(args.config_file, args.filename))

    args = parser.parse_args()

    # Configure logging
    if args.quiet:
        logging.disable(logging.CRITICAL)
    else:
        configure_logging(args.verbose)

    # Run selected command
    args.func(args)
