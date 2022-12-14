import argparse
import logging
import os

from bokeh.application.application import Application
from bokeh.application.handlers import ScriptHandler
from bokeh.server.server import Server

from pyzebra import ANATRIC_PATH, SXTAL_REFGEN_PATH
from pyzebra.app.handler import PyzebraHandler

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """The pyzebra command line interface.

    This is a wrapper around a bokeh server that provides an interface to launch the application,
    bundled with the pyzebra package.
    """
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")

    parser = argparse.ArgumentParser(
        prog="pyzebra", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--port", type=int, default=5006, help="port to listen on for HTTP requests"
    )

    parser.add_argument(
        "--allow-websocket-origin",
        metavar="HOST[:PORT]",
        type=str,
        action="append",
        default=None,
        help="hostname that can connect to the server websocket",
    )

    parser.add_argument(
        "--anatric-path", type=str, default=ANATRIC_PATH, help="path to anatric executable"
    )

    parser.add_argument(
        "--sxtal-refgen-path",
        type=str,
        default=SXTAL_REFGEN_PATH,
        help="path to Sxtal_Refgen executable",
    )

    parser.add_argument("--spind-path", type=str, default=None, help="path to spind scripts folder")

    parser.add_argument(
        "--args",
        nargs=argparse.REMAINDER,
        default=[],
        help="command line arguments for the pyzebra application",
    )

    args = parser.parse_args()

    logger.info(app_path)

    pyzebra_handler = PyzebraHandler(args.anatric_path, args.spind_path)
    handler = ScriptHandler(filename=app_path, argv=args.args)
    server = Server(
        {"/": Application(pyzebra_handler, handler)},
        port=args.port,
        allow_websocket_origin=args.allow_websocket_origin,
    )

    server.start()
    server.io_loop.start()


if __name__ == "__main__":
    main()
