import os
import subprocess
import sys


def main():
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run(["bokeh", "serve", app_path, *sys.argv[1:]], check=True)


if __name__ == "__main__":
    main()
