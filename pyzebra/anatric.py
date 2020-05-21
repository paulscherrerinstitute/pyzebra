import subprocess

def anatric(config_file):
    subprocess.run(["anatric", config_file], check=True)
