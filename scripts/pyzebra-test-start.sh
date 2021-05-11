source /home/pyzebra/miniconda3/etc/profile.d/conda.sh

conda activate test
python ~/pyzebra/pyzebra/app/cli.py --allow-websocket-origin=pyzebra.psi.ch:5006 --spind-path=/home/pyzebra/spind
