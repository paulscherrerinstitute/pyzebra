source /opt/miniconda3/etc/profile.d/conda.sh

conda activate test
python /opt/pyzebra/pyzebra/app/cli.py --port=5010 --allow-websocket-origin=pyzebra.psi.ch:5010 --spind-path=/opt/spind
