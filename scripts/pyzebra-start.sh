source /home/pyzebra/miniconda3/etc/profile.d/conda.sh

conda activate prod
pyzebra --port=80 --allow-websocket-origin=pyzebra.psi.ch:80
