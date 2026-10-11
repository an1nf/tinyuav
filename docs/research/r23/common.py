"""Where the R23 CPU experiments read and write (R23_DIR, default results/r23_cpu)."""
import os

D = os.path.abspath(os.environ.get('R23_DIR', os.path.join('results', 'r23_cpu')))
os.makedirs(D, exist_ok=True)
