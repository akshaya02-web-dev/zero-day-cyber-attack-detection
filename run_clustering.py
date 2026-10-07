"""
run_clustering.py
=================
Executes the full 5-cluster unsupervised learning pipeline to predict
held-out unknown attacks and stores the detected unknown attacks into output/unknown/.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from src.clustering_pipeline import run_all_clustering

if __name__ == "__main__":
    run_all_clustering()
