import os
import sys
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

datasets = [
    'Tuesday-WorkingHours',
    'Wednesday-workingHours',
    'Thursday-WorkingHours-Morning-WebAttacks'
]

base_dir = 'data/unknown_experiment'

for ds in datasets:
    print(f"=== DATASET: {ds} ===")
    for split in ['train', 'validation', 'test']:
        p = os.path.join(base_dir, ds, 'known', f'{split}.csv')
        df = pd.read_csv(p, usecols=['Label'])
        vc = df['Label'].value_counts().to_dict()
        print(f"  known/{split}.csv: {len(df)} rows | {vc}")
    
    hp = os.path.join(base_dir, ds, 'held_out', 'unknown_test.csv')
    df_h = pd.read_csv(hp, usecols=['Label'])
    vc_h = df_h['Label'].value_counts().to_dict()
    print(f"  held_out/unknown_test.csv: {len(df_h)} rows | {vc_h}")
    print()
