import pandas as pd, glob, os

print('='*90)
print('COMPLETE EXPERIMENT RESULTS')
print('='*90)

sections = [
    ('MULTIYEAR S2 2022-2025 (crown-level model sweep)', 'outputs/multiyear_s2_2022_2025_buffer10_all_holdouts'),
    ('PIXEL-LEVEL (>=20m2 crowns, 8 sim pixels/crown)',  'outputs/pixel_level_multiyear'),
]

for title, outdir in sections:
    print(f'\n{title}')
    print('-'*90)
    files = sorted(glob.glob(f'{outdir}/*.csv'))
    if not files:
        print('  (no files)')
        continue
    rows = []
    for f in files:
        df = pd.read_csv(f)
        best = df.loc[df['balanced_accuracy'].idxmax()]
        fname = os.path.basename(f)
        tag = fname.replace('_model_sweep.csv','').replace('_pixel_sweep.csv','')
        label = tag.split('_random')[0].split('_leave')[0]
        holdout = tag.split('leave_area_out_')[-1] if 'leave_area_out' in tag else 'random'
        n_test = int(best['n_test_crowns']) if 'n_test_crowns' in best.index else '?'
        rows.append({
            'label': label,
            'holdout': holdout,
            'best_model': best['model'],
            'bal_acc': round(best['balanced_accuracy'], 3),
            'macro_f1': round(best['macro_f1'], 3),
            'n_test': n_test,
        })
    print(pd.DataFrame(rows).to_string(index=False))
