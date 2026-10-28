import csv
import hashlib
import json
import math
from pathlib import Path
import re


def verify(root):
    manifest = json.loads((root/'data_artifact_manifest.json').read_text())
    for filename, key in [('benchmark-data.json', 'benchmark_data_sha256'), ('recorded_engine7_forecasts.csv', 'forecast_csv_sha256')]:
        assert hashlib.sha256((root/filename).read_bytes()).hexdigest() == manifest[key], filename
    data = json.loads((root/'benchmark-data.json').read_text())
    html = (root/'index.html').read_text()
    match = re.search(r'<script id="motor-data" type="application/json">(.*?)</script>', html, re.S)
    assert match is not None, 'Embedded data missing'
    assert json.loads(match.group(1)) == data, 'HTML and JSON payloads differ'
    assert set(data['predictions']) == {'7'}, 'Unexpected forecast engine'
    engine = data['engines']['7']
    observed = dict(zip(map(int, engine['cycle']), engine['T50']))
    with (root/'recorded_engine7_forecasts.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == manifest['forecast_count_per_model']
    for row in rows:
        cycle = int(row['target_cycle'])
        assert int(row['engine']) == 7
        assert float(row['observed_T50']) == observed[cycle]
        for name in manifest['models']:
            prediction = float(row[name])
            assert math.isfinite(prediction)
            assert prediction == data['predictions']['7'][name][str(cycle)]
        assert math.isclose(float(row['Persistence']), observed[cycle-1], abs_tol=1e-9)
    return len(rows), len(manifest['models'])


if __name__ == '__main__':
    count, models = verify(Path(__file__).resolve().parent)
    print(f'Verified {count} recorded forecast cycles for {models} models; payload and artifact hashes match.')
