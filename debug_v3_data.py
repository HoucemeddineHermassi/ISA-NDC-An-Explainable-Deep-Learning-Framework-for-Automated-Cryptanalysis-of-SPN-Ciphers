import torch
try:
    d = torch.load('dataset_v3.pt', map_location='cpu')
    print(f'Samples: {len(d["data"])}')
    print(f'Metadata: {d["metadata"]}')
except Exception as e:
    print(f'Error: {e}')
