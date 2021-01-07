import requests

# copy your API key from dashboard.mlhub.earth and paste it in the following
API_KEY = 'PASTE_YOUR_API_KEY_HERE'
API_BASE = 'https://api.radiant.earth/mlhub/v1'

COLLECTION_ID = 'ref_landcovernet_v1_labels'

r = requests.get(f'{API_BASE}/collections/{COLLECTION_ID}', params={'key': API_KEY})
print(f'Description: {r.json()["description"]}')
print(f'License: {r.json()["license"]}')
print(f'DOI: {r.json()["sci:doi"]}')
print(f'Citation: {r.json()["sci:citation"]}')

r = requests.get(f'{API_BASE}/collections/{COLLECTION_ID}/items', params={'key': API_KEY})
label_classes = r.json()['features'][0]['properties']['label:classes']
for label_class in label_classes:
    print(f'Classes for {label_class["name"]}')
    for c in sorted(label_class['classes']):
        print(f'- {c}')

