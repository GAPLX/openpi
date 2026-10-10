from openpi.shared import download


checkpoints = [
    "pi05_base"
]

for checkpoint in checkpoints:
    path = download.maybe_download(f"gs://openpi-assets/checkpoints/{checkpoint}")
    print(f"Downloaded {checkpoint} to {path}")
