import subprocess
import time

machines = [
    "machine-1-1",
    "machine-1-2",
    "machine-1-3",
    "machine-2-1",
    "machine-3-6"
]

subprocess.Popen(["python", "-m", "src.federated.server"])
time.sleep(5)

for m in machines:
    subprocess.Popen(["python", "-m", "src.federated.client", m])