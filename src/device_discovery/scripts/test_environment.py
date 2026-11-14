import csv

from core.utils.constants.device_constants import ProbeType
from device_discovery.environment import load_environment

data_dir = "data/generated/moderate"

with open(f"{data_dir}/test.csv", encoding="utf-8") as file:
    device_id = next(csv.DictReader(file))["device_id"]

env = load_environment(data_dir, split="test")

print("Device:", device_id)
print("Before:", env.observe(device_id))

# Thử probe chưa được thực hiện ban đầu.
with open(f"{data_dir}/initial_attempts.json", encoding="utf-8") as file:
    import json
    attempted = set(json.load(file)[device_id])

probe = next(
    p for p in ProbeType
    if p.value not in attempted
)

result = env.query(device_id, probe)

print("Probe:", probe.value)
print("Result:", result)
print("After:", env.observe(device_id))
print("Total cost:", env.get_total_cost(device_id))