#C:\Users\justm\Desktop\KI-Studium\Projekt CV\Dataset_Normal_Streets\dataset50\Structured Animals NoBg\data.yaml

import yaml

# === Pfade check if necessary ===
names_path = "Dataset Street_Animals/classes.names"
yaml_path = "Dataset Street_Animals/data.yaml"

# === 1. classes.names update ===
with open(names_path, "a+", encoding="utf-8") as f:
    f.seek(0)
    existing_classes = [line.strip() for line in f.readlines()]
    if "Animal" not in existing_classes:
        f.write("Animal\n")
        print("✅ Klasse 'Animal' to classes.names added.")
    else:
        print("ℹ️ Klasse 'Animal' is already exist in classes.names.")

# === 2. data.yaml update ===
with open(yaml_path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

if "Animal" not in data["names"]:
    data["names"].append("Animal")
    data["nc"] = len(data["names"])

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    print("✅ 'Animal' to data.yaml added.")
else:
    print("ℹ️ 'Animal' is already exist.")
