import json
import os
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt   # NEW

# ---------- CONFIG ----------
JSON_FILE = "data.json"
MP4_FOLDER = r"C:\Robotics26\2026_ball_trajectory_calib\film\learn\learn edited"
# ----------------------------

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def parse_mp4_files(folder):
    results = []

    for file in os.listdir(folder):
        if not file.endswith(".mp4"):
            continue

        try:
            name = file.replace(".mp4", "")
            motorP, rest = name.split("_", 1)
            angle = rest.split(".e")[0]

            results.append({
                "motorP": float(motorP),   # convert to float
                "angle": float(angle)
            })
        except ValueError:
            continue

    return results

def extract_v_values(json_data):
    v_values = []

    for item in json_data:
        if isinstance(item, dict) and "v" in item:
            v_values.append(float(item["v"]))

    return v_values

def build_rows(v_values, mp4_data):
    rows = []
    count = min(len(v_values), len(mp4_data))

    for i in range(count):
        rows.append((
            v_values[i],
            mp4_data[i]["motorP"],
            mp4_data[i]["angle"]
        ))

    return rows

# ---------- NEW GRAPH FUNCTION ----------
def plot_v_vs_motorP(rows):
    motorP = [row[1] for row in rows]
    v = [row[0] for row in rows]

    plt.figure()
    plt.scatter(motorP, v)
    plt.xlabel("motorP")
    plt.ylabel("v")
    plt.title("v as a function of motorP")
    plt.grid(True)
    plt.show()
# --------------------------------------

def create_table(rows):
    root = tk.Tk()
    root.title("Data Viewer")
    root.geometry("600x400")

    columns = ("v", "motorP", "angle")
    table = ttk.Treeview(root, columns=columns, show="headings")

    for col in columns:
        table.heading(col, text=col)
        table.column(col, anchor="center", width=180)

    for row in rows:
        table.insert("", "end", values=row)

    scrollbar = ttk.Scrollbar(root, orient="vertical", command=table.yview)
    table.configure(yscrollcommand=scrollbar.set)

    table.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    root.mainloop()

def make_window():
    json_data = load_json(JSON_FILE)
    mp4_data = parse_mp4_files(MP4_FOLDER)
    v_values = extract_v_values(json_data)
    rows = build_rows(v_values, mp4_data)

    plot_v_vs_motorP(rows)   # <-- GRAPH
    create_table(rows)       # <-- TABLE
