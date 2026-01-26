import ntcore
import json


inst = ntcore.NetworkTableInstance.getDefault()
inst.startServer()
table = inst.getTable("shooter_table")
transmit = table.getBooleanTopic("transmit").subscribe(False)
robot_speed = table.getDoubleTopic("robot_speed").subscribe(0.0)
shooter_speed = table.getDoubleTopic("shooter_speed").subscribe(0.0)
shooter_angle = table.getDoubleTopic("shooter_angle").subscribe(0.0)

def getData():
    return (robot_speed.get(), shooter_speed.get(), shooter_angle.get())

def write_data(CoefList, r_s, s_s, s_a):

    with open("data.json", 'r') as f:
        try:
            current_content = json.loads(f.read())
        except:
            # Handle cases where the file might be empty or corrupted
            current_content = []
            print("hi")

    # Deconstructing each sub-list
    for p , v, a, j in CoefList:
        entry = {
            "p": p,
            "v": v,
            "a": a,
            "j": j
        }
        current_content.append(entry)

    # Write the list of dictionaries to a file
    with open("data.json", "w") as f:
        f.write(json.dumps(current_content))