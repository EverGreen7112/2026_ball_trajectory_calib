import ntcore
import json


inst = ntcore.NetworkTableInstance.getDefault()
inst.startServer()
table = inst.getTable("shooter_table")
transmit = table.getBooleanTopic("transmit").subscribe(False)
robot_speed_x = table.getDoubleTopic("robot_speed_x").subscribe(0.0)
robot_speed_y = table.getDoubleTopic("robot_speed_y").subscribe(0.0)
robot_angular_speed = table.getDoubleTopic("robot_angular_speed").subscribe(0.0)
shooter_speed = table.getDoubleTopic("shooter_speed").subscribe(0.0)
shooter_angle = table.getDoubleTopic("shooter_angle").subscribe(0.0)
feeder_speed = table.getDoubleTopic("feeder_speed").subscribe(0.0)


def getData():
    return (robot_speed_x.get(), robot_speed_y.get(), robot_angular_speed.get(), shooter_speed.get(), shooter_angle.get(), feeder_speed.get())

def write_data(CoefList, r_s_x, r_s_y, r_a_s, s_s, s_a, f_s):

    with open("data.json", 'r') as f:
        try:
            current_content = json.loads(f.read())
        except:
            # Handle cases where the file might be empty or corrupted
            current_content = []

    # Deconstructing each sub-list
    for p , v, a, j in CoefList:
        entry = {
            "p": p,
            "v": v,
            "a": a,
            "j": j
        }
        current_content.append(entry)
        current_content.append({r_s_x, r_s_y, r_a_s, s_s, s_a, f_s})
    # Write the list of dictionaries to a file
    with open("data.json", "w") as f:
        f.write(json.dumps(current_content))