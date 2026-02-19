import ntcore
import json


inst = ntcore.NetworkTableInstance.getDefault()

table = inst.getTable("shooter_table")

transmit_start = table.getBooleanTopic("transmit_start").subscribe(False)
transmit_stop = table.getBooleanTopic("transmit_stop").subscribe(False)
transmit_restart = table.getBooleanTopic("transmit_restart").subscribe(False)

robot_speed_x = table.getDoubleTopic("robot_speed_x").subscribe(0.0)
robot_speed_y = table.getDoubleTopic("robot_speed_y").subscribe(0.0)
robot_angular_speed = table.getDoubleTopic("robot_angular_speed").subscribe(0.0)
shooter_speed = table.getDoubleTopic("shooter_speed").subscribe(0.0)
shooter_angle = table.getDoubleTopic("shooter_angle").subscribe(0.0)
feeder_speed = table.getDoubleTopic("feeder_speed").subscribe(0.0)



inst.startClient4("example")


inst.setServerTeam(7112)


inst.startDSClient()

def getTransmitStart():
    return transmit_start.get()
def getTransmitStop():

    return transmit_stop.get()

def getTransmitRestart():
    return transmit_restart.get()

def getData():
    return (robot_speed_x.get(), robot_speed_y.get(), robot_angular_speed.get(),
            shooter_speed.get(), shooter_angle.get(), feeder_speed.get())

def get_shooter_speed():
    return shooter_speed.get()

def write_data(CoefList, robot_vel_x, robot_vel_y, robot_angular_v, shooter_vel, aim_angle, feeder_vel):
    if not CoefList or CoefList[0] is None:
        return

    with open("data.json", 'r') as f:
        try:
            json_contents = json.loads(f.read())
        except:
            # Handle cases where the file might be empty or corrupted
            json_contents = []

    # Deconstructing each sub-list
    cur_example = dict()
    cur_example["robot_state"] = {
        "robot_vel_x": robot_vel_x,
        "robot_vel_y": robot_vel_y,
        "robot_angular_v": robot_angular_v,
        "shooter_vel": shooter_vel,
        "aim_angle": aim_angle,
        "feeder_vel": feeder_vel
    }
    NAMES = ["x", "y", "z"]
    for i, coefs in enumerate(CoefList):
        if i == 3:
            (p, v, a, j) = coefs
        else:
            (p, v, a) = coefs
            j = 0.0
        entry = {
            "j": j,
            "a": a,
            "v": v,
            "p": p
        }
        cur_example[NAMES[i]] = entry
        print(f"-({p} + ({v}x) + ({a}x^2) + ({j}x^3))")
    json_contents.append(cur_example)
    # Write the list of dictionaries to a file
    with open("data.json", "w") as f:
        f.write(json.dumps(json_contents))