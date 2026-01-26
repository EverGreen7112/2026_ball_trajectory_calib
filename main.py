import BallTrajectoryTracking
import recordData

if __name__ == '__main__':
    BallTrajectoryTracking.ballMain()
    templist = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
    recordData.write_data(templist,13,14,15)
    with open("data.json", 'r') as f:
        print(f.read())
