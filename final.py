final code 

import cv2
import numpy as np
import RPi.GPIO as GPIO
import time
import math
import heapq
from flask import *
import threading


app = Flask(__name__)
command = None

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

IN1=17
IN2=27
IN3=22
IN4=23

TRIG=5
ECHO=6

EN1=12
EN2=13
RAIN_SENSOR=16

GPIO.setup(IN1,GPIO.OUT)
GPIO.setup(IN2,GPIO.OUT)
GPIO.setup(IN3,GPIO.OUT)
GPIO.setup(IN4,GPIO.OUT)

GPIO.setup(TRIG,GPIO.OUT)
GPIO.setup(ECHO,GPIO.IN)

GPIO.setup(EN1,GPIO.OUT)
GPIO.setup(EN2,GPIO.OUT)
GPIO.setup(RAIN_SENSOR,GPIO.IN)

pwm_left = GPIO.PWM(EN1,1000)
pwm_right = GPIO.PWM(EN2,1000)

pwm_left.start(90)
pwm_right.start(90)

def forward():
    GPIO.output(IN1,1)
    GPIO.output(IN2,0)
    GPIO.output(IN3,1)
    GPIO.output(IN4,0)

def stop():
    GPIO.output(IN1,0)
    GPIO.output(IN2,0)
    GPIO.output(IN3,0)
    GPIO.output(IN4,0)

def turn_left():
    GPIO.output(IN1,0)
    GPIO.output(IN2,1)
    GPIO.output(IN3,1)
    GPIO.output(IN4,0)
    time.sleep(0.7)
    stop()

def turn_right():
    GPIO.output(IN1,1)
    GPIO.output(IN2,0)
    GPIO.output(IN3,0)
    GPIO.output(IN4,1)
    time.sleep(0.7)
    stop()

def check_rain_speed():

    rain = GPIO.input(RAIN_SENSOR)

    if rain == 0:
        print("Rain detected - reducing speed")
        pwm_left.ChangeDutyCycle(40)
        pwm_right.ChangeDutyCycle(40)

    else:
        pwm_left.ChangeDutyCycle(90)
        pwm_right.ChangeDutyCycle(90)

def get_distance():

    GPIO.output(TRIG,False)
    time.sleep(0.05)

    GPIO.output(TRIG,True)
    time.sleep(0.00001)
    GPIO.output(TRIG,False)

    start=time.time()
    end=time.time()

    while GPIO.input(ECHO)==0:
        start=time.time()

    while GPIO.input(ECHO)==1:
        end=time.time()

    duration=end-start
    distance=duration*17150

    return distance

GRID_SIZE=30
grid=np.zeros((GRID_SIZE,GRID_SIZE))

robot_x=0
robot_y=0

goal=None

direction="SOUTH"

def heuristic(a,b):
    return abs(a[0]-b[0])+abs(a[1]-b[1])

def astar(grid,start,goal):

    neighbors=[(1,0),(-1,0),(0,1),(0,-1)]

    close=set()
    came_from={}

    gscore={start:0}
    fscore={start:heuristic(start,goal)}

    open_list=[]
    heapq.heappush(open_list,(fscore[start],start))

    while open_list:

        current=heapq.heappop(open_list)[1]

        if current==goal:

            path=[]
            while current in came_from:
                path.append(current)
                current=came_from[current]

            return path[::-1]

        close.add(current)

        for i,j in neighbors:

            neighbor=(current[0]+i,current[1]+j)

            if not (0<=neighbor[0]<GRID_SIZE and 0<=neighbor[1]<GRID_SIZE):
                continue

            if grid[neighbor]==1:
                continue

            tentative_g=gscore[current]+1

            if neighbor in close and tentative_g>=gscore.get(neighbor,0):
                continue

            if tentative_g<gscore.get(neighbor,9999):

                came_from[neighbor]=current
                gscore[neighbor]=tentative_g
                fscore[neighbor]=tentative_g+heuristic(neighbor,goal)

                heapq.heappush(open_list,(fscore[neighbor],neighbor))

    return []


def draw_map(path):

    img=np.zeros((GRID_SIZE,GRID_SIZE,3),dtype=np.uint8)

    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):

            if grid[x][y]==1:
                img[y][x]=(0,0,255)

    for p in path:
        img[p[1]][p[0]]=(0,255,0)

    img[robot_y][robot_x]=(255,0,0)

    if goal:
        img[goal[1]][goal[0]]=(0,255,255)

    img=cv2.resize(img,(500,500),interpolation=cv2.INTER_NEAREST)

    cv2.imshow("Robot Map",img)


def update_obstacle():

    global grid

    if direction=="EAST" and robot_x+1<GRID_SIZE:
        grid[robot_x+1][robot_y]=1

    elif direction=="WEST" and robot_x-1>=0:
        grid[robot_x-1][robot_y]=1

    elif direction=="NORTH" and robot_y-1>=0:
        grid[robot_x][robot_y-1]=1

    elif direction=="SOUTH" and robot_y+1<GRID_SIZE:
        grid[robot_x][robot_y+1]=1


def rotate_to(target):

    global direction

    dirs=["NORTH","EAST","SOUTH","WEST"]

    current_index=dirs.index(direction)
    target_index=dirs.index(target)

    diff=(target_index-current_index)%4

    if diff==1:
        turn_right()

    elif diff==2:
        turn_right()
        turn_right()

    elif diff==3:
        turn_left()

    direction=target

def move_forward():

    forward()
    time.sleep(0.4)
    stop()

def move_to(next_node):

    global robot_x,robot_y

    dx=next_node[0]-robot_x
    dy=next_node[1]-robot_y

    if dx==1:
        target="EAST"
    elif dx==-1:
        target="WEST"
    elif dy==1:
        target="SOUTH"
    elif dy==-1:
        target="NORTH"

    rotate_to(target)
    move_forward()

    robot_x+=dx
    robot_y+=dy


@app.route('/api/command', methods=['POST'])
def receive_command():

    global goal

    data = request.json
    cmd = data.get("command")

    if cmd is None:
        return jsonify({"status":"error","message":"No command"}),400

    cmd = cmd.upper()

    if cmd == "A":
        goal = (10,20)

    elif cmd == "B":
        goal = (20,5)

    elif cmd == "C":
        goal = (5,25)

    elif cmd == "D":
        goal = (15,15)

    elif cmd == "E":
        goal = (0,0)

    else:
        return jsonify({"status":"error","message":"Invalid command"}),400

    print("Received Command:",cmd," Goal:",goal)

    return jsonify({
        "status":"success",
        "command":cmd,
        "goal":goal
    })


def robot_loop():

    global goal

    while True:

        check_rain_speed()

        if goal is None:
            time.sleep(0.5)
            continue

        distance=get_distance()

        if distance<25:

            stop()
            update_obstacle()
            time.sleep(0.5)

        path=astar(grid,(robot_x,robot_y),goal)

        draw_map(path)

        if len(path)>0:

            next_node=path[0]
            move_to(next_node)

        if (robot_x,robot_y)==goal:

            print("Goal reached")
            goal=None
            stop()

        if cv2.waitKey(1)==27:
            break



threading.Thread(target=robot_loop).start()

app.run(host='0.0.0.0',port=5000)