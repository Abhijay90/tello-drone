from djitellopy import tello
from time import sleep

me = tello.Tello()
me.connect()

print(me.get_battery())



print("hello")

me.takeoff()
me.send_rc_control(0,30,0,0) #forward movement
sleep(2)
me.send_rc_control(0,-30,0,0) #backward movement
sleep(2)
me.send_rc_control(0,30,0,0) #momentem breaker
sleep(1)
me.send_rc_control(30,0,0,0) # move right
sleep(2)
me.send_rc_control(-30,0,0,0) # move left
sleep(2)
me.send_rc_control(30,0,0,0) #momentem breaker
sleep(1)
me.send_rc_control(0,0,-20,0) #momentem breaker
sleep(2)
me.send_rc_control(0,0,0,0)
me.land()