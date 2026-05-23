#flip script

from djitellopy import tello
from time import sleep

me = tello.Tello()
me.connect()

print(me.get_battery())


me.takeoff()
sleep(2)


while (me.get_height()<100):
	print(me.get_height())
	me.move_up(10)
	sleep(1)

me.flip_back()
sleep(2)

me.land()