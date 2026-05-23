#keyboard_control.pyvim

import pygame


pygame.init()
win=pygame.set_mode(400,400)

def getKey(KeyName):
	ans=False
	for eve in pygame.event.get():pass
	keyInput=pygame.key.get_pressed()
	mykey=getattr(pygame,'K_{}'.format(KeyName))
	if keyInput[mykey]:
		ans=True
	pygame.display.update()
	return ans


