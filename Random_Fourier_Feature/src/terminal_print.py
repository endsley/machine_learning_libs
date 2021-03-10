#!/usr/bin/env python

import sys
import types
import numpy as np
import time

def clear_current_line():
	sys.stdout.write("\r")
	sys.stdout.write("\033[K")
	sys.stdout.flush()

def write_to_current_line(txt):
	clear_current_line()
	sys.stdout.write(txt)
	sys.stdout.flush()

def clear_previous_line():
	clear_current_line()
	sys.stdout.write("\r")
	sys.stdout.write("\033[F")
	sys.stdout.flush()
	clear_current_line()

def loss_optimization_printout(db, epoch, avgLoss, avgGrad, epoc_loop, slope):
	sys.stdout.write("\r\t\t%d/%d, MaxLoss : %f, AvgGra : %f, progress slope : %f" % (epoch, epoc_loop, avgLoss, avgGrad, slope))
	sys.stdout.flush()

def dictionary_to_str(dic):
	outstr = ''
	for i,j in dic.items():
		if type(j) == str: outstr += 	('\t\t' + i + ' : ' + str(j) + '\n')
		elif type(j) == np.float64: outstr += 	('\t\t' + i + ' : ' + str(j) + '\n')
		elif type(j) == bool: outstr += ('\t\t' + i + ' : ' + str(j) + '\n')
		elif type(j) == type: outstr += ('\t\t' + i + ' : ' + j.__name__ + '\n')
		elif type(j) == types.FunctionType: outstr += ('\t\t' + i + ' : ' + j.__name__ + '\n')
		elif type(j) == float: outstr += ('\t\t' + i + ' : ' + str(j) + '\n')
		elif type(j) == int: outstr += 	('\t\t' + i + ' : ' + str(j) + '\n')
		else:
			print('%s , %s is not recognized'%(i, str(type(j))))
			import pdb; pdb.set_trace()	

	return outstr

if __name__ == '__main__':
	while True:
		#print('Hello world\nSecond Line')
		write_to_current_line('Hello world\nSecond Line')
		time.sleep(2)
		clear_previous_line()
		time.sleep(2)
