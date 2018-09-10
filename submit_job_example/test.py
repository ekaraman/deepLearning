#!/usr/bin/python

import datetime
import sys

old_stdout = sys.stdout

log_file = open("message.log","w")

sys.stdout = log_file

print ("Hello World!")
print ()
print ("this will be written to message.log")

now=datetime.datetime.now()

print ()
print ("Current date and time using str method of datetime object:")
print ()
print (str(now))

sys.stdout = old_stdout

log_file.close()
