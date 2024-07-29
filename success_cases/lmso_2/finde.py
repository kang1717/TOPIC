import sys

with open('log','r') as f:
  for fline in f:
    if fline.split()[4] == '0' and fline.split()[6] == '0':
#      if  float(fline.split()[9]) < -809:     
        print (fline,end='')
