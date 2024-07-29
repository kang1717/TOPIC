with open('log','r') as f:
  for fline in f:
    if fline.split()[4] == '0' and fline.split()[6] == '0':
#    if fline.split()[3] == '0':
      print (fline,end='')
