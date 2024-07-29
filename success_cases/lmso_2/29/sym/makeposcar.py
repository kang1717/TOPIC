import sys
import subprocess

with open("POSCARs",'r') as f, open("POSCAR"+sys.argv[1],'w') as w:

  target_i = 0
  activate = 0

  for i,fline in enumerate(f):

    if f'iteration {sys.argv[1]}' in fline:

      target_i = i
      activate = 1

    if activate ==1:

      if i <= target_i + 109: 

        w.write(fline)

      else:

        break

subprocess.call('/home/sung.w.kang/0transfer/VESTA-gtk2/VESTA POSCAR'+str(sys.argv[1]),shell=True)   
