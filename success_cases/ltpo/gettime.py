import os


time = []

for i in range(32):

  time.append(0)

  os.chdir(str(i))

  with open("log","r") as f:

   for i,fline in enumerate(f):

      if i > 0:

        time[-1] += float(fline.split()[-2])

  os.chdir("../")

print (max(time)/3600)
