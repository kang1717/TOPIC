def write_log_head():

  with open("log","w") as w:

      w.write("  Folder   Iter_for_LJ/harmonic     Iter_for_NNP     Struct_gen_time  LJ_success   LJ_time  NNP_success  NNP_time  NNP_time2     Energy       Volume        T    spg\n")
      #print ("     Iter_for_LJ/harmonic     Iter_for_NNP     Struct_gen_time  LJ_success   LJ_time  NNP_success  NNP_time  NNP_time2     Energy       Volume        T")
def write_log(folder,spg,iteration_1, iteration_2, T0, T1, T2, T3, fail_2, fail_3, E, V):

  with open("log","a") as w:

    T = T0 +T1 + T2 + T3

    w.write(f" {folder:5}        {iteration_1:5}                   {iteration_2}                {T0:.2f}             {fail_2:2}         {T1:.2f}        {fail_3:3}        {T2:.2f}     {T3:.2f}         {E:4.2f}      {V:4.2f}      {T:.2f}     {spg}\n")
    #print (f"         {iteration_1:5}                   {iteration_2}                {T0:.2f}             {fail_2:2}         {T1:.2f}        {fail_3:3}        {T2:.2f}     {T3:.2f}         {E:4.2f}      {V:4.2f}      {T:.2f}")
    if fail_3 == 0:

      write_poscar_log("POSCAR","POSCARs",iteration_1)
      write_poscar_log("CONTCAR","CONTCARs",iteration_1)
      write_poscar_log("CONTCAR2","CONTCAR2s",iteration_1)
      write_poscar_log("CONTCAR3","CONTCAR3s",iteration_1)




def write_poscar_log(infile,outfile,iteration_1):

  with open(infile,"r") as f, open(outfile,"a") as w:

    for i,fline in enumerate(f):

      if i == 0:
        w.write(f"iteration {iteration_1}\n")

      else:
        w.write(fline)

