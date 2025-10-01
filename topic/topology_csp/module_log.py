def write_log_head():
    with open("log","a") as w:
        w.write("{:14}  {:29}  {:41}  {:41}  {:41}  {:6}\n".format(\
                "", "Structure gen", "LJ/harmonics", "NNP short", "NNP long", ""))
        w.write("{:>6}  {:>6}  {:>5}  {:>4}  {:>6}  {:>3}  {:>3}  {:>4}  {:>6}  {:>3}  {:>10}  {:>10}  {:>4}  {:>6}  {:>3}  {:>10}  {:>10}  {:>4}  {:>6}  {:>3}  {:>10}  {:>10}  {:>6}\n".format(\
        "Folder", "Num", "Trial", "fail", "time", "spg", "spg",\
        "fail", "time", "spg", "Energy", "Volume",\
        "fail", "time", "spg", "Energy", "Volume",\
        "fail", "time", "spg", "Energy", "Volume", "Time"))

def write_log(folder, pop, trial, fail_0, T0, spg, spg0, fail_1, T1, spg1, E0, V0,\
                                fail_2, T2, spg2, E1, V1, fail_3, T3, spg3, E2, V2):
    with open("log","a") as w:
        T = T0 + T1 + T2 + T3
        w.write("{:>6}  {:>6}  {:>5}  {:>4}  {:>6.2f}  {:>3}  {:>3}  {:>4}  {:>6.2f}  {:>3}  {:>10.2f}  {:>10.2f}  {:>4}  {:>6.2f}  {:>3}  {:>10.2f}  {:>10.2f}  {:>4}  {:>6.2f}  {:>3}  {:>10.2f}  {:>10.2f}  {:>6.2f}\n".format(\
        folder, pop, trial, fail_0, T0, spg, spg0,\
        fail_1, T1, spg1, E0, V0,\
        fail_2, T2, spg2, E1, V1,\
        fail_3, T3, spg3, E2, V2, T))

def write_poscar_log(infile, outfile, trial):
    with open(infile, "r") as f, open(outfile, "a") as w:
        for i, fline in enumerate(f):
            if i == 0:
                w.write(f"iteration {trial}\n")
            else:
                w.write(fline)

def write_poscars_contcars(poscars, contcars, contcar2s, contcar3s):
    with open('POSCARs', 'a') as w:
        for i in range(len(poscars)):
            w.write(poscars[i])

    with open('CONTCARs', 'a') as w:
        for i in range(len(contcars)):
            w.write(contcars[i])

    with open('CONTCAR2s', 'a') as w:
        for i in range(len(contcar2s)):
            w.write(contcar2s[i])

    with open('CONTCAR3s', 'a') as w:
        for i in range(len(contcar3s)):
            w.write(contcar3s[i])

def make_poscars_contcars_prev(filename, foldernum, gen):
    text = "directory: "+str(foldernum)+" / generation: "+str(gen)+"\n"
    with open(filename, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                continue
            else:
                text += line

    return text

def make_poscars_contcars(pos, foldernum, gen):
    text = "directory: "+str(foldernum)+" / generation: "+str(gen)+"\n"
    text += "1.0\n"
    for i in range(len(pos['latt'])):
        text+=f"  {pos['latt'][i][0]}   {pos['latt'][i][1]}   {pos['latt'][i][2]}\n"
    text += " ".join(pos['atomlist'])+"\n"
    text += " ".join(list(map(str, pos['numlist'])))+"\n"
    text += "Cartesian\n"
    for i in range(len(pos['coor'])):
        text+=f"  {pos['coor'][i][0]}   {pos['coor'][i][1]}   {pos['coor'][i][2]}\n"

    return text
