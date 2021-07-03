import os
import shutil


folder_path = '/home/piyush/Downloads/data_0.35/liquid3_x10_y10_f200/pt'
save_path = './d_0.35/'


for i in range(2):
    for j in range(2):
        sample_path = save_path+str(i)+"_"+str(j)
        os.mkdir(sample_path)
        for k in range(200):
            f = open(folder_path+"/{}_{}_{}.txt".format(i,j,k),'r')
            s = open(sample_path+"/{}.txt".format(k),'w')
            lines = f.readlines()
            for count, val in enumerate(lines):
                if (count!=0):
                    val = val.strip().split()
                    val = val[1]
                    val = val.replace('[','')
                    val = val.replace(']','')
                    val = val.split(',')
                    s.write(val[0]+'\t'+val[1]+'\t'+val[2]+"\n")

            f.close()
            s.close()





            

















