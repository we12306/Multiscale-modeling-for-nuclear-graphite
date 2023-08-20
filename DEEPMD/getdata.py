#CP2K AIMD 和 OPT数据集分割，感谢https://wiki.cheng-group.net/wiki/software_usage/DeePMD-kit/
from ase.io import read
import numpy as np
import os, sys
import glob
import shutil

#############################
# USER INPUT PARAMETER HERE #
#############################
# input data path here, string, this directory should contains
#   ./data/*frc-1.xyz ./data/*pos-1.xyz
#必须输出pos.xyz frc.xyz .cell三个文件，同时这里没有维里，
data_path = "/mnt/d/desktop/potential_morse/graphite_Xe/graphite_Xe_CP2K/MD/graphite_Xe_perfect/graphite_300K_Xe_0.01_model01/graphite_300K_Xe_0.01_model01-REFTRAJ"
train_path="data/train"
test_path="data/test"
#input the number of atom in system
atom_num = 808

#train set and test set
train_ratio=0.8

# conversion unit here, modify if you need
au2eV = 2.72113838565563E+01 #27.21138
au2A = 5.29177208590000E-01


####################
# START OF PROGRAM #
####################
def xyz2npy(pos, atom_num, index_train, index_test, train_coord_path,
            test_coord_path, unit_convertion=1.0):
    #train
    total_train = np.empty((0,atom_num*3), float)
    for i in index_train:
        single_pos=pos[i]
        assert atom_num==len(single_pos.get_positions()),\
        "atom_num(%d)!=num(%d) in pos.xyz"%(atom_num,len(single_pos.get_positions()))
        tmp=single_pos.get_positions() #返回每一帧的所有原子坐标(atom_num*3)
        tmp=np.reshape(tmp,(1,atom_num*3)) #把每层(atom_num*3)展开为1*(atom_num*3)
        total_train = np.concatenate((total_train,tmp), axis=0)
    total_train = total_train * unit_convertion
    np.save(train_coord_path, total_train)
    #test
    total_test = np.empty((0,atom_num*3), float)
    for i in index_test:
        single_pos=pos[i]
        assert atom_num==len(single_pos.get_positions()),\
        "atom_num(%d)!=num(%d) in pos.xyz"%(atom_num,len(single_pos.get_positions()))
        tmp=single_pos.get_positions() #返回每一帧的所以原子坐标(atom_num*3)
        tmp=np.reshape(tmp,(1,atom_num*3)) #把每层(atom_num*3)展开为1*(atom_num*3)
        total_test = np.concatenate((total_test,tmp), axis=0)
    total_test = total_test * unit_convertion
    np.save(test_coord_path, total_test)
    #print(total_test[0])
    #np.save(output, total)
    #return total #返回frame*len(atom_num*3)

def energy2npy(pos,index_train, index_test, train_coord_path,
            test_coord_path, unit_convertion=1.0):
     #train
     total_train = np.empty((0), float)
     for i in index_train:
         single_pos=pos[i]
         tmp=single_pos.info.pop('E')
         tmp=np.array(tmp,dtype="float")
         tmp=np.reshape(tmp,1)
         total_train = np.concatenate((total_train,tmp), axis=0)
     total_train = total_train * unit_convertion
     #print(tmp,total[-1])
     np.save(train_coord_path,total_train)
     #test
     total_test = np.empty((0), float)
     for i in index_test:
         single_pos=pos[i]
         tmp=single_pos.info.pop('E')
         tmp=np.array(tmp,dtype="float")
         tmp=np.reshape(tmp,1)
         total_test = np.concatenate((total_test,tmp), axis=0)
     total_test = total_test * unit_convertion
     #print(tmp,total[-1])
     np.save(test_coord_path,total_test)
    # print(total_test[0])
     #return total #返回len(frame)的一维列表

def getcell(cell_path):
    data= open(cell_path,"r",encoding='UTF-8') 
    content = data.readlines()
    cell=[]
    for  i in content[1:]:
        celli=i.split()
        cellx=[float(celli[2]),float(celli[3]),float(celli[4]),]
        celly=[float(celli[5]),float(celli[6]),float(celli[7]),]
        cellz=[float(celli[8]),float(celli[9]),float(celli[10]),]
        cell.append([cellx,celly,cellz])
    return cell #返回frame(*3*3)

def cell2npy(cell,index_train, index_test, train_coord_path,
            test_coord_path, unit_convertion=1.0):
    #train
    total = np.empty((0,9),float)
    frame_num = len(pos)
    for frame in index_train:
        celli = np.array(cell[frame], dtype="float")
        celli = np.reshape(celli, (1,9))
        total = np.concatenate((total,celli),axis=0)
    total = total * unit_convertion
    np.save(train_coord_path,total)
    #test
    total = np.empty((0,9),float)
    frame_num = len(pos)
    for frame in index_test:
        celli = np.array(cell[frame], dtype="float")
        celli = np.reshape(celli, (1,9))
        total = np.concatenate((total,celli),axis=0)
    total = total * unit_convertion
    np.save(test_coord_path,total)
    #print(total[0])
    #return total #返回frame*9
    

def type_raw(single_pos,train_output,train_output2,test_output,test_output2):
    element = single_pos.get_chemical_symbols()
    element = np.array(element)
    tmp, indice = np.unique(element, return_inverse=True)
    #print(tmp)
    np.savetxt(train_output, indice, fmt='%s',newline='\n')
    np.savetxt(train_output2, tmp, fmt='%s',newline='\n')
    np.savetxt(test_output, indice, fmt='%s',newline='\n')
    np.savetxt(test_output2, tmp, fmt='%s',newline='\n')



# read the pos and frc
data_path = os.path.abspath(data_path)
pos_path = os.path.join(data_path, "*pos-1.xyz")
frc_path = os.path.join(data_path, "*frc-1.xyz")
cell_path = os.path.join(data_path, "*-1.cell")
#print(data_path)
pos_path = glob.glob(pos_path)[0]
frc_path = glob.glob(frc_path)[0]
cell_path= glob.glob(cell_path)[0]
#print(pos_path)
#print(frc_path)
#print(cell_path)
pos = read(pos_path, index = ":" )
print("获取%d帧pos数据from%s"%(len(pos),pos_path))
frc = read(frc_path, index = ":" )
print("获取%d帧frc数据from%s"%(len(frc),frc_path))
cell=getcell(cell_path)
print("获取%d帧cell数据from%s"%(len(cell),cell_path))
print("=============")
print("共获取%d帧数据"%len(pos))
print("=============")
#print(pos[0].get_positions()) #打印第0帧的坐标

assert len(cell)==len(pos), \
"请注意，cell的数据与pos的数据长度并不相等，你可能使用了OPT数据，\
最后一步输出了单点能数据,可能没有收敛，可以考虑剔除最后一帧"

#切分数据
data_size=len(cell)
test_size=int((1-train_ratio)*data_size)
train_size=data_size - test_size
print("参与训练的共有%d帧，其中训练集%d帧,验证测试集%d帧"%(data_size,train_size,test_size))
index_test = np.random.choice(data_size,size=test_size,replace=False)
index_train = list(set(range(data_size))-set(index_test))

# train numpy path
train_data_path=os.path.join(train_path, "data")
train_set_path = os.path.join(train_data_path, "set.000")
if os.path.isdir(train_set_path):
    print("detect directory exists\n now remove it")
    shutil.rmtree(train_set_path)
    os.makedirs(train_set_path)
else:
    print("detect directory doesn't exist\n now create it")
    os.makedirs(train_set_path)
train_type_path = os.path.join(train_data_path, "type.raw")
train_type_path2 = os.path.join(train_data_path, "type_map.raw")
train_coord_path = os.path.join(train_set_path, "coord.npy")
train_force_path = os.path.join(train_set_path, "force.npy")
train_box_path = os.path.join(train_set_path, "box.npy")
train_energy_path = os.path.join(train_set_path, "energy.npy")
# test numpy path
test_data_path=os.path.join(test_path, "data")
test_set_path = os.path.join(test_data_path, "set.000")
if os.path.isdir(test_set_path):
    print("detect directory exists\n now remove it")
    shutil.rmtree(test_set_path)
    os.makedirs(test_set_path)
else:
    print("detect directory doesn't exist\n now create it")
    os.makedirs(test_set_path)
test_type_path = os.path.join(test_data_path, "type.raw")
test_type_path2 = os.path.join(test_data_path, "type_map.raw")
test_coord_path = os.path.join(test_set_path, "coord.npy")
test_force_path = os.path.join(test_set_path, "force.npy")
test_box_path = os.path.join(test_set_path, "box.npy")
test_energy_path = os.path.join(test_set_path, "energy.npy")


#tranforrmation
xyz2npy(pos, atom_num,index_train, index_test,train_coord_path,test_coord_path) #; print(posdata.shape,posdata[1])
xyz2npy(frc, atom_num,index_train, index_test,train_force_path,test_force_path, au2eV/au2A)
energy2npy(pos,index_train, index_test, train_energy_path,test_energy_path, au2eV)
cell2npy(cell,index_train, index_test, train_box_path,test_box_path)
type_raw(pos[0], train_type_path,train_type_path2, test_type_path,test_type_path2)


'''
# 读入 ABACUS/MD 格式的数据
data = dpdata.LabeledSystem('DeePMD-kit_Tutorial/00.data/abacus_md', fmt = 'abacus/md') 
print('# 数据包含%d帧' % len(data))

# 随机选择40个索引作为验证集数据
index_validation = np.random.choice(201,size=40,replace=False)

# 其他索引作为训练集数据
index_training = list(set(range(201))-set(index_validation))
data_training = data.sub_system(index_training)
data_validation = data.sub_system(index_validation)

# 将所有训练数据放入文件夹"training_data"中
data_training.to_deepmd_npy('DeePMD-kit_Tutorial/00.data/training_data')

# 将所有验证数据放入文件夹"validation_data"中
data_validation.to_deepmd_npy('DeePMD-kit_Tutorial/00.data/validation_data')

print('# 训练数据包含%d帧' % len(data_training)) 
print('# 验证数据包含%d帧' % len(data_validation))
'''
