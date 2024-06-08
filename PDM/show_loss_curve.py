import numpy as np
import matplotlib.pyplot as plt

def main():
    # path_0 = '/bigpool/homes/yeyun/projects/PyramidDiffusionModel/PDM/exp_local/Swin/CELEBA_64/2024.06.04 (no shifting test)/logs'
    # path_1 = '/bigpool/homes/yeyun/projects/PyramidDiffusionModel/PDM/exp_local/Swin/CELEBA_64/(shifting step 4)/logs'
    # path_2 = '/bigpool/homes/yeyun/projects/PyramidDiffusionModel/PDM/exp_local/Swin/CELEBA_64/(shifting step 8)/logs'
    # path_3 = '/bigpool/homes/yeyun/projects/PyramidDiffusionModel/PDM/exp_local/Swin/CELEBA_64/(shifting step 12)/logs'
    # path_4 = '/bigpool/homes/yeyun/projects/PyramidDiffusionModel/PDM/exp_local/Swin/CELEBA_64/(shifting step 16)/logs'
    # paths = [path_0, path_1, path_2, path_3, path_4]

    path_0 = '/bigpool/homes/yeyun/projects/PyramidDiffusionModel/PDM/exp_local/Swin/CELEBA_64/(no shifting test)/logs'
    path_1 = '/bigpool/homes/yeyun/projects/PyramidDiffusionModel/PDM/exp_local/Swin/CELEBA_64/(shifting step 8)/logs'
    path_2 = '/bigpool/homes/yeyun/projects/PyramidDiffusionModel/PDM/exp_local/Swin/CELEBA_64/(random shift step 8)/logs'
    paths = [path_0, path_1, path_2]

    datas = []
    for path in paths:
        datas.append(get_data(path))

    # length = min([len(data) for data in datas])
    length = 5000 // 80

    for i in range(len(datas)):
        datas[i] = datas[i][1:length]

    x_value = np.arange(len(datas[0])) * 80

    plt.figure()
    for data in datas:
        print(len(data))
        plt.plot(x_value, data)
    # l = ['no shifting', 'shift step 4', 'shift step 8', 'shift step 12', 'shift step 16']
    l = ['no shifting', 'fix_pattern', 'random']
    plt.legend(l)
    # plt.title('Loss CelebA 128x128 scale 4 batchsize 8')
    plt.title('Loss with different shift pattern')
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.savefig('./loss_shift_step.jpg')
    plt.close()

def get_data(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    
    loss = []
    for line in lines:
        if line.startswith('Loss:'):
            loss.append(float(line.split(' ')[-1]))
        
    return np.array(loss)


if __name__ == '__main__':
    main()