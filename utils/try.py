'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-06-18 19:29:00
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2022-06-18 19:39:41
FilePath: /yolov5/utils/try.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import matplotlib.pyplot as plt
if __name__ == '__main__':
    epochs = 30
    lrf = 0.1 
    lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - lrf) + lrf 
    epoch = [float(i) for i in range(30)]
    lr_x = []
    for i in epoch:
        lr_x.append(lf(i))
    plt.plot(epoch,lr_x)
    plt.show()
