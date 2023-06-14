
with open('D:/Project/NerveStitcher-master/stitch6.10/mean_euclidean_distance_twOD.txt', 'r') as f1:
    file1_lines = f1.readlines()

# 读取第二个txt文件
with open('D:/Project/FlowNet/flownet/final_speed_twOD.txt', 'r') as f2:
    file2_lines = f2.readlines()

# 计算对应行差值百分比
with open('../diff_percent_twOD.txt', 'a') as f:
    for i in range(len(file1_lines)):
        line1 = float(file1_lines[i])
        line2 = float(file2_lines[i])
        diff_percent = (line2 - line1) / line1 * 100
        abs_diff_percent = abs(diff_percent)
        f.write(f" {abs_diff_percent:.2f}\n") # 将百分比保存到txt文件

