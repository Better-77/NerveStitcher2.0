count = 0
totalnumber = 0
with open('../diff_percent_zzOS.txt', 'r') as f:
    for line in f:
        if float(line.strip()) > 6:
            count += 1
        totalnumber += 1
print(count)
print(totalnumber)
print(count/totalnumber)
