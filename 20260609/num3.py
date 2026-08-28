with open ('sales.txt','w',encoding='UTF-8') as f:
    f.write("150\n200\n250\n300\n")
total = 0
with open ('sales.txt','r',encoding='UTF-8') as f:
    for line in f:
        num = float(line.strip())
        total += num
with open ('sales.txt','a',encoding='UTF-8') as f:
    f.write(f"\n总计:{total}元\n")
with open ('sales.txt','r',encoding='UTF-8') as f:
    print('最终账单')
    print(f.read())