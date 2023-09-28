import os
if __name__=="__main__":
    value=""
    size = os.path.getsize('test.txt')
    with open('test.txt', 'rb') as binfile:
        for i in range(size):
            data = binfile.read(1)  # 每次输出一个字节
            value += '0x' + data.hex() + ','
            if (i + 1) % 16 == 0:
                value += '\n'
        value = value.strip()[:-1]  # 去除结尾的逗号
    print(value)