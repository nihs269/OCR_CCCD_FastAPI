import requests
import glob

url = 'http://10.10.10.12:1234/trich_xuat_thong_tin_cccd'

# for file in glob.glob('D:/NAMLT/DangKyXe/img_back/*'):
#     print(file)

myfiles = {'file': open('0.264975867447091_rotate.jpg', 'rb')}

x = requests.post(url, files=myfiles)
print(x.text)
