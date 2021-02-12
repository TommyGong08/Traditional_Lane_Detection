import sys
import os
import ujson as json


if __name__ == '__main__':
    try:
        if len(sys.argv) != 2:
            print(sys.argv)
            raise Exception('Invalid input arguments')
        print(sys.argv[1])
        f = open("picture_list.txt", "w")
        fj = open("pic_test.json", "w")
        print("####")
        for line in open(sys.argv[1]).readlines():
            # 将json格式转化为字典
            i = json.loads(line)
            if os.path.isfile(i["raw_file"]):
                f.write(i["raw_file"])
                f.write("\n")
                fj.write(line)
        f.close()
        fj.close()
    except Exception as e:
        print(e.args)
