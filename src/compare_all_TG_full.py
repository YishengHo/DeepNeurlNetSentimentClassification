#!/usr/bin/env mdl
import os

def main():
    fd_list = os.listdir('./')
    topacc_list = []
    for fd in fd_list:
        if os.path.isdir(fd):
            top_path = fd + '/code/TG_full_topacc.txt'
            if os.path.exists(top_path):
                l = open(top_path).readlines()
                topacc_list.extend((fd, l[len(l)-1]))
    for item in topacc_list:
        print(item)


if __name__ == "__main__":
    main()
# vim: ts=4 sw=4 sts=4 expandtab
