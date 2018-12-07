#!/usr/bin/env mdl
import os
from tqdm import tqdm


def main():
    dir_list = os.listdir("./")
    pd = "./src/"
    for d in tqdm(dir_list):
        if os.path.isdir(d):
            sd = "./"+d+"/code"
            if os.path.isdir(sd):
                spd = pd + d + "/"
                spdc = spd + "code/"
                if not os.path.exists(spd):
                    os.system("mkdir {}".format(spd))
                if not os.path.exists(spdc):
                    os.system("mkdir {}".format(spdc))
                cmd = "cp {}/*.py {}".format(sd, spdc)
                cmd2 = "cp {}/*.txt {}".format(sd, spdc)
                os.system(cmd)
                os.system(cmd2)



if __name__ == "__main__":
    main()


# vim: ts=4 sw=4 sts=4 expandtab
