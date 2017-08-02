
import os
import os.path

def reindex_file(olddirname, newdirname, filename_prefix, currind, newind):
    old_filename = os.path.join(olddirname, "%s%d.dat" % (filename_prefix, currind))
    new_filename = os.path.joint(newdirname, "%s%d.dat" % (filename_prefix, newind))
    os.rename(old_filename, new_filename)

if __name__ == "__main__":
    DIRNAME = "../Samples/Train"
    DIRNAME = "../SamplesReindexed/Train"
    N = 10000
    OFFSET = 10000

    for i in range(N):
        reindex_file(DIRNAME, DIRNAME, "img", i, i + OFFSET)
        reindex_file(DIRNAME, DIRNAME, "recon", i, i + OFFSET)


