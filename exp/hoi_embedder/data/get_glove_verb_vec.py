import os
import numpy as np
from tqdm import tqdm

from data.hico.hico_constants import HicoConstants
import utils.io as io


def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in tqdm(f):
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model


def main():
    hico_const = HicoConstants()
    glove_txt = os.path.join(hico_const.proc_dir,'glove.6B/glove.6B.300d.txt')
    glove_model = loadGloveModel(glove_txt)

    verb_list = io.load_json_object(hico_const.verb_list_json)
    num_verbs = len(verb_list)
    glove_verb_vecs = np.zeros([num_verbs,300])
    for verb_info in verb_list:
        verb_id = verb_info['id']
        verb_name = verb_info['name']
        verb_words = verb_name.split('_')
        glove_vec = np.zeros(300)
        for word in verb_words:
            glove_vec = glove_vec + glove_model[word]
        glove_vec = glove_vec / len(verb_words)
        verb_idx = int(verb_id)-1
        glove_verb_vecs[verb_idx] = glove_vec

    glove_verb_vecs_npy = os.path.join(
        hico_const.proc_dir,
        'glove_verb_vecs.npy')
    np.save(glove_verb_vecs_npy,glove_verb_vecs)


if __name__=='__main__':
    main()