import pickle
import os, random, gc
fn = '/home/lijiali1025/projects/AL_framework/hgraph2graph/polymers/tensors-0.0-debug.pkl'
with open(fn, 'rb') as f:
    batches = pickle.load(f)
    record_none_index = []
    max_attachment_for_each_batch = []
    none_index = 0
    for index, batch in enumerate(batches):

        if batch == None:
            record_none_index.append(none_index)
            none_index += 1

            continue
        else:
            tensors = batch[1]
            tree_tensors = tensors[0]
            fmess = tree_tensors[1]
            edge_information = fmess[:, 2]
            max_attach = max(edge_information)
            max_attachment_for_each_batch.append([index,max_attach])
        none_index += 1

    exceed_max_attach_index = []
    for max_attach_information in max_attachment_for_each_batch:
        if max_attach_information[1] > 20:
            exceed_max_attach_index.append(max_attach_information[0])
    record_none_index.extend(exceed_max_attach_index)

    for index in sorted(record_none_index, reverse=True):
        del batches[index]



with open('tensors-5.0-debug.pkl', 'wb') as f:
    pickle.dump(batches, f, pickle.HIGHEST_PROTOCOL)