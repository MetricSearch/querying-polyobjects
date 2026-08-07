from polyhood import Hood, MSEDRep, softmax, gMSED
import zarr
from pathlib import Path
import re
import numpy as np
import ctypes

hood_instance = Hood()
hood_instance_2 = Hood()

# zarr_store = zarr.open("../encode_test/results/results.zarr")
zarr_store = zarr.open("../../experiments/encode_results/raw_results/results.zarr")
zarr_embeddings = list(zarr_store['features'][:])
zarr_fname = list(zarr_store['filenames'][:])
zarr_classes = list(zarr_store['classifications'][:])

# zarr_mdata = list(zip(zarr_fname, zarr_classes))

zarr_mdata = [
    {
        "path": p,
        "class": c,
    }
    for p, c in zip(zarr_fname, zarr_classes)
]

hood_instance.add(zarr_embeddings,zarr_mdata)

# print(zarr_classes)
print(hood_instance.show_info())

firstn = hood_instance.get([i for i in range(10)])
# print(firstn)

# oldfname = ""
# queries = []
# classes = ""
# msedReps = []
# newmdata = []
# for i, elem in enumerate(hood_instance.store.metadatas):
#     fname = Path(elem['path']).name
#     fname = re.sub(r'-im.*jpg','', fname)

#     if i == 0:
#         queries.append(hood_instance.store.embeddings[i])
#         classes = elem['class']
#         oldfname = fname
#         continue
    
#     if fname != oldfname:
#         normalised_queries = softmax(queries)
#         msedReps.append(np.asarray(MSEDRep(normalised_queries).represent()))
#         newmdata.append({'path':str(Path(elem['path']).parent.parent.parent)+"/"+oldfname+"/"+oldfname+".jpg",'class':classes})
#         queries = []
#         queries.append(hood_instance.store.embeddings[i])
#         classes = elem['class']
#         oldfname = fname
#     else:
#         queries.append(hood_instance.store.embeddings[i])
#         classes = classes + ", " + elem['class']


# print(len(msedReps))
# print(len(newmdata))


# distances = np.zeros(len(msedReps), dtype=np.float32) 
# test_id = 0
# tester = msedReps[test_id]
# dim = len(tester)

# print(dim)

# tester_arr = np.ascontiguousarray(tester, dtype=np.float32)
# tester_ptr = tester_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

# for i, elem in enumerate(msedReps):
#     emb_arr = np.ascontiguousarray(elem, dtype=np.float32)
#     emb_ptr = emb_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
#     distances[i] = gMSED.distance(tester_ptr, emb_ptr, dim)

# Now distances is already a numpy array
# nn_neighbours = distances.argsort()[:10]

# print(nn_neighbours)

# print(newmdata[test_id]['path']+ " and classes: " + newmdata[test_id]['class'])
# print()
# for ind in nn_neighbours:
#     print(str(ind) + " , " +newmdata[ind]['path'] + " and classes: " + newmdata[ind]['class'])

# hood_instance_2.add(msedReps,newmdata)
# print(hood_instance_2.show_info())


# hood.dump()
# hood_instance_2.dump(path = "db.zarr")
# hood_instance.dump(path = "db_raw.zarr")