from .Classifier import Classifier
from .encoder.Encode import Encode
from polyhood import Hood, MSEDRep, softmax
from pathlib import Path
import re
import numpy as np
import ctypes
import zarr
from argparse import ArgumentParser

__all__ = ["Classifier", "Encode"]

def pipe_it(imgs_source:str = "/Volumes/Data/mf/images/", test_dir:str = "", classify=True, encode=True, store=True, bypass=False):
    if classify:
        # Run classifier and alayse the results after saving them
        classifier=Classifier(test_dir=(test_dir+"/distribution_results/"))
        classifier.classify(imgs_source, save_cropped_images=True, batch=True)
        classifier.describe_classifier()
        classifier.save_analysis()
    if encode:
        if bypass:
            ig_source = imgs_source
        else:
            ig_source = test_dir+"/distribution_results/results/"
        # Encode the stored images
        encoder = Encode()
        test_dir = test_dir.replace("/","")
        # encoder.encodes((ig_source),(test_dir+"/encode_results/"),dirs=False)
        encoder.encodes((ig_source),(test_dir+"/encode_results/"),dirs=False,sub_mode=False,img_mode=False)
        encoder.encodes((ig_source),(test_dir+"/encode_results_full_img/"),dirs=False,sub_mode=False,img_mode=False)
    if store:
        source_dir = (test_dir+"/encode_results/results.zarr")
        source_dir = source_dir.lstrip("/")
        zarr_store = zarr.open(source_dir)
        zarr_embeddings = list(zarr_store['features'][:])
        zarr_fname = list(zarr_store['filenames'][:])
        zarr_classes = list(zarr_store['classifications'][:])

        source_dir_full_img = (test_dir+"/encode_results_full_img/results.zarr")
        source_dir_full_img = source_dir_full_img.lstrip("/")
        zarr_store_full_img = zarr.open(source_dir_full_img)
        zarr_embeddings_full_img = list(zarr_store_full_img['features'][:])
        zarr_fname_full_img = list(zarr_store_full_img['filenames'][:])
        # zarr_classes_full_img = list(zarr_store_full_img['classifications'][:])

        hood_instance_full_img = Hood()
        hood_instance_raw = Hood()
        hood_instance_gmsed = Hood()

        # Re-organise metadata -> as the raw ones have classes in them
        zarr_mdata = [
            {
                "path": p,
                "class": c,
            }
            for p, c in zip(zarr_fname, zarr_classes)
        ]

        hood_instance_raw.add(zarr_embeddings,zarr_mdata)
        print(hood_instance_raw.show_info())

        drop_dir = (test_dir+"/db_results/db_raw.zarr")
        drop_dir = drop_dir.lstrip("/")
        hood_instance_raw.dump(path = drop_dir)



        oldfname = ""
        queries = []
        classes = ""
        msedReps = []
        newmdata = []
        
        for i, elem in enumerate(hood_instance_raw.store.metadatas):
            if bypass:
                vname = Path(elem['class']).name
                if i == 0:
                    queries.append(hood_instance_raw.store.embeddings[i])
                    classes = elem['class']
                    oldfname = vname
                    continue
                
                if vname != oldfname or i == len(hood_instance_raw.store.metadatas)-1:
                    normalised_queries = softmax(queries)
                    msedReps.append(np.asarray(MSEDRep(normalised_queries).represent(), dtype=np.float64))
                    newmdata.append({'path':str(Path(elem['class']).parent.parent.parent)+"/"+oldfname+"/"+oldfname+".jpg",'class':classes})
                    queries = []
                    queries.append(hood_instance_raw.store.embeddings[i])
                    classes = elem['class']
                    oldfname = vname
                else:
                    queries.append(hood_instance_raw.store.embeddings[i])
                    classes = classes + ", " + elem['class']

            else:
                fname = Path(elem['path']).name
                fname = re.sub(r'-im.*jpg','', fname)

                if i == 0:
                    queries.append(hood_instance_raw.store.embeddings[i])
                    classes = elem['class']
                    oldfname = fname
                    continue
                
                if fname != oldfname or i == len(hood_instance_raw.store.metadatas)-1:
                    normalised_queries = softmax(queries)
                    msedReps.append(np.asarray(MSEDRep(normalised_queries).represent(), dtype=np.float64))
                    newmdata.append({'path':str(Path(elem['path']).parent.parent.parent)+"/"+oldfname+"/"+oldfname+".jpg",'class':classes})
                    queries = []
                    queries.append(hood_instance_raw.store.embeddings[i])
                    classes = elem['class']
                    oldfname = fname
                else:
                    queries.append(hood_instance_raw.store.embeddings[i])
                    classes = classes + ", " + elem['class']
                    



        hood_instance_gmsed.add(msedReps,newmdata)
        print(hood_instance_gmsed.show_info())
        drop_dir2 = (test_dir+"/db_results/db_gmsed.zarr")
        drop_dir2 = drop_dir2.lstrip("/")
        hood_instance_gmsed.dump(path = drop_dir2)

        if not bypass:
            classes_list = [m["class"] for m in hood_instance_gmsed.store.metadatas]
            # classes_list.append("Person,Person")

            # print(hood_instance_gmsed.get([0]))

            zarr_mdata_full_img = [
                {
                    "path": p,
                    "class": c,
                }
                for p, c in zip(zarr_fname_full_img, classes_list)
            ]

            hood_instance_full_img.add(zarr_embeddings_full_img[:-1],zarr_mdata_full_img)
            print(hood_instance_full_img.show_info())

            # print(hood_instance_full_img.get([647391]))
            # print(hood_instance_full_img.get([0]))

            drop_dir_full_img = (test_dir+"/db_results/db_full_img.zarr")
            drop_dir_full_img = drop_dir_full_img.lstrip("/")
            hood_instance_full_img.dump(path = drop_dir_full_img)

    return