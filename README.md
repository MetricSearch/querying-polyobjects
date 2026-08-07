
# querying-polyobjects
In this proof of concept study, we consider storing small sets of images in a database as entities (gMSED object) in their own right, which can be the target of queries in a symmetric manner.

## Experimental Replication Guide

### Prerequisites & Dataset Setup
-   **Source Collection**: Gather a sufficiently large collection of images (we have used MIR-Flickr 1M (http://press.liacs.nl/mirflickr/mirflickr1m.v3b)).
-   **Pipeline Overview**: The process involves identifying sub-images using YOLO, encoding them into vectors, and performing polyadic search queries. You can perfrom this whole pipeline by:

```
import querying_polyobjects as qp
vals = qp.pipe_it("path_to_images_folder","")
```

Or

```bash
Run
python main.py
```

### Step-by-Step Procedure

#### 1. Sub-image Extraction (YOLO Setup)
*   **Model Selection**: Use the `YOLO11x` image classifier to extract sub-images from every source image in your collection.
    *   *Note*: YOLO is often trained on different datasets than MIR-Flickr due to noise and abstract content, so parameter fine-tuning is required for best results.

#### 2. Fine-Tuning Classification Parameters (Noise Filtering)
To ensure meaningful extraction while discarding extreme or noisy cases:
1.  **Initial Distribution Analysis**: Process all images using YOLO's default parameters first to analyze the classification distribution of your specific dataset. 
2.  **Confidence Threshold**: Set the minimum image classification confidence at approximately `36%` (lower quartile).
    *   Values lower than this increase noise extraction risk; values higher limit complexity coverage.
3.  **Bounding Box Counting**: Limit the number of detected bounding boxes per image to a maximum of `5`.
4.  **Filtering Logic**: Discard any images where YOLO fails to recognise or output valid sub-images (`0` detections).

```
qp.pipe_it("path_to_images_folder","path_to_where_you_want_results_to_go",True,False,False,False)
```

#### 3. Sub-image Extraction & Storage
*   Extract and save the filtered sub-images using your tuned model. 
*   Organise storage so that extracted sub-images share a common directory with their parent source image for easy file path referencing.

```
qp.pipe_it("path_to_images_folder","path_to_where_you_want_results_to_go",True,False,False,False)
```

#### 4. Encoding (Vector Representation)
-   **Model**: Encode all extracted sub-images into `384-dimensional` vectors.
-   **Tooling**: Use the `DINOv2s` encoder model.
    *   References: DINOv2 implementations (`facebookresearch/dinov2`).

```
If you have the extracted images stored on disk
qp.pipe_it("path_to_images_folder","path_to_where_you_want_results_to_go",False,True,False,False)

Otherwise
qp.pipe_it("path_to_images_folder","path_to_where_you_want_results_to_go",True,True,False,False)
```

#### 5. Database Construction & Polyadic Search
*   **Storage Medium**: Store encoded sub-images and their file paths in a vector database compatible with polyadic query search (e.g., utilising `gMSED` logic).
*   **Polyadic Representation**: Create the set representation from the stored sub-image vectors.

```
qp.pipe_it("path_to_images_folder","",False,False,True,False)
```

#### 6. Querying & Validation
1.  Randomly sample sets of these representations as queries.
2.  Perform polyadic search using `gMSED`. Input pre-calculated query vectors into the core formula (refer to your implementation docs).
3.  Compare semantics between the query and its nearest neighbours in the database collection to validate semantic consistency.

```bash
Use showresults.ipynb to analyse and view results
```

## Objectives

- [x] Use YOLO to classify sub-images
    - [x] Read mirFlikr images
    - [x] Retrieve statistics for sub-images
        - [x] Distribution of Number of sub-images per image
        - [x] Distribution of sizes of bounding boxes
        - [x] Distribution of confidences
    - [x] Store the sub-images and the full images in "image-id/full-image.jpg + image-id/classification/sub-images.jpg" structured directories for encoding
- [x] Encode the images using DinoV2
    - Use imcoder?
        - [x] Tweak imcoder to accept our file structure
        - [x] Store the extra meta-data (classifications and relations)
            - Stored when reading images using the directory structure as opposed to "Separate CSV with ids from the image being the index"
- [x] Store the encodings in a database
    - Possibly polyhood?
- [x] Use logsumexp to fix geometric multiplication that can result in floating point errors in gMSED
- [x] Create gMSED represenations for each cluster of sub-images
    - [x] Create the gMSED representations 
- [x] Perform KNN search on the gMSED representations
