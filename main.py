"""Python FastAPI Auth0 integration example
"""

from datetime import datetime
from io import BytesIO
from os import getcwd
from os.path import exists
from pathlib import Path
from typing import List

import joblib
import numpy as np
from PIL import Image
from fastapi import Depends, FastAPI, Response, status, File, UploadFile
from fastapi.security import HTTPBearer
from starlette.responses import RedirectResponse, FileResponse
from starlette.staticfiles import StaticFiles

from feature_extractor import FeatureExtractor
from utils import VerifyToken

# Scheme for the Authorization header
token_auth_scheme = HTTPBearer()

app_desc = """<h2>Try this app by uploading any image with `/similarity/search`</h2>"""

# Creates app instance
app = FastAPI(title='AliSoft Image Search Engine V1.0', description=app_desc)
app.mount("/static", StaticFiles(directory="./static"), name="static")

fe = FeatureExtractor()
features = []
img_paths = []
count = 0
use_authorization = False

features_path = "features.npy"
images_path = "images.npy"

if (exists(features_path)):
    # Load the model from the file
    features = joblib.load(features_path)
    img_paths = joblib.load(images_path)
else:
    # count = 15461 # furniture
    count = 13233  # faces
    i = 0
    for feature_path in Path("./static/feature").glob("*.npy"):
        features.append(np.load(feature_path))
        img_paths.append(Path("./static/img") / (feature_path.stem + ".jpg"))
        i += 1
        print(str(round(i * 100 / count, 2)) + "%")
    # Save the model as a pickle in a file
    if len(features) > 0:
        joblib.dump(features, features_path)
        joblib.dump(img_paths, images_path)


def read_image_file(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image


@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")


@app.get("/my-endpoint", include_in_schema=False)
def example_function():
    """
    Documentation for my enpoint. Insert some images

    1) This online image works:

    ![This image works](https://upload.wikimedia.org/wikipedia/commons/0/08/STockholmspanorama_1928b.jpg)

    2) This local image doesn't work:

    ![This image doesn't work](static/img/images (688).jpg)

    3) This local image served by the api works if the link is to the public IP:

    ![This image works](http://127.0.0.1:8000/static/img/images (688).jpg)

    4) This local image served by the api doesn't work because when specified as localhost:

    ![This image doesn't work](http://127.0.0.1:8000/img/example-photo.jpg)

    ![This image doesn't work](http://localhost:8000/img/example-photo.jpg)

    """
    return {"This is my endpoint"}


# An endpoint to serve images for documentation
@app.get("/img/example-photo.jpg", include_in_schema=False)
async def read_image():
    return FileResponse("static/img/images (688).jpg")


@app.post("/similarity/search", responses={200: {"description": "A picture of a vector image.", "content": {
    "image/jpeg": {"example": "No example available. Just imagine a picture of a vector image."}}}}, include_in_schema=True)
async def similarity_search_api(response: Response, token: str = Depends(token_auth_scheme) if use_authorization else "", file: UploadFile = File(...)):
    if use_authorization:
        result = VerifyToken(token.credentials, scopes="PortalApi").verify()
        if result.get("status"):
            response.status_code = status.HTTP_400_BAD_REQUEST
            return result

    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"

    if len(features) > 0:
        # Save query image
        img = read_image_file(await file.read())
        # img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)

        # Run search
        query = fe.extract(img)
        dists = np.linalg.norm(features - query, axis=1)  # L2 distances to features

        # dists = []
        # n = len(features)
        # for feature in features:
        #     sub = feature-query
        #     sum = 0
        #     for x in sub:
        #         sum += x*x
        #     dists.append(sum)

        size = min(12, len(features))  # Top 12 results
        ids = np.argsort(dists)[:size]
        scores = [(dists[id], img_paths[id]) for id in ids]

        response = []
        for res in scores:
            file_path = res[1]
            # if os.path.exists(file_path):
            #     return FileResponse(file_path, media_type="image/jpeg", filename="vector_image_for_you.jpg")
            # path = getcwd() + "/" + str(file_path)
            # resp = {"path": path, "error": f"{res[0]:0.6f}"}
            # resp = {"path": file_path, "error": f"{res[0]:0.6f}", "file": FileResponse(file_path, media_type="image/jpeg")}
            # resp = {"file": FileResponse(file_path, media_type="image/jpeg")}
            resp = {"path": file_path, "error": f"{res[0]:0.6f}"}
            response.append(resp)

        return response
    return {"status": "invalid request"}


@app.post("/similarity/add_image", include_in_schema=True)
async def similarity_add_image_api(response: Response, token: str = Depends(token_auth_scheme) if use_authorization else "", file: UploadFile = File(...)):
    if use_authorization:
        result = VerifyToken(token.credentials, scopes="PortalApi").verify()
        if result.get("status"):
            response.status_code = status.HTTP_400_BAD_REQUEST
            return result

    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"

    img_path = Path(file.filename)
    print(img_path)  # e.g., ./static/img/xxx.jpg
    uploaded_img_path = "static/img/" + file.filename
    if (not exists(uploaded_img_path)):
        # if (True):
        # Save query image
        # img = Image.open(file.stream)  # PIL image
        img = read_image_file(await file.read())
        img.save(uploaded_img_path)

        # save image features
        feature = fe.extract(img=Image.open(uploaded_img_path))
        feature_path = Path("./static/feature") / (img_path.stem + ".npy")  # e.g., ./static/feature/xxx.npy
        np.save(feature_path, feature)

        features.append(np.load(feature_path))
        img_paths.append(uploaded_img_path)
        return {"status": "Image added successfully..."}

    return {"status": "Image exists."}


@app.post("/similarity/add_images", include_in_schema=True)
async def similarity_add_images_api(response: Response, token: str = Depends(token_auth_scheme) if use_authorization else "",
                             files: List[UploadFile] = File(...)):
    if use_authorization:
        result = VerifyToken(token.credentials, scopes="PortalApi").verify()
        if result.get("status"):
            response.status_code = status.HTTP_400_BAD_REQUEST
            return result

    for file in files:
        extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
        if extension:
            img_path = Path(file.filename)
            print(img_path)  # e.g., ./static/img/xxx.jpg
            uploaded_img_path = "static/img/" + file.filename
            if (not exists(uploaded_img_path)):
                # if (True):
                # Save query image
                # img = Image.open(file.stream)  # PIL image
                img = read_image_file(await file.read())
                img.save(uploaded_img_path)

                # save image features
                feature = fe.extract(img=Image.open(uploaded_img_path))
                feature_path = Path("./static/feature") / (img_path.stem + ".npy")  # e.g., ./static/feature/xxx.npy
                np.save(feature_path, feature)

                features.append(np.load(feature_path))
                img_paths.append(uploaded_img_path)
                # return {"status": "Image added successfully..."}

            # return {"status": "Image exists."}

    return {"status": "Images added successfully..."}


@app.post("/similarity/save_model", include_in_schema=True)
async def similarity_save_model_api(response: Response, token: str = Depends(token_auth_scheme) if use_authorization else ""):
    if use_authorization:
        result = VerifyToken(token.credentials, scopes="PortalApi").verify()
        if result.get("status"):
            response.status_code = status.HTTP_400_BAD_REQUEST
            return result

    joblib.dump(features, features_path)
    joblib.dump(img_paths, images_path)

    return {"status": "Model Saved successfully..."}
