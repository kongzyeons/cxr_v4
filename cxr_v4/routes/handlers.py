from fastapi import FastAPI, Depends, Header, HTTPException, Response
from .api import pneumonia, cardiomegaly,tuberculosis,Aortic_enlargement


description = """
Radiological Diagnosis in Lung Disease.

* **Pneumonia** 
* **Cardiomegaly** 
* **Tuberculosis**
"""


app = FastAPI(title = "CXR_V4",
              description = description,
              version = "0.0.4")


app.include_router(
    pneumonia.router,
    prefix="/api/v1",
    tags=["Pneumonia"],
    responses={404: {"message": "Not found"}},
)

app.include_router(
    cardiomegaly.router,
    prefix="/api/v1",
    tags=["Cardiomegaly"],
    responses={404: {"message": "Not found"}},
)


app.include_router(
    tuberculosis.router,
    prefix="/api/v1",
    tags=["Tuberculosis"],
    responses={404: {"message": "Not found"}},
)

app.include_router(
    Aortic_enlargement.router,
    prefix="/api/v1",
    tags=["Aortic_enlargement"],
    responses={404: {"message": "Not found"}},
)