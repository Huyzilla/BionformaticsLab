from routers.alive_classification import alive_classification_router
from routers.upload_normal_image import upload_normal_image_router
from routers.cell_counting import cell_counting_router
from routers.submit_image import submit_router
from routers.analyze_router import analyze_router
from routers.request_status_router import request_status_router
from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,  
    allow_methods=["*"], 
    allow_headers=["*"], 
)
app.include_router(alive_classification_router)
app.include_router(upload_normal_image_router)
app.include_router(cell_counting_router)
app.include_router(submit_router)
app.include_router(analyze_router)
app.include_router(request_status_router)

if "__name__" == "__main__":
   import uvicorn
   uvicorn.run(app, host="0.0.0.0", port = 8080)
#    ssl_keyfile = "key.pem", ssl_certfile = "cert.pem"