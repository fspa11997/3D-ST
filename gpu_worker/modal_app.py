import os
import secrets
import tempfile
from pathlib import Path

import modal

app = modal.App("3d-st-triposr")

# TripoSR's official requirements are installed in a separate GPU container.
# The web app on Railway does NOT install PyTorch or run the model itself.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "build-essential")
    .pip_install(
        "torch",
        "torchvision",
        "fastapi",
        "python-multipart",
        "Pillow==10.1.0",
        "omegaconf==2.3.0",
        "einops==0.7.0",
        "transformers==4.35.0",
        "trimesh==4.0.5",
        "rembg",
        "huggingface-hub<1.0",
        "setuptools>=49.6.0",
        "imageio[ffmpeg]",
        "xatlas==0.0.9",
        "moderngl==5.10.0",
        "numpy<2",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/VAST-AI-Research/TripoSR.git /opt/TripoSR",
        "cd /opt/TripoSR && pip install git+https://github.com/tatsy/torchmcubes.git",
    )
)

@app.function(
    image=image,
    gpu="T4",
    timeout=900,
    scaledown_window=60,
    secrets=[modal.Secret.from_name("triposr-api")],
)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
    from fastapi.responses import Response
    from PIL import Image
    import io
    import sys
    import torch

    sys.path.insert(0, "/opt/TripoSR")
    from tsr.system import TSR
    from tsr.utils import remove_background, resize_foreground
    import numpy as np
    import rembg

    api = FastAPI(title="3D-ST TripoSR GPU API")

    # Load model once per warm worker; Modal scales workers down when idle.
    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(8192)
    model.to("cuda:0")
    bg_session = rembg.new_session()

    @api.get("/health")
    async def health():
        return {"status": "ok", "model": "TripoSR", "device": "cuda:0"}

    @api.post("/generate")
    async def generate(
        image: UploadFile = File(...),
        remove_background: bool = Form(True),
        authorization: str | None = Header(default=None),
    ):
        expected = os.environ.get("GPU_API_TOKEN", "")
        supplied = authorization.removeprefix("Bearer ").strip() if authorization else ""
        if not expected or not supplied or not secrets.compare_digest(supplied, expected):
            raise HTTPException(status_code=401, detail="No autorizado")

        raw = await image.read()
        if not raw or len(raw) > 12 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Imagen vacía o superior a 12 MB")

        try:
            pil = Image.open(io.BytesIO(raw)).convert("RGBA")
            if remove_background:
                processed = remove_background_fn(pil, bg_session)
            else:
                processed = np.array(pil.convert("RGB"))
            processed = resize_foreground(processed, 0.85)
            arr = np.array(processed).astype(np.float32) / 255.0
            if arr.ndim == 3 and arr.shape[-1] == 4:
                arr = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
            processed_pil = Image.fromarray((arr * 255.0).astype(np.uint8))
            with torch.no_grad():
                scene_codes = model([processed_pil], device="cuda:0")
                meshes = model.extract_mesh(scene_codes, True, resolution=256)
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "mesh.obj"
                meshes[0].export(str(out))
                obj = out.read_bytes()
            return Response(
                content=obj,
                media_type="text/plain",
                headers={"Content-Disposition": 'attachment; filename="modelo_3d.obj"'},
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Falló la generación TripoSR: {type(exc).__name__}: {exc}") from exc

    # Avoid name shadowing with the multipart field named remove_background.
    def remove_background_fn(pil, session):
        return remove_background(pil, session)

    return api
