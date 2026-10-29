import os
import secrets
import tempfile
from pathlib import Path

import modal

app = modal.App("3d-st-triposr")

# TripoSR se ejecuta en Modal con GPU. Railway conserva solamente la web.
# CUDA 12.6 + PyTorch cu126: torchmcubes requiere CUDA 12 o posterior para compilar su backend GPU.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-devel-ubuntu22.04",
        add_python="3.11",
    )
    .entrypoint([])
    .apt_install("git", "build-essential", "cmake", "gcc-11", "g++-11")
    .env({"CC": "/usr/bin/gcc-11", "CXX": "/usr/bin/g++-11"})
    .run_commands(
        "python -m pip install --no-cache-dir --upgrade pip setuptools wheel",
        "python -m pip install --no-cache-dir torch==2.6.0+cu126 torchvision==0.21.0+cu126 --index-url https://download.pytorch.org/whl/cu126",
    )
    .pip_install(
        "fastapi",
        "python-multipart",
        "Pillow==10.1.0",
        "omegaconf==2.3.0",
        "einops==0.7.0",
        "transformers==4.35.0",
        "trimesh==4.0.5",
        "rembg==2.0.57",
        "huggingface-hub==0.17.3",
        "setuptools>=49.6.0",
        "imageio[ffmpeg]",
        "xatlas==0.0.9",
        "moderngl==5.10.0",
        "numpy==1.26.4",
        "scikit-build-core",
        "pybind11",
        "ninja",
    )
    .run_commands(
        # Ubuntu 22.04 trae CMake 3.22; torchmcubes requiere CMake >= 3.24.
        # Instalar CMake desde PyPI y verificar que esta versión quede en PATH.
        "python -m pip install --no-cache-dir 'cmake>=3.24,<4.0'",
        "cmake --version",
        "git clone --depth 1 https://github.com/VAST-AI-Research/TripoSR.git /opt/TripoSR",
        "cd /opt/TripoSR && python -m pip install --no-build-isolation --no-cache-dir git+https://github.com/tatsy/torchmcubes.git",
        # Verifica que PyTorch y la extensión se importen tras la compilación.
        "python -c \"import torch; import torchmcubes; assert torch.version.cuda.startswith('12.6'), f'PyTorch CUDA inesperado: {torch.version.cuda}'; print('PyTorch CUDA:', torch.version.cuda, '| torchmcubes importado correctamente')\"",
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
    import io
    import sys

    import numpy as np
    import rembg
    import torch
    from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
    from fastapi.responses import Response
    from PIL import Image

    sys.path.insert(0, "/opt/TripoSR")
    from tsr.system import TSR
    from tsr.utils import remove_background as triposr_remove_background
    from tsr.utils import resize_foreground

    api = FastAPI(title="3D-ST TripoSR GPU API")

    # Se carga una sola vez por cada instancia activa de Modal.
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
        if not raw:
            raise HTTPException(status_code=400, detail="La imagen está vacía")
        if len(raw) > 12 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="La imagen supera los 12 MB")

        try:
            pil = Image.open(io.BytesIO(raw)).convert("RGBA")
            if remove_background:
                processed = triposr_remove_background(pil, bg_session)
            else:
                processed = np.array(pil.convert("RGB"))

            processed = resize_foreground(processed, 0.85)
            arr = np.asarray(processed).astype(np.float32) / 255.0
            if arr.ndim == 3 and arr.shape[-1] == 4:
                alpha = arr[:, :, 3:4]
                arr = arr[:, :, :3] * alpha + (1.0 - alpha) * 0.5

            processed_pil = Image.fromarray((arr * 255.0).clip(0, 255).astype(np.uint8))
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
            raise HTTPException(
                status_code=500,
                detail=f"Falló la generación TripoSR: {type(exc).__name__}: {exc}",
            ) from exc

    return api
