"""
Reference : https://huggingface.co/tencent/Hunyuan3D-2mini
"""
import io
import os
import platform
import random
import sys
import tempfile
import time
import threading
import uuid
import zipfile
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from services.generators.base import (
    BaseGenerator,
    smooth_progress,
    GenerationCancelled,
    pick_device,
    release_device_memory,
)

_HF_REPO_ID       = "tencent/Hunyuan3D-2mini"
_SUBFOLDER        = "hunyuan3d-dit-v2-mini"
_GITHUB_ZIP       = "https://github.com/Tencent/Hunyuan3D-2/archive/refs/heads/main.zip"
_PAINT_HF_REPO    = "tencent/Hunyuan3D-2"
_PAINT_SUBFOLDER  = "hunyuan3d-paint-v2-0-turbo"

# huggingface_hub spawns parallel download workers via multiprocessing. On
# macOS that raced with our subprocess setup and crashed the runner mid-fetch
# ("resource_tracker: leaked semaphore objects"). Force serial downloads on
# Darwin; other platforms keep the default for speed.
_HF_DOWNLOAD_KWARGS = {"max_workers": 1} if platform.system() == "Darwin" else {}


def _log(msg: str) -> None:
    """Write a status line to stderr.

    Generator runs as a JSON-RPC subprocess where stdout is the protocol
    channel — printing diagnostics there shows up as 'bad JSON' in the
    parent's read loop. Stderr is forwarded as plain text instead.
    """
    print(msg, file=sys.stderr, flush=True)


class Hunyuan3DMiniGenerator(BaseGenerator):
    MODEL_ID     = "hunyuan3d-mini"
    DISPLAY_NAME = "Hunyuan3D 2 Mini"
    VRAM_GB      = 6

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def is_downloaded(self) -> bool:
        subfolder = self.download_check if self.download_check else _SUBFOLDER
        model_dir = self.model_dir / subfolder
        return model_dir.exists() and (model_dir / "model.fp16.safetensors").exists()

    def load(self) -> None:
        if self._model is not None:
            return

        if not self.is_downloaded():
            self._download_weights()

        self._ensure_hy3dgen()

        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        device, dtype = pick_device()
        self._device = device

        subfolder = self.download_check if self.download_check else _SUBFOLDER
        _log(f"[Hunyuan3DMiniGenerator] Loading pipeline from {self.model_dir} (subfolder={subfolder})…")
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            str(self.model_dir),
            subfolder=subfolder,
            use_safetensors=True,
            device=device,
            dtype=dtype,
        )
        self._model = pipeline
        _log(f"[Hunyuan3DMiniGenerator] Loaded on {device}.")

    def unload(self) -> None:
        super().unload()
        release_device_memory(getattr(self, "_device", None))

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def generate(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        import torch

        num_steps        = int(params.get("num_inference_steps", 30))
        vert_count       = int(params.get("vertex_count", 0))
        enable_texture   = bool(params.get("enable_texture", False))
        # On Apple Silicon with 8–16 GB unified memory the default octree
        # 380 reliably tips MPS into jetsam SIGKILL during volume decode.
        # 256 still produces usable meshes and lands well under the limit.
        # Users with more headroom can override via the params UI.
        _octree_default  = 256 if platform.system() == "Darwin" else 380
        octree_res       = int(params.get("octree_resolution", _octree_default))
        guidance_scale   = float(params.get("guidance_scale", 5.5))
        seed             = int(params.get("seed", -1))
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        self._report(progress_cb, 5, "Removing background…")
        image = self._preprocess(image_bytes)
        self._check_cancelled(cancel_event)

        # Shape generation is a single pipeline call internally composed of
        # (1) flow-matching diffusion (~num_steps fast steps) and (2) sparse
        # volume decoding (slow, dominates wall time on CPU/MPS). hy3dgen
        # writes its own 'Volume Decoding NN%' tqdm to stderr — the label
        # below tells users the API-level bar plateaus while that runs and
        # is not stuck. We split the smooth_progress into a short diffusion
        # window (12→30) and a long decoding window (30→shape_end) and run
        # them sequentially in the background thread.
        shape_end       = 70 if enable_texture else 82
        diffusion_end   = min(30, shape_end)
        diffusion_label = "Diffusion sampling…"
        decode_label    = "Volume decoding (this is the long step)…"
        on_macos        = platform.system() == "Darwin"
        if on_macos:
            decode_label = "Volume decoding on MPS/CPU — may take several minutes…"

        self._report(progress_cb, 12, diffusion_label)
        stop_evt = threading.Event()
        if progress_cb:
            def _shape_progress() -> None:
                # Phase 1 — diffusion (short). 'Diffusion sampling…' fills
                # 12→28 quickly; once the underlying model starts volume
                # decoding the smooth_progress thread will already be near
                # diffusion_end, so we transition to the decoding label.
                inner_stop = threading.Event()
                phase1 = threading.Thread(
                    target=smooth_progress,
                    args=(progress_cb, 12, diffusion_end, diffusion_label, inner_stop, 1.5),
                    daemon=True,
                )
                phase1.start()
                # Heuristic: diffusion typically finishes within
                # ~num_steps * 0.5 s on GPU, ~2-3 s on CPU. Wait either
                # for the outer stop event (real generation finished) or
                # for that estimate to elapse, then switch label.
                est = max(8.0, num_steps * 1.0)
                stop_evt.wait(est)
                inner_stop.set()
                phase1.join(timeout=2.0)
                if stop_evt.is_set():
                    return
                self._report(progress_cb, diffusion_end, decode_label)
                smooth_progress(
                    progress_cb, diffusion_end, shape_end, decode_label, stop_evt
                )

            t = threading.Thread(target=_shape_progress, daemon=True)
            t.start()

        try:
            with torch.no_grad():
                import torch
                generator = torch.Generator().manual_seed(seed)
                outputs = self._model(
                    image=image,
                    num_inference_steps=num_steps,
                    octree_resolution=octree_res,
                    guidance_scale=guidance_scale,
                    num_chunks=4000,
                    generator=generator,
                    output_type="trimesh",
                )
            mesh = outputs[0]
        finally:
            stop_evt.set()

        self._check_cancelled(cancel_event)

        if enable_texture:
            if platform.system() == "Darwin":
                # custom_rasterizer / differentiable_renderer are CUDA-only C++
                # extensions; they don't build on macOS at all. Bail with an
                # actionable message instead of trying and crashing later.
                raise RuntimeError(
                    "Texture generation requires CUDA — disable the Texture "
                    "toggle on macOS. Shape-only generation is supported."
                )
            self._report(progress_cb, 72, "Freeing VRAM for texture model…")
            self._model = None
            release_device_memory(getattr(self, "_device", None))

            self._check_cancelled(cancel_event)
            mesh = self._run_texture(mesh, image, progress_cb)
        else:
            if vert_count > 0 and hasattr(mesh, "vertices") and len(mesh.vertices) > vert_count:
                self._report(progress_cb, 85, "Optimizing mesh…")
                mesh = self._decimate(mesh, vert_count)

        self._report(progress_cb, 96, "Exporting GLB…")
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.glb"
        path = self.outputs_dir / name
        mesh.export(str(path))

        self._report(progress_cb, 100, "Done")
        return path

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _preprocess(self, image_bytes: bytes) -> Image.Image:
        import rembg
        img = Image.open(io.BytesIO(image_bytes))
        try:
            return rembg.remove(img).convert("RGBA")
        except Exception:
            # cuDNN/CUDA incompatibility — fall back to CPU
            session = rembg.new_session("u2net", providers=["CPUExecutionProvider"])
            return rembg.remove(img, session=session).convert("RGBA")

    def _run_texture(self, mesh, image: "Image.Image", progress_cb=None):
        import torch

        self._check_texgen_extensions()

        self._report(progress_cb, 73, "Preparing texture model…")
        self._ensure_paint_weights()

        self._report(progress_cb, 78, "Loading texture model…")
        from hy3dgen.texgen import Hunyuan3DPaintPipeline

        paint_dir = self.model_dir / "_paint_weights"
        paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
            str(paint_dir), subfolder=_PAINT_SUBFOLDER
        )

        from hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender
        paint_pipeline.config.render_size  = 1024
        paint_pipeline.config.texture_size = 1024
        paint_pipeline.render = MeshRender(default_resolution=1024, texture_size=1024)

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        try:
            image.save(tmp.name)
            tmp.close()

            self._report(progress_cb, 83, "Generating textures…")
            with torch.no_grad():
                result = paint_pipeline(mesh, image=tmp.name)
        finally:
            os.unlink(tmp.name)
            del paint_pipeline
            release_device_memory(getattr(self, "_device", None))

        return result[0] if isinstance(result, (list, tuple)) else result

    def _check_texgen_extensions(self) -> None:
        try:
            from hy3dgen.texgen import Hunyuan3DPaintPipeline  # noqa: F401
        except (ImportError, OSError) as exc:
            base = self.model_dir / "_hy3dgen" / "hy3dgen" / "texgen"
            raise RuntimeError(
                "C++ extensions for texture generation are not compiled.\n"
                "Build them with:\n\n"
                f"  cd \"{base / 'custom_rasterizer'}\"\n"
                f"  python setup.py install\n\n"
                f"  cd \"{base / 'differentiable_renderer'}\"\n"
                f"  python setup.py install\n\n"
                f"Original error: {exc}"
            ) from exc

    def _ensure_paint_weights(self) -> None:
        paint_dir = self.model_dir / "_paint_weights"
        if (paint_dir / _PAINT_SUBFOLDER).exists() and (paint_dir / "hunyuan3d-delight-v2-0").exists():
            return

        from huggingface_hub import snapshot_download
        _log(f"[Hunyuan3DMiniGenerator] Downloading paint model ({_PAINT_HF_REPO})…")
        snapshot_download(
            repo_id=_PAINT_HF_REPO,
            local_dir=str(paint_dir),
            ignore_patterns=[
                "hunyuan3d-dit-v2-0/**",
                "hunyuan3d-dit-v2-0-fast/**",
                "hunyuan3d-dit-v2-0-turbo/**",
                "hunyuan3d-vae-v2-0/**",
                "hunyuan3d-vae-v2-0-turbo/**",
                "hunyuan3d-vae-v2-0-withencoder/**",
                "hunyuan3d-paint-v2-0/**",
                "assets/**",
                "*.md", "LICENSE", "NOTICE", ".gitattributes",
            ],
            **_HF_DOWNLOAD_KWARGS,
        )
        _log("[Hunyuan3DMiniGenerator] Paint model downloaded.")

    def _decimate(self, mesh, target_vertices: int):
        target_faces = max(4, target_vertices * 2)
        try:
            return mesh.simplify_quadric_decimation(target_faces)
        except Exception as exc:
            _log(f"[Hunyuan3DMiniGenerator] Decimation skipped: {exc}")
            return mesh

    def _download_weights(self) -> None:
        from huggingface_hub import snapshot_download
        _log(f"[Hunyuan3DMiniGenerator] Downloading {_HF_REPO_ID} (base variant)…")
        snapshot_download(
            repo_id=_HF_REPO_ID,
            local_dir=str(self.model_dir),
            ignore_patterns=[
                "hunyuan3d-dit-v2-mini-fast/**",
                "hunyuan3d-dit-v2-mini-turbo/**",
                "hunyuan3d-vae-v2-mini-turbo/**",
                "hunyuan3d-vae-v2-mini-withencoder/**",
                "*.md", "LICENSE", "NOTICE", ".gitattributes",
            ],
            **_HF_DOWNLOAD_KWARGS,
        )
        _log("[Hunyuan3DMiniGenerator] Download complete.")

    def _ensure_hy3dgen(self) -> None:
        try:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline  # noqa: F401
            return
        except ImportError:
            pass

        src_dir = self.model_dir / "_hy3dgen"
        if not (src_dir / "hy3dgen").exists():
            self._download_hy3dgen(src_dir)

        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        try:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                f"hy3dgen still not importable after extraction to {src_dir}.\n"
                f"Check the folder contents.\n{exc}"
            ) from exc

    def _download_hy3dgen(self, dest: Path) -> None:
        import urllib.request

        dest.mkdir(parents=True, exist_ok=True)
        _log("[Hunyuan3DMiniGenerator] Downloading hy3dgen source from GitHub…")
        with urllib.request.urlopen(_GITHUB_ZIP, timeout=180) as resp:
            data = resp.read()
        _log("[Hunyuan3DMiniGenerator] Extracting hy3dgen…")

        prefix = "Hunyuan3D-2-main/hy3dgen/"
        strip  = "Hunyuan3D-2-main/"

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for member in zf.namelist():
                if not member.startswith(prefix):
                    continue
                rel    = member[len(strip):]
                target = dest / rel
                if member.endswith("/"):
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(zf.read(member))

        _log(f"[Hunyuan3DMiniGenerator] hy3dgen extracted to {dest}.")

    @classmethod
    def params_schema(cls) -> list:
        return [
            {
                "id":      "num_inference_steps",
                "label":   "Quality",
                "type":    "select",
                "default": 30,
                "options": [
                    {"value": 10, "label": "Fast"},
                    {"value": 30, "label": "Balanced"},
                    {"value": 50, "label": "High"},
                ],
                "tooltip": "Number of diffusion steps. More steps = better quality but slower.",
            },
            {
                "id":      "octree_resolution",
                "label":   "Mesh Resolution",
                "type":    "select",
                "default": 380,
                "options": [
                    {"value": 256, "label": "Low"},
                    {"value": 380, "label": "Medium"},
                    {"value": 512, "label": "High"},
                ],
                "tooltip": "Octree resolution for mesh reconstruction. Higher = more detail but slower and more VRAM.",
            },
            {
                "id":      "guidance_scale",
                "label":   "Guidance Scale",
                "type":    "float",
                "default": 5.5,
                "min":     1.0,
                "max":     10.0,
                "step":    0.5,
                "tooltip": "Classifier-free guidance strength. Higher = closer to the input image.",
            },
            {
                "id":      "seed",
                "label":   "Seed",
                "type":    "int",
                "default": -1,
                "min":     0,
                "max":     2147483647,
                "tooltip": "Seed for reproducibility. Click shuffle for a random seed.",
            },
        ]
