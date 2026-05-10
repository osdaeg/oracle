import io
import os
import re
import json
import time
import base64
import logging
import httpx
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from typing import Any

# ── Nuevo SDK unificado de Google ──────────────────────────────────────────────
from google import genai
from google.genai import types as genai_types

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("oracle")

# ── Config ─────────────────────────────────────────────────────────────────────
GEMINI_API_KEY     = os.environ["GEMINI_API_KEY"]
STABLE_HORDE_KEY   = os.environ.get("STABLE_HORDE_KEY", "0000000000").strip()
OLLAMA_URL         = os.environ.get("OLLAMA_URL", "http://192.168.88.100:11434")
OLLAMA_MODEL       = os.environ.get("OLLAMA_MODEL", "gemma3:1b")
QUEUE_FILE         = Path("/app/data/pending_queue.json")
QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)

# ── Cliente Gemini ─────────────────────────────────────────────────────────────
client = genai.Client(api_key=GEMINI_API_KEY)

# ── Model autodiscovery ────────────────────────────────────────────────────────
PREFERRED_TEXT_KEYWORDS = [
    "gemini-3.1-flash",
    "gemini-3-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-3.1-pro",
    "gemini-3-pro",
    "gemini-2.5-pro",
    "gemini-2.0-pro",
    "gemini-1.5-pro",
    "gemini-1.0-pro",
    "gemini-pro",
]

PREFERRED_IMAGE_KEYWORDS = [
    "imagen-4",
    "nano-banana",
    "imagen-3",
    "imagen-2",
    "imagen",
]

TEXT_EXCLUSIONS = (
    "imagen", "image-generation", "image-preview", "-image",
    "nano-banana", "veo", "tts", "audio", "native-audio",
    "embedding", "aqa", "computer-use", "robotics", "deep-research",
    "customtools",
)

def _is_text_model(model_name: str) -> bool:
    name = model_name.lower()
    return not any(x in name for x in TEXT_EXCLUSIONS)

def _keyword_matches(model_name: str, keyword: str) -> bool:
    name = model_name.lower().split("/")[-1]
    kw   = keyword.lower()
    return name == kw or name.startswith(kw + "-") or name.endswith("-" + kw)

def _get_methods(m) -> list:
    for attr in ("supported_generation_methods", "supported_actions", "actions"):
        val = getattr(m, attr, None)
        if val:
            return list(val)
    name = m.name.lower()
    if not _is_text_model(name):
        return []
    if "gemini" in name or "gemma" in name:
        return ["generateContent"]
    return []

def discover_models():
    all_models = list(client.models.list())
    log.info("All available models: %s", [m.name for m in all_models])

    # ── Text ────────────────────────────────────────────────────────────────
    text_candidates = []
    seen = set()
    for keyword in PREFERRED_TEXT_KEYWORDS:
        for m in all_models:
            if not _is_text_model(m.name):
                continue
            methods = _get_methods(m)
            if _keyword_matches(m.name, keyword) and "generateContent" in methods:
                if m.name not in seen:
                    text_candidates.append(m.name)
                    seen.add(m.name)

    for m in all_models:
        if m.name not in seen and _is_text_model(m.name):
            methods = _get_methods(m)
            if "generateContent" in methods:
                text_candidates.append(m.name)
                seen.add(m.name)

    if not text_candidates:
        log.warning("Autodiscovery returned no text models — using hardcoded fallback")
        text_candidates = [m.name for m in all_models if "flash" in m.name.lower() and "gemini" in m.name.lower()]
        if not text_candidates:
            text_candidates = ["models/gemini-2.5-flash"]

    # ── Image ────────────────────────────────────────────────────────────────
    image_model = None
    for keyword in PREFERRED_IMAGE_KEYWORDS:
        for m in all_models:
            if keyword in m.name.lower():
                image_model = m.name
                break
        if image_model:
            break

    log.info("Selected TEXT model  : %s", text_candidates[0])
    log.info("Selected IMAGE model : %s", image_model)
    return text_candidates[0], image_model, text_candidates

TEXT_MODEL, IMAGE_MODEL, text_candidates_global = discover_models()

# ── FastAPI ────────────────────────────────────────────────────────────────────
app = FastAPI(title="Oracle — AI Gateway")

# ── Modelos de request/response ────────────────────────────────────────────────
class QueryRequest(BaseModel):
    prompt: str
    schema: dict | None = Field(default=None, description="Estructura JSON esperada en la respuesta")
    ai: str = Field(default="gemini", description="Provider: gemini (default) | ollama | auto | image")
    grounding: bool = Field(default=False, description="Habilitar Google Search para datos en tiempo real")
    source: str = Field(default="unknown", description="Identificador del servicio que hace el pedido")

class QueryResponse(BaseModel):
    status: str           # "ok" | "error"
    ai: str
    source: str
    data: Any | None = None
    error: str | None = None
    model_used: str | None = None
    grounding_used: bool = False
    sources: list[str] = []
    timestamp: str

class AnalyzeResponse(BaseModel):
    status: str           # "ok" | "error"
    source: str
    data: Any | None = None
    error: str | None = None
    model_used: str | None = None
    filename: str | None = None
    timestamp: str

# ── Rotación de modelos ────────────────────────────────────────────────────────
_TEXT_MODEL_QUEUE: list[str] = []

def _build_model_queue(primary: str) -> list[str]:
    queue = list(text_candidates_global) if text_candidates_global else [primary]
    if primary in queue:
        queue.remove(primary)
    queue.insert(0, primary)
    return queue

def _parse_retry_delay(err_str: str) -> int:
    m = re.search(r"retry.{0,20}?(\d+)s", err_str, re.IGNORECASE)
    if m:
        return int(m.group(1)) + 2
    return 0

def call_text_model(prompt: str, grounding: bool = False, retries: int = 5) -> tuple[str, str, list[str]]:
    """Llama al modelo de texto. Retorna (texto, modelo_usado, fuentes)."""
    global _TEXT_MODEL_QUEUE
    if not _TEXT_MODEL_QUEUE:
        _TEXT_MODEL_QUEUE = _build_model_queue(TEXT_MODEL)

    last_error = None
    for model_name in list(_TEXT_MODEL_QUEUE):
        model_short = model_name.split("/")[-1]
        for attempt in range(retries):
            try:
                config = None
                if grounding:
                    config = genai_types.GenerateContentConfig(
                        tools=[genai_types.Tool(
                            google_search=genai_types.GoogleSearch()
                        )]
                    )
                response = client.models.generate_content(
                    model=model_short,
                    contents=prompt,
                    config=config,
                )
                log.info("[oracle] Text model used: %s (grounding=%s)", model_short, grounding)

                # Extraer URLs de fuentes si hay grounding metadata
                sources = []
                try:
                    gm = response.candidates[0].grounding_metadata
                    if gm and hasattr(gm, "grounding_chunks"):
                        for chunk in gm.grounding_chunks:
                            if hasattr(chunk, "web") and chunk.web and chunk.web.uri:
                                sources.append(chunk.web.uri)
                except Exception:
                    pass

                return response.text, model_short, sources
            except Exception as e:
                err_str = str(e)
                is_daily     = "per_day" in err_str.lower() or "GenerateRequestsPerDay" in err_str
                is_not_found = "404" in err_str or "NOT_FOUND" in err_str or "not found" in err_str.lower()
                is_quota     = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str

                if is_daily or is_not_found:
                    reason = "daily quota exhausted" if is_daily else "model not found (404)"
                    log.warning("Rotating away from %s: %s", model_short, reason)
                    _TEXT_MODEL_QUEUE = [m for m in _TEXT_MODEL_QUEUE if m != model_name]
                    last_error = e
                    break

                retry_delay = _parse_retry_delay(err_str)
                wait = max(retry_delay, (2 ** attempt) * (5 if is_quota else 1))

                if attempt < retries - 1:
                    log.warning("Text attempt %d/%d (%s) failed: %s. Retry in %ds...",
                                attempt+1, retries, model_short, e, wait)
                    time.sleep(wait)
                else:
                    last_error = e
                    break

    raise RuntimeError(f"All text models failed. Last error: {last_error}")

def call_ollama_model(prompt: str) -> tuple[str, str]:
    """Llama a Ollama. Retorna (texto, modelo_usado)."""
    url = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    try:
        resp = httpx.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        text = data.get("response", "")
        if not text:
            raise RuntimeError("Ollama returned empty response")
        log.info("[oracle] Ollama model used: %s", OLLAMA_MODEL)
        return text, OLLAMA_MODEL
    except Exception as e:
        raise RuntimeError(f"Ollama failed: {e}")

def clean_json(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    return text.strip()

# ── Providers de imagen ────────────────────────────────────────────────────────
def _try_gemini_native_image(prompt_img: str) -> bytes | None:
    candidates = [
        "gemini-2.0-flash-exp-image-generation",
        "gemini-2.5-flash-image",
        "gemini-3-pro-image-preview",
    ]
    for model_name in candidates:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt_img,
                config=genai_types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                ),
            )
            for part in response.candidates[0].content.parts:
                if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                    data = part.inline_data.data
                    img_bytes = base64.b64decode(data) if isinstance(data, str) else data
                    log.info("Image via Gemini native (%s).", model_name)
                    return img_bytes
        except Exception as e:
            log.warning("Gemini native image (%s) failed: %s", model_name, e)
    return None

def _try_pollinations(prompt: str) -> bytes | None:
    # Desactivado temporalmente — devuelve error 500 consistentemente
    return None
    import urllib.parse
    prompt_safe = prompt.encode("ascii", "ignore").decode()
    encoded = urllib.parse.quote(prompt_safe)
    url = f"https://image.pollinations.ai/prompt/{encoded}?width=512&height=768&nologo=true&seed={abs(hash(prompt)) % 9999}"
    for attempt in range(3):
        try:
            log.info("Pollinations.ai attempt %d/3...", attempt + 1)
            resp = httpx.get(url, timeout=90, follow_redirects=True)
            resp.raise_for_status()
            if resp.headers.get("content-type", "").startswith("image/"):
                log.info("Image from Pollinations.ai (%d bytes).", len(resp.content))
                return resp.content
        except Exception as e:
            log.warning("Pollinations.ai attempt %d failed: %s", attempt + 1, e)
            if attempt < 2:
                time.sleep(5)
    return None

def _try_stable_horde(prompt: str) -> bytes | None:
    # Desactivado temporalmente — calidad insuficiente
    # Para reactivar, descomentar el bloque
    return None
    # HORDE_URL = "https://stablehorde.net/api/v2"
    # HEADERS   = {"Content-Type": "application/json", "apikey": STABLE_HORDE_KEY}
    # ...

def generate_image(prompt: str) -> bytes | None:
    """Intenta generar imagen con todos los providers disponibles."""
    # 1. Gemini nativo
    img = _try_gemini_native_image(prompt)
    if img:
        return img
    # 2. Pollinations
    img = _try_pollinations(prompt)
    if img:
        return img
    # 3. Stable Horde (desactivado)
    img = _try_stable_horde(prompt)
    if img:
        return img
    return None

# ── Análisis de PDF ───────────────────────────────────────────────────────────
def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extrae texto de un PDF usando pypdf."""
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())
        result = "\n\n".join(pages)
        if not result.strip():
            raise RuntimeError("El PDF no contiene texto extraíble (puede ser escaneado).")
        log.info("PDF text extracted: %d chars from %d pages.", len(result), len(reader.pages))
        return result
    except ImportError:
        raise RuntimeError("pypdf no está instalado. Agregá 'pypdf' a requirements.txt.")

def _build_analyze_prompt(pdf_text: str, prompt: str, schema: dict | None) -> str:
    full = f"{prompt}\n\nCONTENIDO DEL DOCUMENTO:\n{pdf_text[:4000]}"
    if schema:
        schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
        full += (
            "\n\nResponde ÚNICAMENTE con un JSON con esta estructura exacta, "
            "sin texto adicional ni marcas de código:\n" + schema_str
        )
    return full

def analyze_pdf_with_ollama(pdf_bytes: bytes, prompt: str, schema: dict | None) -> tuple[str, str]:
    """Extrae texto del PDF y lo analiza con Ollama."""
    pdf_text  = extract_pdf_text(pdf_bytes)
    full_prompt = _build_analyze_prompt(pdf_text, prompt, schema)
    text, model = call_ollama_model(full_prompt)
    return text, model

def analyze_pdf_with_gemini(pdf_bytes: bytes, prompt: str, schema: dict | None) -> tuple[str, str]:
    """Envía un PDF a Gemini de forma nativa (sin extracción de texto)."""
    global _TEXT_MODEL_QUEUE
    if not _TEXT_MODEL_QUEUE:
        _TEXT_MODEL_QUEUE = _build_model_queue(TEXT_MODEL)

    full_prompt = prompt
    if schema:
        schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
        full_prompt += (
            "\n\nResponde ÚNICAMENTE con un JSON con esta estructura exacta, "
            "sin texto adicional ni marcas de código:\n" + schema_str
        )

    last_error = None
    for model_name in list(_TEXT_MODEL_QUEUE):
        model_short = model_name.split("/")[-1]
        try:
            response = client.models.generate_content(
                model=model_short,
                contents=[
                    genai_types.Part.from_bytes(
                        data=pdf_bytes,
                        mime_type="application/pdf",
                    ),
                    full_prompt,
                ],
            )
            log.info("[oracle] PDF analyzed with Gemini model: %s", model_short)
            return response.text, model_short
        except Exception as e:
            err_str = str(e)
            is_daily     = "per_day" in err_str.lower() or "GenerateRequestsPerDay" in err_str
            is_not_found = "404" in err_str or "NOT_FOUND" in err_str or "not found" in err_str.lower()
            if is_daily or is_not_found:
                reason = "daily quota exhausted" if is_daily else "model not found (404)"
                log.warning("Rotating away from %s: %s", model_short, reason)
                _TEXT_MODEL_QUEUE = [m for m in _TEXT_MODEL_QUEUE if m != model_name]
                last_error = e
                continue
            last_error = e
            log.warning("PDF analysis failed (%s): %s", model_short, e)

    raise RuntimeError(f"All Gemini models failed for PDF analysis. Last error: {last_error}")

# ── Cola de reintentos ─────────────────────────────────────────────────────────
def queue_load() -> list:
    if QUEUE_FILE.exists():
        try:
            return json.loads(QUEUE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def queue_save(queue: list) -> None:
    QUEUE_FILE.write_text(json.dumps(queue, ensure_ascii=False, indent=2), encoding="utf-8")

def queue_add(request_data: dict, reason: str) -> None:
    queue = queue_load()
    entry = {
        "request":    request_data,
        "reason":     reason,
        "added_at":   datetime.now().isoformat(),
        "attempts":   1,
    }
    queue.append(entry)
    queue_save(queue)
    log.info("Queue: request from '%s' added to pending queue (%d total).",
             request_data.get("source", "unknown"), len(queue))

async def process_query(req: QueryRequest) -> dict:
    """Procesa una query y retorna el resultado. Lanza excepción si falla."""
    if req.ai == "image":
        img_bytes = generate_image(req.prompt)
        if not img_bytes:
            raise RuntimeError("All image providers failed.")
        img_b64 = base64.b64encode(img_bytes).decode()
        return {"image_base64": img_b64}

    # Texto/JSON — elegir provider
    full_prompt = req.prompt
    if req.schema:
        schema_str = json.dumps(req.schema, ensure_ascii=False, indent=2)
        full_prompt += (
            f"\n\nResponde ÚNICAMENTE con un JSON con esta estructura exacta, "
            f"sin texto adicional ni marcas de código:\n{schema_str}"
        )

    sources    = []
    model_used = None

    if req.ai == "ollama":
        # Solo Ollama — sin grounding
        if req.grounding:
            log.warning("grounding=True no está soportado con Ollama, se ignora.")
        text, model_used = call_ollama_model(full_prompt)

    elif req.ai == "auto":
        # Gemini primero, Ollama como fallback
        try:
            text, model_used, sources = call_text_model(full_prompt, grounding=req.grounding)
        except Exception as e:
            log.warning("Gemini failed in auto mode (%s), falling back to Ollama.", e)
            if req.grounding:
                log.warning("grounding no disponible en Ollama fallback.")
            text, model_used = call_ollama_model(full_prompt)

    else:
        # ai == "gemini" (default)
        text, model_used, sources = call_text_model(full_prompt, grounding=req.grounding)

    clean = clean_json(text)

    # Con grounding activo, extraer JSON si viene con texto extra
    if req.schema and req.grounding:
        json_match = re.search(r"\{.*\}", clean, re.DOTALL)
        if json_match:
            clean = json_match.group(0)

    # Intentar parsear como JSON si hay schema definido
    if req.schema:
        data = json.loads(clean)
    else:
        data = clean

    return {"data": data, "model_used": model_used, "sources": sources, "grounding_used": req.grounding and req.ai != "ollama"}

async def retry_pending_queue() -> None:
    queue = queue_load()
    if not queue:
        return
    log.info("Queue: attempting to retry %d pending item(s).", len(queue))
    remaining = []
    for item in queue:
        try:
            req = QueryRequest(**item["request"])
            result = await process_query(req)
            log.info("Queue: retry successful for source '%s'.", req.source)
            # No hay a dónde reenviar la respuesta — simplemente se descarta
            # En el futuro se podría agregar un webhook/callback
        except Exception as e:
            item["attempts"] = item.get("attempts", 1) + 1
            item["last_error"] = str(e)[:200]
            item["last_attempt"] = datetime.now().isoformat()
            log.warning("Queue: retry failed: %s", e)
            remaining.append(item)
    queue_save(remaining)

# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    log.info("[%s] Query received — ai=%s", req.source, req.ai)
    ts = datetime.now().isoformat()

    try:
        result = await process_query(req)
    except Exception as e:
        # Las imágenes no se encolan — en tier gratuito nunca van a poder reintentarse
        if req.ai != "image":
            queue_add(req.model_dump(), str(e)[:200])
        return QueryResponse(
            status="error",
            ai=req.ai,
            source=req.source,
            error=str(e)[:500],
            timestamp=ts,
        )

    # Éxito — reintentar pendientes
    await retry_pending_queue()

    return QueryResponse(
        status="ok",
        ai=req.ai,
        source=req.source,
        data=result.get("data") or result.get("image_base64"),
        model_used=result.get("model_used"),
        grounding_used=result.get("grounding_used", False),
        sources=result.get("sources", []),
        timestamp=ts,
    )

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file:   UploadFile = File(...),
    schema: str | None = Form(default=None, description="JSON string con la estructura esperada"),
    prompt: str        = Form(default="Analizá este documento y describí su contenido."),
    ai:     str        = Form(default="ollama", description="Provider: ollama (default) | gemini"),
    source: str        = Form(default="unknown"),
):
    """Analiza un PDF. Default: Ollama (local, privado). Opción: Gemini (nativo, envía el PDF a Google)."""
    log.info("[%s] PDF analyze request — file=%s ai=%s", source, file.filename, ai)
    ts = datetime.now().isoformat()

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return AnalyzeResponse(
            status="error", source=source,
            error="Solo se aceptan archivos PDF.",
            filename=file.filename, timestamp=ts,
        )

    pdf_bytes = await file.read()
    if len(pdf_bytes) > 20 * 1024 * 1024:  # 20MB límite
        return AnalyzeResponse(
            status="error", source=source,
            error="El archivo supera el límite de 20MB.",
            filename=file.filename, timestamp=ts,
        )

    schema_dict = None
    if schema:
        try:
            schema_dict = json.loads(schema)
        except json.JSONDecodeError:
            return AnalyzeResponse(
                status="error", source=source,
                error="El parámetro schema no es un JSON válido.",
                filename=file.filename, timestamp=ts,
            )

    try:
        if ai == "gemini":
            text, model_used = analyze_pdf_with_gemini(pdf_bytes, prompt, schema_dict)
        else:
            # ollama (default) — extrae texto localmente, no envía el PDF a ningún servicio externo
            text, model_used = analyze_pdf_with_ollama(pdf_bytes, prompt, schema_dict)
    except Exception as e:
        return AnalyzeResponse(
            status="error", source=source,
            error=str(e)[:500],
            filename=file.filename, timestamp=ts,
        )

    clean = clean_json(text)
    if schema_dict:
        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            # Intentar extraer JSON del texto
            json_match = re.search(r"\{.*\}", clean, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
            else:
                data = clean
    else:
        data = clean

    return AnalyzeResponse(
        status="ok", source=source,
        data=data, model_used=model_used,
        filename=file.filename, timestamp=ts,
    )

@app.get("/health")
def health():
    queue = queue_load()
    # Verificar Ollama
    ollama_status = "unreachable"
    ollama_models = []
    try:
        resp = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        if resp.status_code == 200:
            ollama_status = "ok"
            ollama_models = [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        pass
    return {
        "status":        "ok",
        "gemini_model":  TEXT_MODEL,
        "image_model":   IMAGE_MODEL,
        "ollama":        {"status": ollama_status, "active_model": OLLAMA_MODEL, "available": ollama_models},
        "pending":       len(queue),
    }

@app.get("/pending")
def get_pending():
    queue = queue_load()
    return {"total": len(queue), "items": queue}

@app.post("/retry")
async def retry_now():
    queue_before = queue_load()
    if not queue_before:
        return {"status": "ok", "message": "No pending items.", "processed": 0, "remaining": 0}
    await retry_pending_queue()
    queue_after = queue_load()
    return {
        "status":    "ok",
        "processed": len(queue_before) - len(queue_after),
        "remaining": len(queue_after),
        "items":     queue_after,
    }

@app.delete("/pending/{index}")
def delete_pending(index: int):
    queue = queue_load()
    if index < 0 or index >= len(queue):
        raise HTTPException(status_code=404, detail=f"Item {index} not found in queue.")
    removed = queue.pop(index)
    queue_save(queue)
    return {"status": "ok", "message": f"Item {index} removed.", "item": removed}
