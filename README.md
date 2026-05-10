# Oracle — AI Gateway

Microservicio de IA autoalojado en Docker. Actúa como gateway centralizado para consultas a modelos de inteligencia artificial. Cualquier contenedor de la red puede enviarle un prompt y recibir una respuesta estructurada en JSON, sin necesidad de tener la API key ni conocer los modelos disponibles.

> Oracle es la columna vertebral de [Butler API](https://codeberg.org/osdaeg/butler) y otros servicios del homelab.

## Características

- **Autodescubrimiento de modelos** — al arrancar consulta la API de Gemini y selecciona automáticamente el mejor modelo disponible
- **Rotación automática** — si un modelo agota su cuota diaria o devuelve error, rota al siguiente sin intervención manual
- **Google Search Grounding** — opcionalmente ancla las respuestas a resultados de búsqueda en tiempo real
- **Análisis de PDFs local** — via Ollama + pypdf, sin enviar documentos a servicios externos
- **Generación de imágenes** — con fallbacks en cascada (Gemini nativo → Pollinations → Stable Horde)
- **Cola de reintentos** — las consultas de texto fallidas se guardan y reintentan automáticamente
- **Respuesta estructurada** — siempre devuelve `status: ok | error` para manejo sencillo en el cliente

## Requisitos

- Docker y Docker Compose
- API key de Google Gemini ([obtener en Google AI Studio](https://aistudio.google.com))
- Red Docker externa ya creada
- Ollama corriendo (opcional, para análisis de PDFs local)

## Instalación

```bash
git clone <repo>
cd oracle
cp .env.example .env
nano .env   # completar GEMINI_API_KEY y ajustar puertos y red
docker-compose up -d --build
```

Verificar que levantó correctamente:

```bash
curl http://localhost:7998/health
```

## Configuración

| Variable | Descripción | Default |
|---|---|---|
| `GEMINI_API_KEY` | API key de Google Gemini | — |
| `OLLAMA_URL` | URL del servidor Ollama | `http://192.168.88.100:11434` |
| `OLLAMA_MODEL` | Modelo Ollama a usar | `gemma3:1b` |
| `STABLE_HORDE_KEY` | API key de Stable Horde (opcional) | `0000000000` |
| `ORACLE_EXTERNAL_PORT` | Puerto externo del contenedor | `7998` |

En `docker-compose.yml` ajustar también:
- `volumes` → ruta local donde se guarda la cola de reintentos
- `networks` → nombre de la red Docker existente

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/query` | Consulta de texto o generación de imagen |
| `POST` | `/analyze` | Análisis de PDF |
| `GET` | `/health` | Estado del servicio, modelos y Ollama |
| `GET` | `/pending` | Lista la cola de reintentos |
| `POST` | `/retry` | Dispara reintentos manualmente |
| `DELETE` | `/pending/{index}` | Elimina un item de la cola por índice |

## Uso — /query

### Consulta de texto simple

```bash
curl -X POST http://localhost:7998/query \
  -H "Content-Type: application/json" \
  -d '{"prompt": "¿Qué es la saga The Expanse?", "source": "mi-servicio"}'
```

### Consulta con schema JSON

Cuando se especifica un `schema`, Oracle instruye al modelo para que responda exactamente con esa estructura y devuelve el JSON ya parseado.

```bash
curl -X POST http://localhost:7998/query \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Datos de la película Blade Runner 1982",
    "schema": {
      "titulo": "string",
      "director": "string",
      "año": 0,
      "genero": "string",
      "resena": "string"
    },
    "source": "mi-servicio"
  }'
```

### Consulta con Google Search (datos en tiempo real)

```bash
curl -X POST http://localhost:7998/query \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Resultados del Turismo Carretera este fin de semana",
    "schema": {
      "actividades": [{"competicion": "string", "fecha": "string", "hora": "string"}]
    },
    "grounding": true,
    "source": "deportes-service"
  }'
```

### Generación de imagen

Devuelve la imagen codificada en base64 en el campo `data`. **Nota:** en el tier gratuito de Gemini los modelos de imagen tienen cuota 0. Los errores de imagen no se encolan — se recomienda usar placeholder en el cliente.

```bash
curl -X POST http://localhost:7998/query \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Cover art for a sci-fi novel", "ai": "image", "source": "butler-api"}'
```

## Uso — /analyze

Analiza el contenido de un PDF. Por defecto usa **Ollama localmente** — el documento nunca sale del servidor.

```bash
# Análisis general
curl -X POST http://localhost:7998/analyze \
  -F "file=@documento.pdf" \
  -F "source=mi-servicio"

# Con schema estructurado
curl -X POST http://localhost:7998/analyze \
  -F "file=@factura.pdf" \
  -F 'schema={"tipo_documento":"string","empresa":"string","total":"string","fecha":"string"}' \
  -F "source=mi-servicio"

# Con Gemini (envía el PDF a Google)
curl -X POST http://localhost:7998/analyze \
  -F "file=@documento.pdf" \
  -F "ai=gemini" \
  -F "source=mi-servicio"
```

**Limitaciones del análisis con Ollama:**
- Solo procesa PDFs con texto extraíble (no escaneados)
- `gemma3:1b` es limitado — puede errar datos específicos
- Modelos más grandes (`gemma3:4b`, `llama3.2:3b`) darían mejores resultados

## Estructura del request (/query)

| Campo | Tipo | Descripción | Default |
|---|---|---|---|
| `prompt` | string | Consulta en lenguaje natural | — |
| `schema` | object | Estructura JSON esperada en la respuesta | `null` |
| `ai` | string | Provider: `gemini` \| `ollama` \| `auto` \| `image` | `gemini` |
| `grounding` | bool | Habilitar Google Search en tiempo real (solo gemini) | `false` |
| `source` | string | Identificador del servicio cliente (para logs) | `unknown` |

## Estructura de la respuesta

```json
{
  "status": "ok",
  "ai": "gemini",
  "source": "mi-servicio",
  "data": { ... },
  "error": null,
  "model_used": "gemini-2.5-flash",
  "grounding_used": false,
  "sources": [],
  "timestamp": "2026-05-02T12:00:00"
}
```

## Cola de reintentos

Las consultas de texto fallidas se guardan en `/app/data/pending_queue.json` y se reintentan automáticamente tras cada consulta exitosa.

**Las imágenes (`ai="image") nunca se encolan** — en el tier gratuito de Gemini los modelos de imagen tienen cuota 0 y no podrían reintentarse.

## Providers de imagen

| Provider | Estado | Notas |
|---|---|---|
| Gemini nativo | ⚠️ Cuota 0 en tier gratuito | Funciona con billing activado |
| Pollinations.ai | ⚠️ Desactivado | Error 530 desde Docker |
| Stable Horde | ⚠️ Desactivado | Calidad insuficiente |

## Integración desde otro servicio

```python
import httpx

def oracle_query(prompt: str, schema: dict = None) -> dict:
    resp = httpx.post("http://oracle:8000/query", json={
        "prompt": prompt,
        "schema": schema,
        "source": "mi-servicio",
    }, timeout=300)
    result = resp.json()
    if result["status"] != "ok":
        raise RuntimeError(result["error"])
    return result["data"]

def oracle_analyze_pdf(pdf_path: str, schema: dict = None) -> dict:
    with open(pdf_path, "rb") as f:
        resp = httpx.post("http://oracle:8000/analyze",
            files={"file": f},
            data={"schema": json.dumps(schema) if schema else None,
                  "source": "mi-servicio"},
            timeout=300)
    result = resp.json()
    if result["status"] != "ok":
        raise RuntimeError(result["error"])
    return result["data"]
```
