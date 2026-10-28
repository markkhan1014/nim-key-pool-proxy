# NIM Key Pool Proxy

OpenAI-compatible gateway in front of `integrate.api.nvidia.com` with a pooled
NVIDIA key ring, 429/401 hot switching, client `sk-alb-*` credentials and a web
dashboard — deployed as a Render web service.

| Route | Purpose |
|:--|:--|
| `POST /v1/chat/completions` | OpenAI-compatible passthrough (Bearer client key) |
| `GET /v1/models` | upstream model list |
| `GET /` | management dashboard (admin key) |
| `GET /healthz` | liveness + pool snapshot (cron keep-alive target) |

State lives in Supabase PostgreSQL (`nvidia_keys`, `proxy_api_keys`,
`admin_config`), so redeploys and restarts are lossless.

## Client usage

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://<service>.onrender.com/v1",
    api_key="sk-alb-...",           # from the dashboard
)

print(client.chat.completions.create(
    model="meta/llama-guard-4-12b",
    messages=[{"role": "user", "content": "ping"}],
).choices[0].message.content)
```

## Environment

| Name | Required | Meaning |
|:--|:--|:--|
| `SUPABASE_URL` | yes | project URL holding the key pool |
| `SUPABASE_SERVICE_ROLE_KEY` | yes | service role key (server-side only) |
| `ADMIN_KEY_FALLBACK` | no | admin password used if the DB lookup fails |

## Deploy

Blueprint (`render.yaml`) or manually: Python runtime, build
`pip install -r requirements.txt`, start
`uvicorn app:app --host 0.0.0.0 --port $PORT`, health check `/healthz`.

Free instances sleep after ~15 minutes idle; a cron-job.org ping every 5 minutes
keeps it warm.