"""OpenAI-compatible NVIDIA NIM proxy — Render web service.

Stateless compute: the key pool, client keys and admin secret live in Supabase
PostgreSQL and are reached through PostgREST RPCs, so restarts and redeploys are
lossless and the service can be scaled or duplicated freely.

Routes
    GET  /                     management dashboard
    ANY  /admin/api/...        admin API      (X-Admin-Key)
    ANY  /v1/...               OpenAI proxy   (Bearer client key)
    GET  /healthz              liveness + pool snapshot (cron keep-alive target)
"""

from __future__ import annotations

import os
import re
import secrets as py_secrets
from datetime import datetime, timezone
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

UPSTREAM_ORIGIN = "https://integrate.api.nvidia.com"
MAX_RETRIES = 5

SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SERVICE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
FALLBACK_ADMIN = os.environ.get("ADMIN_KEY_FALLBACK", "alb-nim-secret-2026")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_FILE = os.path.join(BASE_DIR, "dashboard.html")

DROP_REQUEST_HEADERS = {
    "host",
    "authorization",
    "content-length",
    "connection",
    "accept-encoding",
    "x-forwarded-for",
    "x-forwarded-proto",
    "x-forwarded-host",
    "x-forwarded-port",
    "x-real-ip",
    "forwarded",
    "cf-connecting-ip",
    "true-client-ip",
    "cdn-loop",
}

DROP_RESPONSE_HEADERS = {
    "content-length",
    "content-encoding",
    "transfer-encoding",
    "connection",
    "keep-alive",
}

app = FastAPI(title="NIM Key Pool Proxy", docs_url=None, redoc_url=None)


# --------------------------------------------------------------------------- #
# Supabase access (PostgREST only — no SDK needed)
# --------------------------------------------------------------------------- #

def sb_headers() -> dict[str, str]:
    return {
        "apikey": SERVICE_KEY,
        "Authorization": f"Bearer {SERVICE_KEY}",
        "Content-Type": "application/json",
    }


async def rpc(name: str, params: dict[str, Any]) -> Any:
    async with httpx.AsyncClient(timeout=20) as client:
        res = await client.post(
            f"{SUPABASE_URL}/rest/v1/rpc/{name}", json=params, headers=sb_headers()
        )
        res.raise_for_status()
        return None if not res.text else res.json()


async def rest_select(table: str, query: str) -> list[dict[str, Any]]:
    async with httpx.AsyncClient(timeout=20) as client:
        res = await client.get(
            f"{SUPABASE_URL}/rest/v1/{table}?{query}", headers=sb_headers()
        )
        res.raise_for_status()
        return res.json()


async def rest_write(
    method: str, path: str, payload: Any = None, prefer: str | None = None
) -> Any:
    headers = sb_headers()
    if prefer:
        headers["Prefer"] = prefer
    async with httpx.AsyncClient(timeout=30) as client:
        res = await client.request(
            method, f"{SUPABASE_URL}/rest/v1/{path}", json=payload, headers=headers
        )
        if res.status_code >= 400:
            raise RuntimeError(f"{res.status_code}: {res.text[:300]}")
        return None if not res.text else res.json()


def as_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


# --------------------------------------------------------------------------- #
# Auth helpers
# --------------------------------------------------------------------------- #

def admin_candidate(request: Request) -> str:
    header = request.headers.get("x-admin-key")
    if header:
        return header
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.query_params.get("admin_key", "")


async def is_admin(request: Request) -> bool:
    candidate = admin_candidate(request)
    if not candidate:
        return False
    try:
        return bool(await rpc("verify_admin_key", {"p_candidate": candidate}))
    except Exception:
        return candidate == FALLBACK_ADMIN


def client_key_of(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.query_params.get("api_key", "")


def openai_error(message: str, code: str, status: int) -> JSONResponse:
    return JSONResponse(
        {
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": None,
                "code": code,
            }
        },
        status_code=status,
    )


# --------------------------------------------------------------------------- #
# Dashboard + health
# --------------------------------------------------------------------------- #

def proxy_base_for(request: Request) -> str:
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host", request.headers.get("host", ""))
    if host:
        origin = f"{proto}://{host}"
    else:
        origin = str(request.base_url).rstrip("/")
    return f"{origin}/v1"


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    with open(DASHBOARD_FILE, "r", encoding="utf-8") as handle:
        html = handle.read()
    return HTMLResponse(html.replace("__PROXY_BASE_URL__", proxy_base_for(request)))


@app.get("/healthz")
async def healthz() -> JSONResponse:
    snapshot: dict[str, Any] = {"ok": True, "service": "nim-key-pool-proxy"}
    try:
        rows = await rest_select("nvidia_keys", "select=id,status&limit=1000")
        now = datetime.now(timezone.utc)
        active = 0
        cooling = 0
        for row in rows:
            if row.get("status") == "active":
                active += 1
            elif row.get("status") == "cooling":
                cooling += 1
        snapshot["upstream_keys"] = len(rows)
        snapshot["active"] = active
        snapshot["cooling"] = cooling
    except Exception as exc:  # pragma: no cover - diagnostics only
        snapshot["ok"] = False
        snapshot["error"] = str(exc)
    return JSONResponse(snapshot, status_code=200 if snapshot["ok"] else 503)


# --------------------------------------------------------------------------- #
# Admin API
# --------------------------------------------------------------------------- #

async def admin_guard(request: Request) -> JSONResponse | None:
    if await is_admin(request):
        return None
    return JSONResponse(
        {
            "success": False,
            "error": "Unauthorized: Invalid or missing admin key in X-Admin-Key or Authorization header",
        },
        status_code=401,
    )


@app.get("/admin/api/keys")
async def admin_keys(request: Request):
    if (guard := await admin_guard(request)) is not None:
        return guard
    rows = await rest_select(
        "nvidia_keys",
        "select=id,api_key,status,cooldown_until,error_count,success_count,last_used_at&order=id.asc",
    )
    now = datetime.now(timezone.utc)
    active = cooling = success = 0
    for row in rows:
        status = row.get("status")
        if status == "active":
            active += 1
        elif status == "cooling":
            until = as_utc(row.get("cooldown_until"))
            if until and until <= now:
                active += 1
            else:
                cooling += 1
        success += row.get("success_count") or 0
    return {
        "success": True,
        "stats": {
            "total": len(rows),
            "active": active,
            "cooling": cooling,
            "success_count": success,
        },
        "keys": rows,
    }


@app.post("/admin/api/import")
async def admin_import(request: Request):
    if (guard := await admin_guard(request)) is not None:
        return guard
    body = await request.json()
    text = body.get("text") or ""
    mode = "overwrite" if body.get("mode") == "overwrite" else "append"

    matches = re.findall(r"nvapi-[A-Za-z0-9_\-]+", text)
    keys = matches or [
        chunk.strip() for chunk in re.split(r"[\r\n,]", text) if len(chunk.strip()) > 10
    ]
    unique = list(dict.fromkeys(keys))
    if not unique:
        return JSONResponse(
            {"success": False, "error": "未检测到有效 Key"}, status_code=400
        )

    if mode == "overwrite":
        await rpc("truncate_and_restart_keys", {})

    for start in range(0, len(unique), 100):
        batch = [
            {"api_key": key, "status": "active"} for key in unique[start : start + 100]
        ]
        if mode == "overwrite":
            await rest_write("POST", "nvidia_keys", batch)
        else:
            await rest_write(
                "POST",
                "nvidia_keys?on_conflict=api_key",
                batch,
                prefer="resolution=ignore-duplicates",
            )

    await rpc("resequence_keys", {})
    return {"success": True, "mode": mode, "count": len(unique)}


@app.post("/admin/api/resequence")
async def admin_resequence(request: Request):
    if (guard := await admin_guard(request)) is not None:
        return guard
    await rpc("resequence_keys", {})
    return {"success": True}


@app.post("/admin/api/reset-cooling")
async def admin_reset_cooling(request: Request):
    if (guard := await admin_guard(request)) is not None:
        return guard
    await rest_write(
        "PATCH",
        "nvidia_keys?status=eq.cooling",
        {"status": "active", "cooldown_until": None},
    )
    return {"success": True}


@app.delete("/admin/api/key/{key_id}")
async def admin_delete_key(key_id: int, request: Request):
    if (guard := await admin_guard(request)) is not None:
        return guard
    await rest_write("DELETE", f"nvidia_keys?id=eq.{key_id}")
    await rpc("resequence_keys", {})
    return {"success": True}


@app.get("/admin/api/proxy-keys")
async def admin_proxy_keys(request: Request):
    if (guard := await admin_guard(request)) is not None:
        return guard
    rows = await rest_select(
        "proxy_api_keys",
        "select=id,key_name,api_key,status,created_at,last_used_at&order=id.asc",
    )
    return {"success": True, "keys": rows}


@app.post("/admin/api/proxy-keys")
async def admin_proxy_keys_create(request: Request):
    if (guard := await admin_guard(request)) is not None:
        return guard
    body = await request.json()
    api_key = (body.get("api_key") or "").strip() or "sk-alb-" + py_secrets.token_hex(20)
    rows = await rest_write(
        "POST",
        "proxy_api_keys",
        [
            {
                "key_name": body.get("key_name") or "New Client Key",
                "api_key": api_key,
                "status": "active",
            }
        ],
        prefer="return=representation",
    )
    return {"success": True, "key": rows[0] if rows else None}


@app.put("/admin/api/proxy-keys/{key_id}")
async def admin_proxy_keys_update(key_id: int, request: Request):
    if (guard := await admin_guard(request)) is not None:
        return guard
    body = await request.json()
    patch: dict[str, Any] = {}
    if body.get("key_name") is not None:
        patch["key_name"] = body["key_name"]
    if body.get("api_key"):
        patch["api_key"] = body["api_key"].strip()
    if body.get("status") is not None:
        patch["status"] = body["status"]
    if not patch:
        return JSONResponse(
            {"success": False, "error": "nothing to update"}, status_code=400
        )
    rows = await rest_write(
        "PATCH", f"proxy_api_keys?id=eq.{key_id}", patch, prefer="return=representation"
    )
    return {"success": True, "key": rows[0] if rows else None}


@app.delete("/admin/api/proxy-keys/{key_id}")
async def admin_proxy_keys_delete(key_id: int, request: Request):
    if (guard := await admin_guard(request)) is not None:
        return guard
    await rest_write("DELETE", f"proxy_api_keys?id=eq.{key_id}")
    return {"success": True}


@app.post("/admin/api/change-admin-key")
async def admin_change_key(request: Request):
    if (guard := await admin_guard(request)) is not None:
        return guard
    body = await request.json()
    new_key = (body.get("new_key") or "").strip()
    current = (body.get("current_key") or admin_candidate(request)).strip()
    if len(new_key) < 4:
        return JSONResponse(
            {"success": False, "error": "新管理密钥长度不能少于 4 个字符"},
            status_code=400,
        )
    ok = await rpc("update_admin_key", {"p_current": current, "p_new": new_key})
    if not ok:
        return JSONResponse(
            {"success": False, "error": "当前密钥校验失败，无法修改"}, status_code=400
        )
    return {"success": True, "message": "管理员登录密钥修改成功，已即时热生效！"}


# --------------------------------------------------------------------------- #
# OpenAI-compatible passthrough with pool hot-switching
# --------------------------------------------------------------------------- #

async def relay_chunks(upstream: httpx.Response, client: httpx.AsyncClient) -> AsyncIterator[bytes]:
    try:
        async for chunk in upstream.aiter_raw():
            yield chunk
    finally:
        await upstream.aclose()
        await client.aclose()


@app.api_route(
    "/v1/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
)
async def proxy(path: str, request: Request):
    if request.method == "OPTIONS":
        return Response(
            status_code=204,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "*",
            },
        )

    client_key = client_key_of(request)
    if not client_key:
        return openai_error(
            "You didn't provide an API key. You need to provide your API key in an "
            "Authorization header using Bearer auth (i.e. Authorization: Bearer YOUR_KEY).",
            "missing_api_key",
            401,
        )

    try:
        valid = bool(
            await rpc("verify_and_touch_client_key", {"p_client_key": client_key})
        )
    except Exception:
        valid = False
    if not valid:
        return openai_error(
            "Incorrect API key provided. Check your API key in the management dashboard.",
            "invalid_api_key",
            401,
        )

    body = await request.body()
    incoming = {
        k: v for k, v in request.headers.items() if k.lower() not in DROP_REQUEST_HEADERS
    }
    upstream_url = f"{UPSTREAM_ORIGIN}/v1/{path}"
    if request.url.query:
        upstream_url += f"?{request.url.query}"

    tried: list[str] = []
    for _ in range(MAX_RETRIES):
        try:
            api_key = await rpc("get_next_available_key", {"p_exclude": tried})
        except Exception:
            api_key = None
        if not api_key:
            return JSONResponse(
                {
                    "error": {
                        "message": "No available upstream API keys in pool (all cooling or exhausted)",
                        "type": "key_pool_exhausted",
                        "code": 503,
                    }
                },
                status_code=503,
            )

        tried.append(api_key)
        headers = dict(incoming)
        headers["Authorization"] = f"Bearer {api_key}"

        client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=15, read=600, write=60, pool=15)
        )
        try:
            upstream = await client.send(
                client.build_request(
                    request.method, upstream_url, headers=headers, content=body
                ),
                stream=True,
            )
        except Exception as exc:
            await client.aclose()
            print(f"[proxy] upstream error {api_key[:12]}...: {exc}", flush=True)
            continue

        status = upstream.status_code

        if status in (429, 401, 403):
            await upstream.aclose()
            await client.aclose()
            try:
                await rpc(
                    "mark_key_result", {"p_key": api_key, "p_status_code": status}
                )
            except Exception:
                pass
            print(f"[proxy] {status} on {api_key[:12]}... switching", flush=True)
            continue

        if 200 <= status < 300:
            try:
                await rpc("mark_key_result", {"p_key": api_key, "p_status_code": status})
            except Exception:
                pass

        out_headers = {
            k: v
            for k, v in upstream.headers.items()
            if k.lower() not in DROP_RESPONSE_HEADERS
        }
        out_headers["Access-Control-Allow-Origin"] = "*"

        if "text/event-stream" in upstream.headers.get("content-type", ""):
            return StreamingResponse(
                relay_chunks(upstream, client), status_code=status, headers=out_headers
            )

        payload = await upstream.aread()
        await upstream.aclose()
        await client.aclose()
        return Response(content=payload, status_code=status, headers=out_headers)

    return JSONResponse(
        {
            "error": {
                "message": f"Upstream retries exhausted after {MAX_RETRIES} attempts across keys",
                "type": "proxy_error",
                "code": 502,
            }
        },
        status_code=502,
    )