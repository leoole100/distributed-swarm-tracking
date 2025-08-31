from aiohttp import web
from typing import Set
import asyncio
import ssl
from datetime import datetime

ws_clients: Set[web.WebSocketResponse] = set()

async def time_handler(req: web.Request):
    ws = web.WebSocketResponse()
    await ws.prepare(req)

    ws_clients.add(ws)
    try:
        async for msg in ws:
            if msg.type != web.WSMsgType.ERROR:
                if msg.type == web.WSMsgType.TEXT and msg.data.strip() == "time":
                    timestamp = datetime.now().timestamp()
                    await ws.send_str(f"{timestamp:.6f}")
    finally:
        ws_clients.discard(ws)
    
    return ws

async def index(_):
    return web.FileResponse("static/index.html")


app = web.Application()
app.add_routes([
    web.get("/", index),
    web.get("/time", time_handler),
    web.static("/", "static"),
])


if __name__ == "__main__":
    sslctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    sslctx.load_cert_chain("cert.pem", "key.pem")       # see readme on how to generate
    web.run_app(app, host="0.0.0.0", port=8443, ssl_context=sslctx)
