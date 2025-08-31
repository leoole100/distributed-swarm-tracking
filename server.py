from aiohttp import web
import ssl

from time_server import time_handler
import video_server as vs

async def index(_):
    return web.FileResponse("static/index.html")

app = web.Application()
app.add_routes([
    web.get("/", index),
    web.static("/", "static"),

    web.get("/api/time", time_handler),

    web.post("/api/webrtc/publish", vs.offer),
    # web.post("/api/webrtc/view", vs.view_offer),
    # web.get("/api/webrtc/stats", vs.stats),
])


if __name__ == "__main__":
    sslctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    sslctx.load_cert_chain("cert.pem", "key.pem")       # see readme on how to generate
    web.run_app(app, host="0.0.0.0", port=8443, ssl_context=sslctx)
