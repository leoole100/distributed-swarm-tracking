from aiohttp import web
import ssl

from time_server import time_handler

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
