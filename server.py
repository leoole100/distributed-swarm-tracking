import asyncio, json
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
import av

pcs = set()

class VideoPrinter(MediaStreamTrack):
    kind = "video"
    def __init__(self, track, cam_id):
        super().__init__()
        self.track, self.cam_id = track, cam_id
    async def recv(self):
        frame = await self.track.recv()
        print(f"Got frame from {self.cam_id} ts={frame.time}")
        return frame  # just forward it

async def index(request):
    return web.FileResponse("static/client.html")

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    cam_id = params.get("camId", "cam")

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            pc.addTrack(VideoPrinter(track, cam_id))

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return web.json_response(
        {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    )

app = web.Application()
app.add_routes([web.get("/", index), web.post("/offer", offer), web.static("/static", "static")])

if __name__ == "__main__":
    import ssl
    sslctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    sslctx.load_cert_chain("cert.pem", "key.pem")
    # HTTPS on 8443
    web.run_app(app, host="0.0.0.0", port=8443, ssl_context=sslctx)