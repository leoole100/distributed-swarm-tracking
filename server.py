# server.py
import asyncio
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay

pcs = set()
relay = MediaRelay()

# cam_id -> source track (the one coming from the publisher)
sources = {}

async def index(_):
    return web.FileResponse("static/index.html")

async def offer_publisher(request):
    """
    Publisher posts {sdp, type, camId}. Server receives the track and stores it by camId.
    """
    params = await request.json()
    cam_id = params.get("camId", "cam")
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)
    print(f"[PUB:{cam_id}] new PC (total {len(pcs)})")

    # Tell browser we're only receiving video
    pc.addTransceiver("video", direction="recvonly")

    @pc.on("track")
    def on_track(track):
        print(f"[PUB:{cam_id}] got track kind={track.kind}")
        if track.kind == "video":
            # store the *source* track; relay.subscribe(...) will be used per viewer later
            sources[cam_id] = track

            @track.on("ended")
            async def _ended():
                print(f"[PUB:{cam_id}] track ended")
                # remove source when publisher stops
                if sources.get(cam_id) is track:
                    sources.pop(cam_id, None)

    @pc.on("connectionstatechange")
    async def on_state_change():
        print(f"[PUB:{cam_id}] state -> {pc.connectionState}")
        if pc.connectionState in ("failed", "closed"):
            await pc.close()
            pcs.discard(pc)
        elif pc.connectionState == "disconnected":
            await asyncio.sleep(3)
            if pc.connectionState == "disconnected":
                await pc.close()
                pcs.discard(pc)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

async def offer_viewer(request):
    """
    Viewer posts {sdp, type, camId}. Server subscribes to that camId via MediaRelay
    and sends it to the viewer (sendonly to viewer).
    """
    params = await request.json()
    cam_id = params.get("camId", "cam")
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    if cam_id not in sources:
        return web.json_response({"error": f"camId '{cam_id}' not available"}, status=404)

    pc = RTCPeerConnection()
    pcs.add(pc)
    print(f"[VIEW:{cam_id}] new PC (total {len(pcs)})")

    # Subscribe to the publisher's source using the shared relay
    sub_track = relay.subscribe(sources[cam_id])
    pc.addTrack(sub_track)  # sendonly from server to this viewer

    @pc.on("connectionstatechange")
    async def on_state_change():
        print(f"[VIEW:{cam_id}] state -> {pc.connectionState}")
        if pc.connectionState in ("failed", "closed"):
            await pc.close()
            pcs.discard(pc)
        elif pc.connectionState == "disconnected":
            await asyncio.sleep(3)
            if pc.connectionState == "disconnected":
                await pc.close()
                pcs.discard(pc)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

async def on_shutdown(app):
    print("Shutting down...")
    for pc in list(pcs):
        await pc.close()
    pcs.clear()
    sources.clear()


async def list_sources(_):
    return web.json_response({"cams": sorted(list(sources.keys()))})

app = web.Application()
app.add_routes([
    web.get("/", index),
    web.get("/sources", list_sources),
    web.post("/offer/publisher", offer_publisher),
    web.post("/offer/viewer", offer_viewer),
    web.static("/static", "static"),
])
app.on_shutdown.append(on_shutdown)


if __name__ == "__main__":
    import ssl
    sslctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    sslctx.load_cert_chain("cert.pem", "key.pem")
    web.run_app(app, host="0.0.0.0", port=8443, ssl_context=sslctx)
