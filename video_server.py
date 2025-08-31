import asyncio, json
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate
from aiortc.contrib.media import MediaBlackhole
from aiortc.sdp import candidate_from_sdp
import uuid

pcs = set()

async def offer(request: web.Request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("track")
    def on_track(track):
        print("track:", track.kind)

        id = str(uuid.uuid4())
        
        async def pump():
            try:
                while True:
                    frame = await track.recv()
                    print(id, frame)
            except:
                pass
        asyncio.create_task(pump())

    async for msg in ws:
        if msg.type != web.WSMsgType.TEXT:
            continue
        data = json.loads(msg.data)

        if data["type"] == "offer":
            await pc.setRemoteDescription(RTCSessionDescription(sdp=data["sdp"], type="offer"))
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            await ws.send_str(json.dumps({
                "type": "answer",
                "sdp": pc.localDescription.sdp
            }))

        elif data["type"] == "candidate":
            # data looks like:
            # { "type":"candidate", "candidate":"candidate:...", "sdpMid":"0", "sdpMLineIndex":0 }
            c = candidate_from_sdp(data["candidate"])
            c.sdpMid = data.get("sdpMid")
            c.sdpMLineIndex = data.get("sdpMLineIndex")
            await pc.addIceCandidate(c)

        elif data["type"] == "end_of_candidates":
            await pc.addIceCandidate(None)

        elif data["type"] == "bye":
            break

    await pc.close()
    pcs.discard(pc)
    await ws.close()
    return ws
