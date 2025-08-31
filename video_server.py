import asyncio, json, uuid
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.sdp import candidate_from_sdp

async def offer(request: web.Request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    pc = RTCPeerConnection()

    uid = str(uuid.uuid4())

    @pc.on("datachannel")
    def on_datachannel(dc):
        print("datachannel opened:", dc.label)

        @dc.on("message")
        def on_message(msg):
            t = json.loads(msg)     # { "frame_id": ..., "time": ... }
            print(uid, "tag:", t)

    @pc.on("track")
    def on_track(track):
        """ Assuming one track per offer and ws connection for now. """
        print(id, "track:", track.kind)

        async def pump():
            i = 0
            try:
                while True:
                    frame = await track.recv()
                    print(uid, "frame", i, frame)
                    i += 1
            except Exception as e:
                print("pump stopped:", e)

        asyncio.create_task(pump())

    # --- signaling over WebSocket ---
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
            c = candidate_from_sdp(data["candidate"])
            c.sdpMid = data.get("sdpMid")
            c.sdpMLineIndex = data.get("sdpMLineIndex")
            await pc.addIceCandidate(c)

        elif data["type"] == "end_of_candidates":
            await pc.addIceCandidate(None)

        elif data["type"] == "bye":
            break

    await pc.close()
    await ws.close()
    return ws
