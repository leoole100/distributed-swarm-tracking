#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
import uuid
from datetime import datetime
from pathlib import Path

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc import RTCPeerConnection, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaBlackhole, MediaRecorder

# Keep references so connections don't get GC'd
pcs: set[RTCPeerConnection] = set()


async def offer(request: web.Request):
    """
    Receive an SDP offer from a client, create a PeerConnection on the server,
    attach handlers for incoming tracks, and reply with an SDP answer.
    """
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection(
        RTCConfiguration(
            iceServers=[RTCIceServer("stun:stun.l.google.com:19302")]
            # For reliability across networks, add TURN:
            # iceServers=[RTCIceServer(urls="turn:your.turn:3478", username="u", credential="p")]
        )
    )
    pcs.add(pc)
    print(f"Created PeerConnection {id(pc)} (total: {len(pcs)})")

    # Per-connection recorders (or blackholes)
    recorders = []

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"PC {id(pc)} state -> {pc.connectionState}")
        if pc.connectionState in ("failed", "closed", "disconnected"):
            await cleanup_pc(pc, recorders)

    @pc.on("track")
    def on_track(track):
        print(f"PC {id(pc)}: Track {track.kind} received")

        blackhole = MediaBlackhole()
        recorders.append(blackhole)
        asyncio.create_task(blackhole.start())
        asyncio.create_task(pipe_track(track, blackhole, pc))

        @track.on("ended")
        async def on_ended():
            print(f"PC {id(pc)}: Track {track.kind} ended")

    # Apply the remote description (from client) and create/send answer
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    # Return the answer
    return web.json_response(
        {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
        }
    )

async def pipe_track(track, sink, pc: RTCPeerConnection):
    """
    Read frames/packets from an incoming track and feed them to the sink (recorder or blackhole).
    This lets us easily stop/close the sink when the track ends or the pc closes.
    """
    try:
        while True:
            frame = await track.recv()
            await sink.write(frame)
    except Exception as e:
        # Usually ends with asyncio.CancelledError or when the track/pc closes
        pass
    finally:
        # A track finished; if all tracks done, you can decide to stop the sink here.
        # We leave stopping to cleanup so multi-track recorders close together.
        ...

async def cleanup_pc(pc: RTCPeerConnection, recorders):
    # Gracefully stop recorders/sinks
    for r in recorders:
        try:
            await r.stop()
        except Exception:
            pass

    # Close the PeerConnection
    try:
        await pc.close()
    except Exception:
        pass

    if pc in pcs:
        pcs.discard(pc)
    print(f"Closed PC {id(pc)} (remaining: {len(pcs)})")

async def on_shutdown(app: web.Application):
    # Close all PCs on server shutdown
    coros = []
    for pc in list(pcs):
        coros.append(cleanup_pc(pc, []))
    if coros:
        await asyncio.gather(*coros, return_exceptions=True)