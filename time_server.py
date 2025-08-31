from aiohttp import web
from datetime import datetime

async def time_handler(req: web.Request):
    ws = web.WebSocketResponse()
    await ws.prepare(req)

    async for msg in ws:
        if msg.type != web.WSMsgType.ERROR:
            if msg.type == web.WSMsgType.TEXT and msg.data.strip() == "time":
                timestamp = datetime.now().timestamp()
                await ws.send_str(f"{timestamp:.6f}")
    
    return ws