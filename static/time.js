const timeSync = {
    offset: 0,
    ws: new WebSocket(`wss://${location.host}/api/time`),
    sync: function() {
        return new Promise((resolve) => {
            const start = performance.now() / 1000;
            this.ws.send("time");
            this.ws.addEventListener("message", function handler(event) {
                const end = performance.now() / 1000;
                const ping = end - start; // seconds
                const serverTime = Number(event.data);
                const computedOffset = serverTime - ((start + end) / 2);
                const offsetChange = computedOffset - timeSync.offset;
                timeSync.offset = computedOffset;
                timeSync.ws.removeEventListener("message", handler);
                resolve({ ping: ping, offsetChange: offsetChange });
            });
        });
    },
    currentTime: function() {
        return (performance.now() / 1000) + this.offset;
    }
};