"""
AgentLens — WebSocket Connection Manager
Manages active WebSocket connections and broadcasts new trace events.
"""
import json
from typing import List
from fastapi import WebSocket


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"  WS: client connected ({len(self.active_connections)} total)")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"  WS: client disconnected ({len(self.active_connections)} remaining)")

    async def broadcast(self, event_type: str, data: dict):
        """Send event to all connected clients."""
        message = json.dumps({"type": event_type, "data": data})
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                disconnected.append(connection)
        for conn in disconnected:
            self.disconnect(conn)

    async def send_personal(self, websocket: WebSocket, event_type: str, data: dict):
        """Send event to one specific client."""
        message = json.dumps({"type": event_type, "data": data})
        await websocket.send_text(message)


# Global singleton
manager = ConnectionManager()
