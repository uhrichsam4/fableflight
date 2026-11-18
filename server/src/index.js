// Samiam's Flight Sim multiplayer relay: a Cloudflare Worker with one Durable
// Object per room code. Clients open a WebSocket to /room/<CODE>; whatever
// one sends, the others in the room receive, wrapped as {"from":N,"d":<msg>}.
// The room also tells everyone who comes and goes: {"t":"join"|"leave","id":N},
// and greets each newcomer with {"t":"welcome","id":N,"peers":[...]}.
// It keeps no state but the sockets, and uses the WebSocket Hibernation API
// so an idle room costs nothing.

import { DurableObject } from "cloudflare:workers";

const MAX_PLAYERS = 8;
const MAX_MSG = 2048;
const CODE = /^[A-Z0-9]{4,8}$/;

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === "/" || url.pathname === "") {
      return new Response("ok", { headers: { "content-type": "text/plain" } });
    }
    const m = url.pathname.match(/^\/room\/([^/]+)\/?$/);
    if (!m) return new Response("not found", { status: 404 });
    const code = decodeURIComponent(m[1]).toUpperCase();
    if (!CODE.test(code)) return new Response("bad room code", { status: 400 });
    if (request.headers.get("Upgrade")?.toLowerCase() !== "websocket") {
      return new Response("expected a WebSocket upgrade", { status: 426 });
    }
    const stub = env.ROOMS.get(env.ROOMS.idFromName(code));
    return stub.fetch(request);
  },
};

export class Room extends DurableObject {
  constructor(ctx, env) {
    super(ctx, env);
  }

  idOf(ws) {
    const a = ws.deserializeAttachment();
    return a && typeof a.id === "number" ? a.id : -1;
  }

  async fetch(request) {
    // (sockets that have left but are still closing do not count)
    const sockets = this.ctx.getWebSockets().filter((s) => this.idOf(s) >= 0);
    if (sockets.length >= MAX_PLAYERS) {
      return new Response("room full", { status: 409 });
    }
    // Ids are never reused within a room, so a late message from someone
    // who left cannot be taken for a newcomer's.
    let id = (await this.ctx.storage.get("next")) || 1;
    for (const s of sockets) id = Math.max(id, this.idOf(s) + 1);
    await this.ctx.storage.put("next", id + 1);
    const pair = new WebSocketPair();
    const [client, server] = Object.values(pair);
    this.ctx.acceptWebSocket(server);
    server.serializeAttachment({ id });
    const peers = sockets.map((s) => this.idOf(s));
    server.send(JSON.stringify({ t: "welcome", id, peers }));
    this.broadcast(JSON.stringify({ t: "join", id }), server);
    return new Response(null, { status: 101, webSocket: client });
  }

  broadcast(text, except) {
    for (const s of this.ctx.getWebSockets()) {
      if (s === except || this.idOf(s) < 0) continue;
      try {
        s.send(text);
      } catch (_) {
        // closing: its webSocketClose will follow
      }
    }
  }

  async webSocketMessage(ws, message) {
    if (typeof message !== "string") return; // text only
    if (message.length > MAX_MSG) return;
    const id = this.idOf(ws);
    if (id < 0) return; // already left
    // The message is relayed as it is, inside a wrapper naming its sender,
    // so it has to be JSON itself or the wrapper would be spoilt too.
    try {
      JSON.parse(message);
    } catch (_) {
      return;
    }
    this.broadcast('{"from":' + id + ',"d":' + message + "}", ws);
  }

  async webSocketClose(ws, code, reason, wasClean) {
    this.leave(ws);
    try {
      ws.close(1000, "bye");
    } catch (_) {}
  }

  async webSocketError(ws, error) {
    this.leave(ws);
  }

  leave(ws) {
    const id = this.idOf(ws);
    if (id < 0) return;
    ws.serializeAttachment({ id: -1 });
    this.broadcast(JSON.stringify({ t: "leave", id }), ws);
  }
}
