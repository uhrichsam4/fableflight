# Multiplayer relay

A tiny Cloudflare Worker that lets players of Samiam's Flight Sim fly together
by typing the same room code. Each code gets its own Durable Object, which
relays every message one player sends to everyone else in the room. It keeps
no game state; the free Workers plan is enough.

- `GET /` answers `ok` (health check).
- `GET /room/<CODE>` (WebSocket upgrade) joins room `<CODE>`: 4 to 8 letters
  or digits, upper case. At most 8 players a room (`409 room full` past that),
  messages of at most 2 KB.
- The room sends `{"t":"welcome","id":N,"peers":[...]}` to a newcomer,
  `{"t":"join","id":N}` / `{"t":"leave","id":N}` to the others, and every
  relayed message as `{"from":N,"d":<message>}`.

## Run it locally

```sh
cd server
npm install
npx wrangler dev            # ws://127.0.0.1:8787
```

Then start two games with
`--mp-server ws://127.0.0.1:8787 --mp-join TEST --mp-name A` (and `B`), each
with its own `FABLE_CONFIG_DIR` so they do not share settings. `--mp-create`
makes a room with a random code instead (printed on start). These flags are
for this run only; they are not saved.

## Deploy

```sh
cd server
npm install
npx wrangler login          # once, opens the browser
npx wrangler deploy
```

Wrangler prints the address, `https://fableflight-mp.<your-subdomain>.workers.dev`.
Put it in `MP_SERVER` in `src/mp.rs` as `wss://fableflight-mp.<your-subdomain>.workers.dev`
and rebuild (or type it under Multiplayer > Server in the game, or pass `--mp-server`).

The free Workers plan is enough: the room class is a SQLite-backed Durable
Object (`new_sqlite_classes` in `wrangler.toml`), and the WebSocket Hibernation
API means an idle room costs nothing. `node_modules/` and `.wrangler/` are
ignored (`.gitignore`); do not commit them.

## What a game sends

About ten times a second, its own vehicle as one JSON text (see `State::encode`
in `src/mp.rs`):

```json
{"n":"Sam","a":0,"p":[37.6189354,-122.3749812,1523.37],"q":[x,y,z,w],"v":[vx,vy,vz],"g":1,"f":0.33,"h":0.8,"c":0,"ts":12.3}
```

name, vehicle (0..4 the aircraft of `aircraft::ALL`, 5 the speedboat),
lat/lon (degrees)/altitude (m), body-to-world rotation, world velocity (m/s),
gear, flaps, throttle (0..1), crashed, and the sender's clock (s). The relay
only checks that it is JSON of at most 2 KB.
