Open world flight simulator in pure rust created by Claude Fable 5.2 and Opus 5.5

# Demo
[Demo](https://github.com/user-attachments/assets/4f8833da-13c0-4d20-a50b-c0acf2d88df1)



# Compiling
* Install rust
```
cargo build --release
```

running
```
cargo run --release
```

# Speedboat and waves
Besides the aircraft, the start screen's carousel (or `--aircraft boat`) has a speedboat to drive: W/S throttle and reverse, A/D steer, Shift (or Space) for full power. It goes in at the nearest open water to the chosen spot and rides the same waves that are drawn; it planes, banks into turns, jumps off crests and leaves a wake. At Water High the sea near the camera is a fine displaced mesh, so swell and chop stand up as real 3D waves; Water Low keeps the lighter surface.

# Language
The interface is in English or French. It follows the system language, can be changed in Settings (start screen or pause menu), and is remembered. For one run: `cargo run --release -- --lang fr`.

# Live air traffic
Real aircraft around you, as reported by ADS-B receivers (data from [adsb.lol](https://adsb.lol), ODbL), fly in the sky with tags (callsign, type, altitude, speed, distance); the flight panel (Tab) lists the nearest. It can be switched off in the pause menu's Settings.

# Real weather
The weather follows the real conditions where you fly: the nearest METAR within 60 km ([aviationweather.gov](https://aviationweather.gov)) sets the wind, gusts, visibility, cloud layers and thunderstorms; with no airport near, [Open-Meteo](https://open-meteo.com)'s model stands in. It is looked up again every 10 minutes or 30 km, and the sky eases into each report. Switch it off in the weather settings (start screen, flight panel or F3) to set the weather by hand.

The clock can follow real time too ("Real time" above the day sliders): the sun and the date are those where you fly, now (solar time, so the local hour can differ from the legal one by up to an hour or so). Pressing T / Y to set the time by hand leaves real time.

Settings (language, sound, camera, HUD, anti-aliasing) are kept in `settings.json` in the config folder (`~/.config/fablegame` on Linux).

# Contributions
Please contribute anything if you want to! Improvements to flight sim realism, graphics, optimization, UI, and especially plane models are needed! (((=
