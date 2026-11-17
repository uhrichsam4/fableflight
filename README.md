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

# Ships and carriers
Big ships sail the open sea near you, the same ones in the same places for each patch of ocean: the Titanic (and her sisters Olympic and Britannic, 269 m), a 1 km container ship, Nimitz-class aircraft carriers (333 m) and mega carriers 3.6 km long with a 420 m deck. They ride the waves, leave wakes, carry navigation lights at night and show on the minimap and the large map (M).

Start on one from the start screen (START LOCATION → Ships) or with `--ship carrier|megacarrier|titanic|megaship` (add `--approach` for a final approach to a carrier). The ship sails into the wind on the open sea nearest the spot, at 28 kt, with one of each of the others in company. On the carrier you sit on the bow catapult: run up to full power and hold Space for a moment to launch. O lowers the tail hook; catch one of the four wires on the angled deck to stop. A 747 (or the 747 ×5) starts at the stern of the mega carrier: full power releases the brakes, rotate at about 150 kt. Decks are solid and move with the ship; flying into a hull, island or superstructure is a crash. R puts you back on (or on approach to) your ship. The Titanic and the mega ship start you in the air alongside (or, with the speedboat, in the water next to them).

# Sea
The weather panel (F3, the flight panel's Weather page and the start screen's Weather) has a SEA section: a Storm slider that takes wind, gusts, cloud, storm cells, waves, swell and whitecaps from calm to a full gale in one move; the waves' significant height (Auto follows the wind, or 0 m glassy to 12 m); the swell's size and direction; whitecaps (0–200%); the water's clarity; and calm lakes. Real weather keeps raising the waves from its wind unless Waves is set by hand. For one run: `--waves 3`, `--swell 2,250`, `--storm 0.8`.

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
