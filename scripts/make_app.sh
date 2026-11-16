#!/bin/zsh
# Build "Samiam's Flight Sim.app" from the release binary and install it in /Applications
# (Spotlight finds it). Game Mode keys included. Usage: scripts/make_app.sh
set -e
cd "$(dirname $0)/.."
taskpolicy -c utility cargo build --release -j 4
APP="/Applications/Samiam's Flight Sim.app"
T=$(mktemp -d)
B="$T/Samiam's Flight Sim.app/Contents"
mkdir -p "$B/MacOS" "$B/Resources"
cp target/release/fablegame "$B/MacOS/samiams-flight-sim"
# icon from the Cessna picture
I="$T/icon.iconset"; mkdir -p "$I"
for s in 16 32 64 128 256 512; do
  sips -z $s $s assets/aircraft/c172.png --out "$I/icon_${s}x${s}.png" >/dev/null
  sips -z $((s*2)) $((s*2)) assets/aircraft/c172.png --out "$I/icon_${s}x${s}@2x.png" >/dev/null
done
iconutil -c icns "$I" -o "$B/Resources/AppIcon.icns"
cat > "$B/Info.plist" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>Samiam's Flight Sim</string>
  <key>CFBundleDisplayName</key><string>Samiam's Flight Sim</string>
  <key>CFBundleIdentifier</key><string>local.samiam.flightsim</string>
  <key>CFBundleExecutable</key><string>samiams-flight-sim</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>LSApplicationCategoryType</key><string>public.app-category.simulation-games</string>
  <key>LSSupportsGameMode</key><true/>
  <key>NSHighResolutionCapable</key><true/>
</dict></plist>
PL
codesign --force --deep -s - "$T/Samiam's Flight Sim.app" 2>/dev/null || true
rm -rf "$APP"; mv "$T/Samiam's Flight Sim.app" "$APP"; rm -rf "$T"
touch "$APP"; mdimport "$APP" 2>/dev/null || true
echo "installed: $APP"
