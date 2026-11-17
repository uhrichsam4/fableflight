//! Interface language: English (the default) and French.
//!
//! Text written in the code stays in English. Where it is shown, it goes
//! through one of:
//! - `t!("English", "Français")` for a one-off line, with format arguments
//!   when it has some: `t!("{n} tiles", "{n} tuiles", n = 3)`;
//! - `tr("English")` for text that lives in a table (the controls list,
//!   place names, presets…): it looks the French up in `fr()` below and
//!   falls back to the English;
//! - `caps(..)` for the damage messages the flight model writes in capitals.

use std::sync::atomic::{AtomicU8, Ordering};

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Lang {
    En,
    Fr,
}

impl Lang {
    pub const ALL: [Lang; 2] = [Lang::En, Lang::Fr];

    /// The language's own name, for the language picker.
    pub fn name(self) -> &'static str {
        match self {
            Lang::En => "English",
            Lang::Fr => "Français",
        }
    }

    pub fn code(self) -> &'static str {
        match self {
            Lang::En => "en",
            Lang::Fr => "fr",
        }
    }

    pub fn parse(s: &str) -> Option<Lang> {
        let s = s.trim().to_ascii_lowercase();
        if s.starts_with("fr") {
            Some(Lang::Fr)
        } else if s.starts_with("en") {
            Some(Lang::En)
        } else {
            None
        }
    }

    /// The system's language, from the usual locale variables (Linux, macOS);
    /// English when it is not one we have.
    pub fn detect() -> Lang {
        ["LC_ALL", "LC_MESSAGES", "LANG", "LANGUAGE"]
            .iter()
            .filter_map(|k| std::env::var(k).ok())
            .find(|v| !v.is_empty() && v != "C" && v != "POSIX")
            .and_then(|v| Lang::parse(&v))
            .unwrap_or(Lang::En)
    }
}

static LANG: AtomicU8 = AtomicU8::new(0);

pub fn lang() -> Lang {
    if LANG.load(Ordering::Relaxed) == 1 { Lang::Fr } else { Lang::En }
}

pub fn set_lang(l: Lang) {
    LANG.store(if l == Lang::Fr { 1 } else { 0 }, Ordering::Relaxed);
}

pub fn is_fr() -> bool {
    lang() == Lang::Fr
}

/// `t!("English", "Français")`, or with format arguments
/// `t!("{} km", "{} km", d)`: both strings get the same arguments.
macro_rules! t {
    ($en:literal, $fr:literal $(,)?) => {
        if $crate::i18n::is_fr() { $fr } else { $en }
    };
    ($en:literal, $fr:literal, $($arg:tt)*) => {
        if $crate::i18n::is_fr() { format!($fr, $($arg)*) } else { format!($en, $($arg)*) }
    };
}
pub(crate) use t;

/// Table text in the current language (the English itself when there is no
/// translation for it).
pub fn tr(en: &'static str) -> &'static str {
    if is_fr() { fr(en).unwrap_or(en) } else { en }
}

/// The same for text that is not `'static` (a place name kept in a `String`).
pub fn tr_owned(en: &str) -> String {
    if is_fr() { fr(en).unwrap_or(en).to_string() } else { en.to_string() }
}

/// "{n} request(s)" and the like.
pub fn plural(n: u64, en_one: &str, en_many: &str, fr_one: &str, fr_many: &str) -> String {
    let w = match (is_fr(), n) {
        (false, 1) => en_one,
        (false, _) => en_many,
        // In French, 0 and 1 take the singular.
        (true, 0 | 1) => fr_one,
        (true, _) => fr_many,
    };
    format!("{n} {w}")
}

fn fr(en: &str) -> Option<&'static str> {
    Some(match en {
        // ---- pages of the in-flight panels
        "Flight" => "Vol",
        "Go to" => "Destinations",
        "Weather" => "Météo",
        "Game" => "Jeu",
        "Route" => "Itinéraire",
        "Settings" => "Réglages",
        "Graphics" => "Graphismes",
        "Controls" => "Commandes",

        // ---- controls list: headings
        "FLYING" => "PILOTAGE",
        "VIEW" => "VUE",
        "WORLD" => "MONDE",
        "GAME" => "JEU",
        "GAMEPAD (MSFS 2020 layout)" => "MANETTE (disposition MSFS 2020)",
        "GAMEPAD · FLYING" => "MANETTE · PILOTAGE",
        "GAMEPAD · MENUS" => "MANETTE · MENUS",
        "GAMEPAD · GLOBE" => "MANETTE · GLOBE",
        // keys
        "Space" => "Espace",
        "Mouse drag" => "Glisser souris",
        "Mouse wheel" => "Molette",
        "Left stick" => "Stick gauche",
        "Right stick" => "Stick droit",
        "Left stick click" => "Clic stick gauche",
        "Right stick click" => "Clic stick droit",
        "Y + D-pad up / dn" => "Y + croix haut / bas",
        "D-pad" => "Croix",
        "A after a crash" => "A après un crash",
        // what they do
        "pitch down / up" => "piquer / cabrer",
        "roll left / right" => "roulis gauche / droite",
        "rudder left / right" => "palonnier gauche / droite",
        "throttle up / down" => "plus / moins de gaz",
        "boost" => "boost (postcombustion)",
        "wheel brakes" => "freins de roues",
        "flaps down / up a notch" => "volets : sortir / rentrer d'un cran",
        "landing gear down / up" => "train d'atterrissage : sortir / rentrer",
        "speedbrake (airliners)" => "aérofreins (avions de ligne)",
        "thrust reversers, on the ground (hold)" => "inverseurs de poussée, au sol (maintenir)",
        "set the heading indicator to the compass" => "recaler le conservateur de cap sur le compas",
        "chase / cockpit camera" => "caméra extérieure / cockpit",
        "look around" => "regarder autour",
        "camera distance / cockpit zoom" => "distance caméra / zoom cockpit",
        "hide / show the HUD" => "masquer / afficher le HUD",
        "large map (wheel zoom, drag pan)" => "grande carte (molette : zoom, glisser : déplacer)",
        "screenshot" => "capture d'écran",
        "time of day forward / back" => "avancer / reculer l'heure",
        "more / fewer clouds" => "plus / moins de nuages",
        "weather & time window" => "fenêtre météo et heure",
        "sim speed slower / faster (1× to 8×)" => "vitesse de simulation moins / plus vite (1× à 8×)",
        "autopilot: fly to the next route waypoint" => "pilote automatique : vers le prochain point de l'itinéraire",
        "flight panel: instruments, places, weather" => "panneau de vol : instruments, destinations, météo",
        "pause menu: settings, globe, quit" => "menu pause : réglages, globe, quitter",
        "pause without the menu" => "pause sans le menu",
        "go to a place or address" => "aller à un lieu ou une adresse",
        "respawn in the air" => "réapparaître en vol",
        "sound on / off" => "son activé / coupé",
        "anti-aliasing (TAA) on / off" => "anticrénelage (TAA) activé / désactivé",
        "frame rate counter" => "compteur d'images par seconde",
        "fullscreen on / off" => "plein écran activé / désactivé",
        "quick key reference" => "aide-mémoire des touches",
        "pitch and roll" => "tangage et roulis",
        "throttle up / down (hold)" => "plus / moins de gaz (maintenir)",
        "wheel brakes (hold)" => "freins de roues (maintenir)",
        "parking brake on / off" => "frein de parc serré / desserré",
        "trim nose down / up" => "trim à piquer / à cabrer",
        "flaps up / down a notch" => "volets : rentrer / sortir d'un cran",
        "landing gear" => "train d'atterrissage",
        "reset the view" => "réinitialiser la vue",
        "quick views" => "vues rapides",
        "fly again after a crash" => "repartir après un crash",
        "pause menu (B closes it)" => "menu pause (B pour fermer)",
        "flight panel" => "panneau de vol",
        "fly again" => "repartir",
        "pause menu" => "menu pause",
        "flight panel (beside the flying)" => "panneau de vol (pendant le vol)",
        "move between choices" => "naviguer entre les choix",
        "press / choose" => "appuyer / choisir",
        "slide a slider" => "déplacer un curseur",
        "turn the page" => "changer de page",
        "back, close" => "retour, fermer",
        "turn the globe" => "tourner le globe",
        "zoom out / in" => "dézoomer / zoomer",
        "start at the crosshair" => "partir depuis le viseur",
        "take off" => "décoller",
        "settings and this controls list" => "réglages et cette liste des commandes",
        // gamepad prompts
        "move" => "naviguer",
        "press" => "appuyer",
        "page" => "page",
        "resume" => "reprendre",
        "panel" => "panneau",
        "menu" => "menu",
        "respawn" => "réapparaître",
        "go somewhere else" => "aller ailleurs",
        "menu: respawn, globe" => "menu : réapparaître, globe",

        // ---- weather
        "DAY" => "JOUR",
        "SKY" => "CIEL",
        "WIND" => "VENT",
        "PRESETS" => "PRÉRÉGLAGES",
        "Clear" => "Dégagé",
        "Fair" => "Beau temps",
        "Hazy" => "Brumeux",
        "Overcast" => "Couvert",
        "Windy" => "Venteux",
        "Storms" => "Orages",
        "new" => "nouvelle lune",
        "waxing crescent" => "premier croissant",
        "first quarter" => "premier quartier",
        "waxing gibbous" => "gibbeuse croissante",
        "full" => "pleine lune",
        "waning gibbous" => "gibbeuse décroissante",
        "last quarter" => "dernier quartier",
        "waning crescent" => "dernier croissant",
        "SW" => "SO",
        "W" => "O",
        "NW" => "NO",

        // ---- aircraft
        "Twin-engine carrier fighter, 15.5 t. Fly-by-wire holds it to 7.5 g; Space for afterburner, / speedbrake. 135 kt approach." => "Chasseur embarqué biréacteur, 15,5 t. Les commandes de vol électriques le limitent à 7,5 g ; Espace pour la postcombustion, / aérofreins. Approche à 135 kt.",
        "Four-seat single, 180 hp, retractable gear. Forgiving and slow: 125 kt flat out." => "Monomoteur quatre places, 180 ch, train rentrant. Indulgent et lent : 125 kt à fond.",
        "Speedboat" => "Bateau à moteur",
        "7.4 m centre-console runabout, 300 hp outboard. W/S throttle and reverse, A/D steer, Shift for the last few knots: about 40 kt flat out." => "Open de 7,4 m à console centrale, hors-bord de 300 ch. W/S gaz et marche arrière, A/D barre, Maj pour les derniers nœuds : environ 40 nœuds à fond.",
        "Four-engine wide-body airliner, 290 t. Heavy and slow to answer the controls; Mach 0.85 cruise, 150 kt approach. / speedbrake, Z reversers." => "Quadriréacteur gros-porteur, 290 t. Lourd et lent à répondre aux commandes ; croisière à Mach 0,85, approche à 150 kt. / aérofreins, Z inverseurs.",
        "The 747-400 at five times its size: 353 m long, 322 m across, 7,250 t. Same speeds and handling as the 747. / speedbrake, Z reversers." => "Le 747-400 cinq fois plus grand : 353 m de long, 322 m d'envergure, 7 250 t. Mêmes vitesses et même pilotage que le 747. / aérofreins, Z inverseurs.",

        // ---- places (cities, one-click places, airports)
        "Mexico City" => "Mexico",
        "Havana" => "La Havane",
        "London" => "Londres",
        "Lisbon" => "Lisbonne",
        "Venice" => "Venise",
        "Athens" => "Athènes",
        "Moscow" => "Moscou",
        "Cairo" => "Le Caire",
        "Marrakesh" => "Marrakech",
        "Cape Town" => "Le Cap",
        "Dubai" => "Dubaï",
        "Tehran" => "Téhéran",
        "Kathmandu" => "Katmandou",
        "Singapore" => "Singapour",
        "Beijing" => "Pékin",
        "Seoul" => "Séoul",
        "Manila" => "Manille",
        "Yosemite Valley" => "Vallée de Yosemite",
        "Matterhorn" => "Cervin",
        "Sydney Harbour" => "Port de Sydney",
        "Black Forest" => "Forêt-Noire",
        "London Heathrow" => "Londres Heathrow",
        "St Maarten" => "Saint-Martin",

        _ => return None,
    })
}

/// Damage messages ("LEFT WING TORN OFF (FLUTTER)", "R AILERON GONE",
/// "ENGINE 40%  OIL LEAK") in the current language.
pub fn caps(s: &str) -> String {
    if !is_fr() {
        return s.to_string();
    }
    // The cause, in brackets at the end.
    let (body, why) = match s.strip_suffix(')').and_then(|b| b.rsplit_once(" (")) {
        Some((b, w)) => (b, Some(w)),
        None => (s, None),
    };
    // Which side, at the front.
    let (side, body) = [("LEFT ", 0), ("RIGHT ", 1), ("L ", 0), ("R ", 1)]
        .iter()
        .find_map(|(p, k)| body.strip_prefix(p).map(|b| (Some(*k), b)))
        .unwrap_or((None, body));
    // The longest phrase it starts with.
    let found = CAPS.iter().filter(|(en, ..)| body.starts_with(en)).max_by_key(|(en, ..)| en.len());
    let mut out = match found {
        Some((en, fr, fem)) => {
            let s = match side {
                Some(0) => " GAUCHE",
                Some(_) if *fem => " DROITE",
                Some(_) => " DROIT",
                None => "",
            };
            let rest = body[en.len()..].replace("OIL LEAK", "FUITE D'HUILE");
            format!("{}{rest}", fr.replace(" {s}", s))
        }
        None => s.to_string(),
    };
    if found.is_some() {
        if let Some(w) = why {
            let w = CAUSES.iter().find(|(en, _)| *en == w).map(|(_, fr)| *fr).unwrap_or(w);
            out = format!("{out} ({w})");
        }
    }
    out
}

/// (English, French with `{s}` where the side goes, whether the French noun
/// is feminine — "AILE DROITE" but "VOLET DROIT")
const CAPS: &[(&str, &str, bool)] = &[
    ("WING BUCKLED, HANGING", "AILE {s} FLAMBÉE, PENDANTE", true),
    ("WING BUCKLED", "AILE {s} FLAMBÉE", true),
    ("WING TORN OFF", "AILE {s} ARRACHÉE", true),
    ("WING TORN AWAY", "AILE {s} ARRACHÉE", true),
    ("WING", "AILE {s}", true),
    ("FUEL TANK RUPTURED", "RÉSERVOIR {s} PERCÉ", false),
    ("TANK RUPTURED", "RÉSERVOIR {s} PERCÉ", false),
    ("AILERON HINGE BROKEN", "CHARNIÈRE D'AILERON {s} CASSÉE", false),
    ("AILERON TORN OFF", "AILERON {s} ARRACHÉ", false),
    ("AILERON GONE", "AILERON {s} PERDU", false),
    ("AILERON HANGING", "AILERON {s} PENDANT", false),
    ("FLAP TORN OFF", "VOLET {s} ARRACHÉ", false),
    ("FLAP GONE", "VOLET {s} PERDU", false),
    ("STABILISER TORN OFF", "STABILISATEUR {s} ARRACHÉ", false),
    ("STABILISER GONE", "STABILISATEUR {s} PERDU", false),
    ("ELEVATOR TORN OFF", "PROFONDEUR {s} ARRACHÉE", true),
    ("ELEVATOR GONE", "PROFONDEUR {s} PERDUE", true),
    ("ENGINE POD STRUCK", "NACELLE MOTEUR {s} TOUCHÉE", true),
    ("TAIL BROKEN OFF", "QUEUE ARRACHÉE", true),
    ("RUDDER TORN OFF", "GOUVERNE DE DIRECTION ARRACHÉE", true),
    ("RUDDER DAMAGED", "GOUVERNE DE DIRECTION ENDOMMAGÉE", true),
    ("RUDDER GONE", "GOUVERNE DE DIRECTION PERDUE", true),
    ("RUDDER", "DIRECTION", true),
    ("FIN BROKEN", "DÉRIVE CASSÉE", true),
    ("FIN", "DÉRIVE", true),
    ("PROPELLER BLADE BROKEN", "PALE D'HÉLICE CASSÉE", true),
    ("PROP BLADE BROKEN", "PALE D'HÉLICE CASSÉE", true),
    ("PROPELLER GONE", "HÉLICE PERDUE", true),
    ("ENGINE SEIZED", "MOTEUR GRIPPÉ", false),
    ("ENGINE DAMAGED, LOSING OIL", "MOTEUR ENDOMMAGÉ, FUITE D'HUILE", false),
    ("ENGINE DAMAGED", "MOTEUR ENDOMMAGÉ", false),
    ("ENGINES FAILED", "MOTEURS EN PANNE", false),
    ("ENGINE FAILED", "MOTEUR EN PANNE", false),
    ("ENGINES DEAD", "MOTEURS HORS SERVICE", false),
    ("ENGINE DEAD", "MOTEUR HORS SERVICE", false),
    ("ENGINES", "MOTEURS", false),
    ("ENGINE", "MOTEUR", false),
    ("OIL ON WINDSHIELD", "HUILE SUR LE PARE-BRISE", true),
    ("WINDSHIELD CRACKED", "PARE-BRISE FISSURÉ", false),
    ("FUSELAGE CRUMPLED", "FUSELAGE ÉCRASÉ", false),
    ("FUSELAGE DENTED", "FUSELAGE CABOSSÉ", false),
    ("MAIN GEAR TORN OFF", "TRAIN PRINCIPAL {s} ARRACHÉ", false),
    ("MAIN GEAR GONE", "TRAIN PRINCIPAL {s} PERDU", false),
    ("WING GEAR TORN OFF", "TRAIN D'AILE {s} ARRACHÉ", false),
    ("WING GEAR GONE", "TRAIN D'AILE {s} PERDU", false),
    ("BODY GEAR TORN OFF", "TRAIN DE FUSELAGE {s} ARRACHÉ", false),
    ("BODY GEAR GONE", "TRAIN DE FUSELAGE {s} PERDU", false),
    ("NOSE GEAR TORN OFF", "TRAIN AVANT ARRACHÉ", false),
    ("NOSE GEAR GONE", "TRAIN AVANT PERDU", false),
];

/// What broke it, in brackets after a damage message.
const CAUSES: &[(&str, &str)] = &[
    ("FLUTTER", "FLOTTEMENT"),
    ("BUILDING", "BÂTIMENT"),
    ("GROUND", "SOL"),
    ("WATER", "EAU"),
    ("TREE", "ARBRE"),
    ("HARD IMPACT", "CHOC VIOLENT"),
    ("HARD LANDING", "ATTERRISSAGE DUR"),
    ("NOSE IMPACT", "CHOC DU NEZ"),
    ("ROLLED OVER", "RETOURNÉ"),
    ("OVER G", "FACTEUR DE CHARGE"),
    ("OVERSPEED", "SURVITESSE"),
    ("AIRFLOW", "ÉCOULEMENT D'AIR"),
];

#[cfg(test)]
mod tests {
    use super::*;

    // One test touches the global language, so the others cannot race it.
    #[test]
    fn french() {
        assert_eq!(caps("LEFT WING TORN OFF (FLUTTER)"), "LEFT WING TORN OFF (FLUTTER)");
        set_lang(Lang::Fr);
        assert_eq!(caps("LEFT WING TORN OFF (FLUTTER)"), "AILE GAUCHE ARRACHÉE (FLOTTEMENT)");
        assert_eq!(caps("R AILERON GONE"), "AILERON DROIT PERDU");
        assert_eq!(caps("RIGHT ELEVATOR TORN OFF (BUILDING)"), "PROFONDEUR DROITE ARRACHÉE (BÂTIMENT)");
        assert_eq!(caps("L WING  -1.3 m"), "AILE GAUCHE  -1.3 m");
        assert_eq!(caps("ENGINE 40%  OIL LEAK"), "MOTEUR 40%  FUITE D'HUILE");
        assert_eq!(caps("LEFT MAIN GEAR GONE"), "TRAIN PRINCIPAL GAUCHE PERDU");
        assert_eq!(caps("NOSE GEAR TORN OFF (HARD IMPACT)"), "TRAIN AVANT ARRACHÉ (CHOC VIOLENT)");
        assert_eq!(caps("TAIL BROKEN OFF (OVER G)"), "QUEUE ARRACHÉE (FACTEUR DE CHARGE)");
        assert_eq!(caps("SOMETHING NEW"), "SOMETHING NEW");
        assert_eq!(tr("London"), "Londres");
        assert_eq!(tr("Paris"), "Paris");
        let n = 3;
        assert_eq!(t!("{n} km", "{n} kilomètres", n = n), "3 kilomètres");
        assert_eq!(plural(1, "request", "requests", "requête", "requêtes"), "1 requête");
        assert_eq!(plural(0, "request", "requests", "requête", "requêtes"), "0 requête");
        set_lang(Lang::En);
        assert_eq!(t!("{n} km", "{n} kilomètres", n = n), "3 km");
        assert_eq!(plural(0, "request", "requests", "requête", "requêtes"), "0 requests");
        assert_eq!(Lang::parse("fr_BE.UTF-8"), Some(Lang::Fr));
    }
}
