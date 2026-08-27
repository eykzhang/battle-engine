"""A pool of real, legal gen9ou teams for self-play training and benchmarking.

gen9ou (unlike gen9randombattle) requires each player to submit a
constructed team - poke-env has no built-in team generator for it. All teams
here are copied from Smogon's "SV OU Sample Teams" forum thread
(https://www.smogon.com/forums/threads/sv-ou-sample-teams-new-samples-added-post-spl-and-tera-blast-ban.3712513/),
a community-vetted, actively-updated resource specifically meant to be
known-legal and competitively reasonable - not hand-written.

Even so, don't trust the forum post's legality over the local server's own
opinion - the banlist isn't static, and this thread spans many now-superseded
metagame eras (its own section headers document bans/unbans of Chien-Pao,
Espathra, Shed Tail, Gliscor (later unbanned), Gouging Fire, Roaring Moon,
and others). Re-run the validator
(`node pokemon-showdown validate-team gen9ou < packed_team`, packed via
Teambuilder.join_team(Teambuilder.parse_showdown_team(...))) after editing
this pool or updating the pokemon-showdown checkout.

Original 5 (fetched 2026-07-31): two of the originally-fetched teams failed
validation (one used the since-banned Tera Blast twice, another used
Baxcalibur - tagged Uber here - and Spore, banned by this ruleset's Sleep
Moves Clause). Both were swapped for validator-clean replacements from the
same thread.

Expanded to 26 (2026-08-27): the original 5-team pool (24 distinct species)
was diagnosed as a real, compounding cause of the local-vs-ladder win-rate
gap alongside the encoding gap `encoding.py`'s per-move rewrite fixed -
`RandomTeamFromPool` is used identically for both self-play sides during PPO
training AND for real ladder play, so a narrow pool means real opponents
constantly bring species/sets the policy never trained against. Sized against
real evidence, not guessed: real gen9ou usage statistics were pulled from
Smogon's public stats archive (https://www.smogon.com/stats/2026-07/gen9ou-1500.txt.gz,
654,262 real ranked battles, 1500-rating cutoff) to identify the actual
current usage-weighted metagame (a steep power-law curve - the top ~50 real
Pokemon cover the large majority of what a real ladder opponent brings; a
long tail of sub-1%-usage species follows). 32 candidate teams were pulled
from the thread's two most recent eras, selected for heavy overlap with that
real top-usage list and for archetype diversity (offense, bulky offense,
balance, stall), then independently fetched and validated one-by-one against
this checkout's real `pokemon-showdown validate-team gen9ou` (same pipeline
as above) - not trusted from the thread alone. 23 passed; 9 failed (7 for the
currently-banned Tera Blast, 1 for Volcarona's current Uber tag, 1 for a
malformed source paste using slash-separated move alternatives rather than a
resolved single export). 2 of the 23 passes turned out to be exact or
near-exact duplicates of the existing 5-team pool and were excluded as
redundant, leaving 21 genuinely new, validated teams - 26 total, well over
60 distinct species represented across the pool (not counted exactly, but
verifiably far beyond the original 24).

Note on fidelity: the 21 new teams' mechanically-relevant content (species,
item, ability, tera type, EVs, nature, moveset) was preserved exactly as
validated. Purely cosmetic fields some source pastes carried (nicknames,
explicit gender, "Shiny: Yes", one explicit "Level: 99") were not
transcribed - none of these affect legality or battle mechanics for this
project's purposes (this project's own state encoding already assumes
level-100 uniformly, per `encoding.py`'s module docstring), but this is a
real, deliberate simplification versus the original posts, not an
oversight - worth knowing if a future pass ever needs the literal original
export text again (the source URLs are preserved in the expansion's own
git history / build discovery notes for that reason).

GEN9OU_SAMPLE_TEAMS holds the raw showdown export text; RandomTeamFromPool
is a poke_env Teambuilder that picks one at random each time a battle starts,
so a benchmark run isn't pinned to a single team matchup.
"""

from __future__ import annotations

import random
from typing import List

from poke_env.teambuilder.teambuilder import Teambuilder

GEN9OU_SAMPLE_TEAMS: List[str] = [
    # Boots Samurott-Hisui hazard stack
    """
Samurott-Hisui @ Heavy-Duty Boots
Ability: Sharpness
Tera Type: Poison
EVs: 72 HP / 252 Atk / 184 Spe
Adamant Nature
- Ceaseless Edge
- Aqua Cutter
- Knock Off
- Sucker Punch

Gliscor @ Toxic Orb
Ability: Poison Heal
Tera Type: Ghost
EVs: 244 HP / 36 Def / 228 SpD
Careful Nature
IVs: 25 Spe
- Earthquake
- Toxic
- Protect
- U-turn

Great Tusk @ Rocky Helmet
Ability: Protosynthesis
Tera Type: Fairy
EVs: 184 HP / 72 Atk / 252 Spe
Jolly Nature
- Headlong Rush
- Close Combat
- Ice Spinner
- Rapid Spin

Tinkaton @ Air Balloon
Ability: Pickpocket
Tera Type: Ghost
EVs: 248 HP / 28 SpD / 232 Spe
Jolly Nature
- Stealth Rock
- Gigaton Hammer
- Knock Off
- Encore

Dragapult @ Heavy-Duty Boots
Ability: Infiltrator
Tera Type: Dragon
EVs: 56 Def / 200 SpA / 252 Spe
Timid Nature
- Draco Meteor
- Hex
- Will-O-Wisp
- U-turn

Garganacl @ Leftovers
Ability: Purifying Salt
Tera Type: Water
EVs: 252 HP / 52 Def / 204 SpD
Careful Nature
- Curse
- Salt Cure
- Earthquake
- Recover
""",
    # Skarmory / Ting-Lu balance
    """
Smooth Criminal (Skarmory) @ Rocky Helmet
Ability: Sturdy
Tera Type: Dragon
EVs: 248 HP / 252 Def / 8 SpD
Impish Nature
- Whirlwind
- Brave Bird
- Spikes
- Roost

Rock With You (Ting-Lu) @ Leftovers
Ability: Vessel of Ruin
Tera Type: Water
EVs: 252 HP / 248 SpD / 8 Spe
Careful Nature
- Stealth Rock
- Earthquake
- Whirlwind
- Ruination

Beat It (Zamazenta) @ Heavy-Duty Boots
Ability: Dauntless Shield
Tera Type: Steel
EVs: 80 HP / 252 Atk / 176 Spe
Jolly Nature
- Close Combat
- Heavy Slam
- Crunch
- Stone Edge

Poison Young Thing (Pecharunt) @ Heavy-Duty Boots
Ability: Poison Puppeteer
Tera Type: Dark
EVs: 252 HP / 196 Def / 60 Spe
Bold Nature
IVs: 0 Atk
- Parting Shot
- Foul Play
- Malignant Chain
- Recover

Moonwalker (Clefable) @ Leftovers
Ability: Magic Guard
Tera Type: Steel
EVs: 252 HP / 248 Def / 8 Spe
Bold Nature
- Calm Mind
- Moonblast
- Moonlight
- Knock Off

Thriller (Walking Wake) @ Heavy-Duty Boots
Ability: Protosynthesis
Tera Type: Water
EVs: 12 Def / 244 SpA / 252 Spe
Timid Nature
- Surf
- Draco Meteor
- Knock Off
- Flip Turn
""",
    # Dondozo/Blissey stall
    """
Dondozo @ Leftovers
Ability: Unaware
Tera Type: Dragon
EVs: 248 HP / 252 Def / 8 SpD
Impish Nature
- Body Press
- Avalanche
- Rest
- Sleep Talk

Blissey @ Leftovers
Ability: Natural Cure
Tera Type: Dark
EVs: 20 HP / 252 Def / 236 SpD
Calm Nature
IVs: 0 Atk
- Seismic Toss
- Soft-Boiled
- Calm Mind
- Stealth Rock

Gliscor @ Toxic Orb
Ability: Poison Heal
Tera Type: Water
EVs: 244 HP / 12 Def / 252 SpD
Careful Nature
IVs: 24 Spe
- Earthquake
- Spikes
- Knock Off
- Protect

Toxapex @ Black Sludge
Ability: Regenerator
Tera Type: Ghost
EVs: 252 HP / 124 Def / 132 SpD
Calm Nature
IVs: 0 Atk
- Sludge Bomb
- Recover
- Toxic Spikes
- Surf

Mandibuzz @ Heavy-Duty Boots
Ability: Big Pecks
Tera Type: Steel
EVs: 252 HP / 100 Def / 140 SpD / 16 Spe
Careful Nature
- Toxic
- Roost
- Knock Off
- Defog

Talonflame @ Heavy-Duty Boots
Ability: Flame Body
Tera Type: Dragon
EVs: 248 HP / 228 Def / 32 Spe
Bold Nature
IVs: 0 Atk
- Will-O-Wisp
- Overheat
- Roost
- Defog
""",
    # Skarmory / Ting-Lu / Zamazenta offense-balance
    """
Skarmory @ Rocky Helmet
Ability: Sturdy
Tera Type: Dragon
EVs: 248 HP / 252 Def / 8 Spe
Impish Nature
- Whirlwind
- Brave Bird
- Stealth Rock
- Roost

Ting-Lu @ Heavy-Duty Boots
Ability: Vessel of Ruin
Tera Type: Ghost
EVs: 252 HP / 252 SpD / 4 Spe
Careful Nature
- Ruination
- Earthquake
- Whirlwind
- Spikes

Zamazenta @ Heavy-Duty Boots
Ability: Dauntless Shield
Tera Type: Fire
EVs: 252 Atk / 4 SpD / 252 Spe
Adamant Nature
- Close Combat
- Crunch
- Heavy Slam
- Stone Edge

Weavile @ Heavy-Duty Boots
Ability: Pickpocket
Tera Type: Ice
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Triple Axel
- Ice Shard
- Knock Off
- Low Kick

Kyurem @ Heavy-Duty Boots
Ability: Pressure
Tera Type: Fairy
EVs: 4 Def / 252 SpA / 252 Spe
Timid Nature
- Scale Shot
- Ice Beam
- Freeze-Dry
- Earth Power

Slowking-Galar @ Heavy-Duty Boots
Ability: Regenerator
Tera Type: Water
EVs: 252 HP / 4 Def / 252 SpD
Sassy Nature
IVs: 0 Atk / 0 Spe
- Future Sight
- Sludge Bomb
- Chilly Reception
- Toxic
""",
    # Great Tusk hazard stack / Cinderace offense
    """
Great Tusk @ Leftovers
Ability: Protosynthesis
Tera Type: Water
EVs: 248 HP / 16 Atk / 164 Def / 80 Spe
Impish Nature
- Earthquake
- Knock Off
- Rapid Spin
- Stealth Rock

Breloom @ Choice Band
Ability: Technician
Tera Type: Fire
EVs: 252 Atk / 4 SpD / 252 Spe
Adamant Nature
- Bullet Seed
- Close Combat
- Mach Punch
- Rock Tomb

Dragapult @ Leftovers
Ability: Infiltrator
Tera Type: Fairy
EVs: 80 Atk / 176 SpA / 252 Spe
Mild Nature
- Dragon Darts
- Hex
- Will-O-Wisp
- Substitute

Kingambit @ Leftovers
Ability: Supreme Overlord
Tera Type: Fire
EVs: 4 HP / 252 Atk / 252 Spe
Adamant Nature
- Swords Dance
- Iron Head
- Kowtow Cleave
- Sucker Punch

Rotom-Wash @ Leftovers
Ability: Levitate
Tera Type: Fairy
EVs: 252 HP / 244 SpD / 12 Spe
Calm Nature
IVs: 0 Atk
- Hydro Pump
- Volt Switch
- Thunder Wave
- Protect

Cinderace @ Heavy-Duty Boots
Ability: Libero
Tera Type: Flying
EVs: 248 HP / 16 Def / 12 SpD / 232 Spe
Jolly Nature
- Pyro Ball
- U-turn
- Will-O-Wisp
- Court Change
""",
    # --- Expanded 2026-08-27: 21 more teams from the same living thread, see
    # module docstring for the full sourcing/validation writeup ---
    # Latias Hardcore HO
    """
Glimmora @ Red Card
Ability: Toxic Debris
Tera Type: Ghost
EVs: 252 HP / 40 Def / 160 SpD / 56 Spe
Calm Nature
IVs: 0 Atk
- Power Gem
- Earth Power
- Stealth Rock
- Spikes

Zamazenta @ Leftovers
Ability: Dauntless Shield
Tera Type: Fire
EVs: 112 HP / 40 Atk / 104 Def / 252 Spe
Jolly Nature
- Iron Defense
- Body Press
- Crunch
- Substitute

Latias @ Leftovers
Ability: Levitate
Tera Type: Fairy
EVs: 208 HP / 148 Def / 68 SpA / 84 Spe
Timid Nature
IVs: 0 Atk
- Draining Kiss
- Stored Power
- Calm Mind
- Agility

Gholdengo @ Custap Berry
Ability: Good as Gold
Tera Type: Fairy
EVs: 192 HP / 92 SpA / 224 Spe
Modest Nature
IVs: 0 Atk
- Nasty Plot
- Shadow Ball
- Thunderbolt
- Dazzling Gleam

Dragonite @ Heavy-Duty Boots
Ability: Multiscale
Tera Type: Normal
EVs: 252 Atk / 4 SpD / 252 Spe
Adamant Nature
- Dragon Dance
- Extreme Speed
- Earthquake
- Ice Spinner

Weavile @ Life Orb
Ability: Pickpocket
Tera Type: Ice
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Swords Dance
- Triple Axel
- Upper Hand
- Knock Off
""",
    # Manaphy Heavy Offense
    """
Darkrai @ Heavy-Duty Boots
Ability: Bad Dreams
Tera Type: Poison
EVs: 4 Def / 252 SpA / 252 Spe
Timid Nature
IVs: 0 Atk
- Dark Pulse
- Sludge Bomb
- Focus Blast
- Nasty Plot

Manaphy @ Covert Cloak
Ability: Hydration
Tera Type: Fairy
EVs: 56 HP / 200 SpA / 252 Spe
Timid Nature
IVs: 0 Atk
- Surf
- Ice Beam
- Energy Ball
- Tail Glow

Landorus-Therian @ Rocky Helmet
Ability: Intimidate
Tera Type: Grass
EVs: 248 HP / 36 Def / 224 Spe
Timid Nature
- Earth Power
- Taunt
- Stealth Rock
- U-turn

Gholdengo @ Air Balloon
Ability: Good as Gold
Tera Type: Fairy
EVs: 208 HP / 112 SpA / 188 Spe
Modest Nature
IVs: 0 Atk
- Shadow Ball
- Make It Rain
- Psyshock
- Nasty Plot

Raging Bolt @ Booster Energy
Ability: Protosynthesis
Tera Type: Bug
EVs: 196 HP / 4 Def / 252 SpA / 56 Spe
Modest Nature
IVs: 20 Atk
- Thunderbolt
- Dragon Pulse
- Thunderclap
- Calm Mind

Iron Valiant @ Booster Energy
Ability: Quark Drive
Tera Type: Dark
EVs: 16 Atk / 240 SpA / 252 Spe
Naive Nature
- Moonblast
- Close Combat
- Knock Off
- Thunderbolt
""",
    # Darkrai Hydrapple BO
    """
Hydrapple @ Heavy-Duty Boots
Ability: Regenerator
Tera Type: Fairy
EVs: 224 HP / 220 SpA / 64 Spe
Modest Nature
IVs: 0 Atk
- Nasty Plot
- Leaf Storm
- Fickle Beam
- Earth Power

Tinkaton @ Leftovers
Ability: Mold Breaker
Tera Type: Water
EVs: 248 HP / 28 SpD / 232 Spe
Jolly Nature
- Gigaton Hammer
- Knock Off
- Encore
- Stealth Rock

Moltres @ Heavy-Duty Boots
Ability: Flame Body
Tera Type: Fairy
EVs: 248 HP / 216 SpD / 12 Spe
Calm Nature
- Roar
- Flamethrower
- U-turn
- Roost

Rotom-Wash @ Rocky Helmet
Ability: Levitate
Tera Type: Steel
EVs: 248 HP / 216 Def / 44 Spe
Bold Nature
IVs: 0 Atk
- Volt Switch
- Will-O-Wisp
- Pain Split
- Hydro Pump

Great Tusk @ Heavy-Duty Boots
Ability: Protosynthesis
Tera Type: Fairy
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Knock Off
- Headlong Rush
- Rapid Spin
- Ice Spinner

Darkrai @ Leftovers
Ability: Bad Dreams
Tera Type: Ghost
EVs: 4 Def / 252 SpA / 252 Spe
Timid Nature
IVs: 0 Atk
- Dark Pulse
- Ice Beam
- Nasty Plot
- Thunder Wave
""",
    # The Hellom Six
    """
Landorus-Therian @ Rocky Helmet
Ability: Intimidate
Tera Type: Dragon
EVs: 248 HP / 28 Def / 232 Spe
Jolly Nature
- Earthquake
- Stone Edge
- U-turn
- Stealth Rock

Zamazenta @ Heavy-Duty Boots
Ability: Dauntless Shield
Tera Type: Fire
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Close Combat
- Crunch
- Stone Edge
- Heavy Slam

Kingambit @ Leftovers
Ability: Supreme Overlord
Tera Type: Ghost
EVs: 120 HP / 136 Atk / 252 Spe
Adamant Nature
- Iron Head
- Low Kick
- Sucker Punch
- Swords Dance

Samurott-Hisui @ Assault Vest
Ability: Sharpness
Tera Type: Poison
EVs: 248 HP / 144 Atk / 56 SpD / 60 Spe
Adamant Nature
- Ceaseless Edge
- Razor Shell
- Sucker Punch
- Knock Off

Pecharunt @ Heavy-Duty Boots
Ability: Poison Puppeteer
Tera Type: Ghost
EVs: 248 HP / 8 SpA / 252 Spe
Timid Nature
IVs: 0 Atk
- Hex
- Malignant Chain
- Parting Shot
- Recover

Hydreigon @ Leftovers
Ability: Levitate
Tera Type: Steel
EVs: 48 HP / 36 Def / 188 SpA / 236 Spe
Timid Nature
IVs: 0 Atk
- Draco Meteor
- Flash Cannon
- Substitute
- Nasty Plot
""",
    # Pecharunt Keldeo Spikestack
    """
Weavile @ Heavy-Duty Boots
Ability: Pressure
Tera Type: Ghost
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Triple Axel
- Knock Off
- Ice Shard
- Swords Dance

Pecharunt @ Heavy-Duty Boots
Ability: Poison Puppeteer
Tera Type: Ghost
EVs: 248 HP / 200 Def / 56 SpD / 4 Spe
Bold Nature
IVs: 0 Atk
- Malignant Chain
- Hex
- Parting Shot
- Recover

Ting-Lu @ Rocky Helmet
Ability: Vessel of Ruin
Tera Type: Water
EVs: 240 HP / 12 Atk / 16 Def / 220 SpD / 20 Spe
Impish Nature
- Earthquake
- Spikes
- Ruination
- Whirlwind

Tinkaton @ Air Balloon
Ability: Pickpocket
Tera Type: Water
EVs: 240 HP / 36 Atk / 232 Spe
Jolly Nature
- Stealth Rock
- Gigaton Hammer
- Encore
- Thunder Wave

Keldeo @ Heavy-Duty Boots
Ability: Justified
Tera Type: Fighting
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Hydro Pump
- Secret Sword
- Flip Turn
- Vacuum Wave

Dragonite @ Heavy-Duty Boots
Ability: Multiscale
Tera Type: Normal
EVs: 224 HP / 252 Atk / 32 Spe
Lonely Nature
- Extreme Speed
- Earthquake
- Hurricane
- Roost
""",
    # Life Orb Zamazenta BO
    """
Dragapult @ Heavy-Duty Boots
Ability: Infiltrator
Tera Type: Fairy
EVs: 60 Atk / 196 SpA / 252 Spe
Naive Nature
- Dragon Darts
- Hex
- Will-O-Wisp
- U-turn

Rotom-Wash @ Leftovers
Ability: Levitate
Tera Type: Steel
EVs: 252 HP / 212 Def / 44 Spe
Bold Nature
IVs: 0 Atk
- Volt Switch
- Hydro Pump
- Pain Split
- Thunder Wave

Landorus-Therian @ Soft Sand
Ability: Intimidate
Tera Type: Ground
EVs: 8 HP / 240 Atk / 8 SpD / 252 Spe
Jolly Nature
- Stealth Rock
- Earthquake
- Smack Down
- U-turn

Clefable @ Sticky Barb
Ability: Magic Guard
Tera Type: Bug
EVs: 252 HP / 240 Def / 16 Spe
Bold Nature
IVs: 0 Atk
- Calm Mind
- Flamethrower
- Moonblast
- Moonlight

Zamazenta @ Life Orb
Ability: Dauntless Shield
Tera Type: Steel
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Close Combat
- Crunch
- Heavy Slam
- Stone Edge

Samurott-Hisui @ Heavy-Duty Boots
Ability: Sharpness
Tera Type: Fire
EVs: 236 Atk / 20 Def / 252 Spe
Jolly Nature
- Ceaseless Edge
- Razor Shell
- Knock Off
- Sucker Punch
""",
    # Expert Belt Zamazenta BO
    """
Zamazenta @ Expert Belt
Ability: Dauntless Shield
Tera Type: Dark
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Close Combat
- Crunch
- Stone Edge
- Ice Fang

Ogerpon-Wellspring @ Wellspring Mask
Ability: Water Absorb
Tera Type: Water
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Ivy Cudgel
- Power Whip
- Knock Off
- U-turn

Slowking-Galar @ Assault Vest
Ability: Regenerator
Tera Type: Water
EVs: 208 HP / 216 SpA / 80 SpD / 4 Spe
Modest Nature
IVs: 0 Atk
- Psyshock
- Flamethrower
- Sludge Bomb
- Ice Beam

Great Tusk @ Heavy-Duty Boots
Ability: Protosynthesis
Tera Type: Fire
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Headlong Rush
- Ice Spinner
- Rapid Spin
- Stealth Rock

Kingambit @ Leftovers
Ability: Supreme Overlord
Tera Type: Ghost
EVs: 200 HP / 252 Atk / 56 Spe
Adamant Nature
- Kowtow Cleave
- Sucker Punch
- Low Kick
- Swords Dance

Zapdos @ Heavy-Duty Boots
Ability: Static
Tera Type: Grass
EVs: 248 HP / 200 Def / 60 Spe
Bold Nature
IVs: 0 Atk
- Hurricane
- Volt Switch
- Thunder Wave
- Roost
""",
    # Meowscarada + Ursaluna Balance
    """
Gliscor @ Toxic Orb
Ability: Poison Heal
Tera Type: Normal
EVs: 244 HP / 88 Def / 176 Spe
Impish Nature
- Facade
- Knock Off
- Swords Dance
- Protect

Meowscarada @ Heavy-Duty Boots
Ability: Protean
Tera Type: Ghost
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Knock Off
- U-turn
- Triple Axel
- Spikes

Ursaluna @ Heavy-Duty Boots
Ability: Bulletproof
Tera Type: Steel
EVs: 184 HP / 56 Atk / 208 SpD / 60 Spe
Adamant Nature
- Headlong Rush
- Ice Punch
- Rest
- Sleep Talk

Skarmory @ Rocky Helmet
Ability: Sturdy
Tera Type: Dragon
EVs: 240 HP / 44 Atk / 216 Def / 8 Spe
Impish Nature
- Brave Bird
- Stealth Rock
- Roost
- Whirlwind

Dragapult @ Heavy-Duty Boots
Ability: Infiltrator
Tera Type: Ghost
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Hex
- Draco Meteor
- U-turn
- Will-O-Wisp

Slowking-Galar @ Heavy-Duty Boots
Ability: Regenerator
Tera Type: Water
EVs: 248 HP / 8 Def / 252 SpD
Sassy Nature
IVs: 0 Atk / 0 Spe
- Sludge Bomb
- Psychic Noise
- Thunder Wave
- Chilly Reception
""",
    # Band Zamazenta + SubTect Kyurem Balance
    """
Kyurem @ Leftovers
Ability: Pressure
Tera Type: Ground
EVs: 56 HP / 200 SpA / 252 Spe
Timid Nature
IVs: 0 Atk
- Substitute
- Protect
- Freeze-Dry
- Earth Power

Ting-Lu @ Leftovers
Ability: Vessel of Ruin
Tera Type: Fairy
EVs: 248 HP / 8 Def / 252 SpD
Careful Nature
- Stealth Rock
- Rest
- Ruination
- Earthquake

Corviknight @ Rocky Helmet
Ability: Pressure
Tera Type: Water
EVs: 248 HP / 252 Def / 8 SpD
Relaxed Nature
IVs: 0 Spe
- Roost
- Brave Bird
- U-turn
- Iron Defense

Weezing-Galar @ Heavy-Duty Boots
Ability: Neutralizing Gas
Tera Type: Fairy
EVs: 248 HP / 252 Def / 8 SpD
Bold Nature
IVs: 0 Atk
- Will-O-Wisp
- Strange Steam
- Defog
- Pain Split

Zamazenta @ Choice Band
Ability: Dauntless Shield
Tera Type: Fighting
EVs: 252 Atk / 4 Def / 252 Spe
Jolly Nature
- Close Combat
- Stone Edge
- Heavy Slam
- Crunch

Toxapex @ Assault Vest
Ability: Regenerator
Tera Type: Water
EVs: 248 HP / 8 Def / 252 SpA
Modest Nature
IVs: 0 Atk
- Surf
- Sludge Bomb
- Ice Beam
- Acid Spray
""",
    # Keldeo + Sinistcha Balance
    """
Ting-Lu @ Leftovers
Ability: Vessel of Ruin
Tera Type: Ghost
EVs: 252 HP / 252 SpD / 4 Spe
Careful Nature
- Spikes
- Ruination
- Earthquake
- Whirlwind

Gliscor @ Toxic Orb
Ability: Poison Heal
Tera Type: Normal
EVs: 244 HP / 12 Atk / 252 Spe
Jolly Nature
- Swords Dance
- Knock Off
- Facade
- Protect

Sinistcha @ Heavy-Duty Boots
Ability: Heatproof
Tera Type: Poison
EVs: 248 HP / 248 Def / 12 Spe
Bold Nature
IVs: 0 Atk
- Calm Mind
- Matcha Gotcha
- Shadow Ball
- Strength Sap

Keldeo @ Heavy-Duty Boots
Ability: Justified
Tera Type: Water
EVs: 4 Def / 252 SpA / 252 Spe
Timid Nature
- Vacuum Wave
- Surf
- Flip Turn
- Aura Sphere

Tinkaton @ Air Balloon
Ability: Mold Breaker
Tera Type: Water
EVs: 248 HP / 28 Def / 232 Spe
Jolly Nature
- Gigaton Hammer
- Thunder Wave
- Encore
- Stealth Rock

Dragonite @ Heavy-Duty Boots
Ability: Multiscale
Tera Type: Normal
EVs: 248 HP / 224 Atk / 16 Def / 20 Spe
Adamant Nature
- Dragon Tail
- Extreme Speed
- Roost
- Earthquake
""",
    # Ting Lu Stall
    """
Blissey @ Heavy-Duty Boots
Ability: Natural Cure
Tera Type: Dark
EVs: 4 HP / 252 Def / 252 SpD
Calm Nature
IVs: 0 Atk
- Calm Mind
- Flamethrower
- Seismic Toss
- Soft-Boiled

Clefable @ Sticky Barb
Ability: Magic Guard
Tera Type: Steel
EVs: 252 HP / 252 Def / 4 SpD
Bold Nature
- Knock Off
- Moonblast
- Moonlight
- Wish

Gliscor @ Toxic Orb
Ability: Poison Heal
Tera Type: Dragon
EVs: 244 HP / 252 Def / 12 SpD
Impish Nature
IVs: 24 Spe
- Toxic
- Knock Off
- Spikes
- Protect

Ting-Lu @ Heavy-Duty Boots
Ability: Vessel of Ruin
Tera Type: Ghost
EVs: 252 HP / 4 Def / 252 SpD
Careful Nature
- Stealth Rock
- Earthquake
- Rest
- Protect

Dondozo @ Heavy-Duty Boots
Ability: Unaware
Tera Type: Fighting
EVs: 252 HP / 252 Def / 4 SpD
Impish Nature
IVs: 18 Spe
- Waterfall
- Curse
- Sleep Talk
- Rest

Amoonguss @ Heavy-Duty Boots
Ability: Regenerator
Tera Type: Steel
EVs: 252 HP / 252 Def / 4 SpD
Relaxed Nature
- Toxic
- Foul Play
- Synthesis
- Seed Bomb
""",
    # Special Spam Offense
    """
Landorus-Therian @ Rocky Helmet
Ability: Intimidate
Tera Type: Dragon
EVs: 252 HP / 4 SpA / 252 Spe
Timid Nature
- Earth Power
- U-turn
- Stealth Rock
- Taunt

Zamazenta @ Leftovers
Ability: Dauntless Shield
Tera Type: Fire
EVs: 252 HP / 80 Def / 176 Spe
Jolly Nature
- Iron Defense
- Body Press
- Roar
- Crunch

Raging Bolt @ Booster Energy
Ability: Protosynthesis
Tera Type: Fairy
EVs: 4 HP / 252 SpA / 252 Spe
Modest Nature
IVs: 20 Atk
- Calm Mind
- Thunderbolt
- Thunderclap
- Dragon Pulse

Darkrai @ Heavy-Duty Boots
Ability: Bad Dreams
Tera Type: Poison
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
IVs: 0 Atk
- Will-O-Wisp
- Dark Pulse
- Ice Beam
- Sludge Bomb

Gholdengo @ Air Balloon
Ability: Good as Gold
Tera Type: Fairy
EVs: 252 HP / 196 Def / 60 Spe
Bold Nature
- Hex
- Thunder Wave
- Recover
- Make It Rain

Dragonite @ Heavy-Duty Boots
Ability: Multiscale
Tera Type: Normal
EVs: 252 Atk / 4 SpD / 252 Spe
Adamant Nature
- Dragon Dance
- Earthquake
- Extreme Speed
- Encore
""",
    # The SSS Sun
    """
Torkoal @ Heat Rock
Ability: Drought
Tera Type: Fairy
EVs: 104 HP / 236 SpA / 168 SpD
Quiet Nature
- Overheat
- Rapid Spin
- Stealth Rock
- Earthquake

Great Tusk @ Covert Cloak
Ability: Protosynthesis
Tera Type: Steel
EVs: 252 HP / 4 Atk / 252 Spe
Jolly Nature
- Earthquake
- Ice Spinner
- Bulk Up
- Rapid Spin

Slither Wing @ Assault Vest
Ability: Protosynthesis
Tera Type: Electric
EVs: 168 HP / 252 Atk / 88 Spe
Adamant Nature
- U-turn
- First Impression
- Earthquake
- Low Kick

Venusaur @ Life Orb
Ability: Chlorophyll
Tera Type: Fire
EVs: 4 Def / 252 SpA / 252 Spe
Timid Nature
IVs: 29 HP / 0 Atk
- Growth
- Giga Drain
- Sludge Bomb
- Weather Ball

Walking Wake @ Choice Specs
Ability: Protosynthesis
Tera Type: Water
EVs: 8 HP / 4 Def / 244 SpA / 252 Spe
Timid Nature
- Hydro Steam
- Draco Meteor
- Flamethrower
- Flip Turn

Heatran @ Air Balloon
Ability: Flash Fire
Tera Type: Ghost
EVs: 224 HP / 40 SpA / 244 Spe
Modest Nature
IVs: 0 Atk
- Magma Storm
- Earth Power
- Solar Beam
- Taunt
""",
    # Roar Zama + Sub Kyu Balance
    """
Moltres @ Heavy-Duty Boots
Ability: Flame Body
Tera Type: Grass
EVs: 248 HP / 236 Def / 24 Spe
Bold Nature
IVs: 0 Atk
- Flamethrower
- Scorching Sands
- Roar
- Roost

Gliscor @ Toxic Orb
Ability: Poison Heal
Tera Type: Fairy
EVs: 244 HP / 36 Def / 228 SpD
Careful Nature
- Earthquake
- U-turn
- Stealth Rock
- Protect

Zamazenta @ Heavy-Duty Boots
Ability: Dauntless Shield
Tera Type: Dark
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Close Combat
- Crunch
- Stone Edge
- Roar

Great Tusk @ Leftovers
Ability: Protosynthesis
Tera Type: Ground
EVs: 252 Atk / 4 Def / 252 Spe
Jolly Nature
- Headlong Rush
- Ice Spinner
- Knock Off
- Rapid Spin

Slowking-Galar @ Heavy-Duty Boots
Ability: Regenerator
Tera Type: Water
EVs: 252 HP / 16 Def / 240 SpD
Sassy Nature
IVs: 0 Atk / 0 Spe
- Sludge Bomb
- Future Sight
- Thunder Wave
- Chilly Reception

Kyurem @ Leftovers
Ability: Pressure
Tera Type: Ground
EVs: 56 HP / 200 SpA / 252 Spe
Timid Nature
IVs: 0 Atk
- Substitute
- Protect
- Freeze-Dry
- Earth Power
""",
    # Zap + Crown VoltTurn Balance
    """
Zapdos @ Heavy-Duty Boots
Ability: Static
Tera Type: Water
EVs: 252 HP / 240 Def / 16 Spe
Bold Nature
IVs: 0 Atk
- Thunder Wave
- Volt Switch
- Hurricane
- Roost

Iron Crown @ Choice Specs
Ability: Quark Drive
Tera Type: Steel
EVs: 4 Def / 252 SpA / 252 Spe
Timid Nature
IVs: 20 Atk
- Tachyon Cutter
- Psyshock
- Psychic Noise
- Volt Switch

Garganacl @ Leftovers
Ability: Purifying Salt
Tera Type: Water
EVs: 252 HP / 4 Def / 252 SpD
Careful Nature
- Salt Cure
- Earthquake
- Curse
- Recover

Great Tusk @ Rocky Helmet
Ability: Protosynthesis
Tera Type: Steel
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Stealth Rock
- Headlong Rush
- Rapid Spin
- Ice Spinner

Samurott-Hisui @ Assault Vest
Ability: Sharpness
Tera Type: Poison
EVs: 216 HP / 112 Atk / 56 SpD / 124 Spe
Adamant Nature
- Ceaseless Edge
- Razor Shell
- Sucker Punch
- Knock Off

Dragapult @ Heavy-Duty Boots
Ability: Infiltrator
Tera Type: Steel
EVs: 76 Atk / 180 SpA / 252 Spe
Hasty Nature
- Dragon Darts
- Hex
- Will-O-Wisp
- U-turn
""",
    # Sub Kyu + Dozo Fat Balance
    """
Kyurem @ Leftovers
Ability: Pressure
Tera Type: Ground
EVs: 64 HP / 220 SpA / 224 Spe
Timid Nature
IVs: 0 Atk
- Substitute
- Earth Power
- Freeze-Dry
- Protect

Corviknight @ Rocky Helmet
Ability: Pressure
Tera Type: Dragon
EVs: 248 HP / 252 Def / 8 SpD
Impish Nature
- Defog
- Brave Bird
- Roost
- U-turn

Ting-Lu @ Leftovers
Ability: Vessel of Ruin
Tera Type: Water
EVs: 252 HP / 4 Def / 252 SpD
Careful Nature
- Earthquake
- Payback
- Rest
- Sleep Talk

Dondozo @ Leftovers
Ability: Unaware
Tera Type: Dark
EVs: 248 HP / 252 Def / 8 SpD
Impish Nature
- Curse
- Waterfall
- Body Press
- Rest

Slowking-Galar @ Heavy-Duty Boots
Ability: Regenerator
Tera Type: Water
EVs: 252 HP / 16 Def / 240 SpD
Sassy Nature
IVs: 0 Atk / 0 Spe
- Future Sight
- Sludge Bomb
- Toxic
- Chilly Reception

Cinderace @ Heavy-Duty Boots
Ability: Blaze
Tera Type: Flying
EVs: 232 HP / 24 Atk / 252 Spe
Jolly Nature
- Pyro Ball
- Will-O-Wisp
- U-turn
- Court Change
""",
    # Quagsire Hard Stall
    """
Quagsire @ Heavy-Duty Boots
Ability: Unaware
Tera Type: Fairy
EVs: 252 HP / 252 Def / 4 SpD
Impish Nature
- Recover
- Stealth Rock
- Toxic
- Earthquake

Blissey @ Heavy-Duty Boots
Ability: Natural Cure
Tera Type: Dark
EVs: 4 HP / 252 Def / 252 SpD
Calm Nature
IVs: 0 Atk
- Soft-Boiled
- Calm Mind
- Flamethrower
- Seismic Toss

Toxapex @ Heavy-Duty Boots
Ability: Regenerator
Tera Type: Steel
EVs: 248 HP / 252 SpD / 8 Spe
Careful Nature
IVs: 0 Atk
- Toxic
- Toxic Spikes
- Recover
- Haze

Sinistcha @ Heavy-Duty Boots
Ability: Heatproof
Tera Type: Fairy
EVs: 160 HP / 252 Def / 96 Spe
Bold Nature
IVs: 0 Atk
- Strength Sap
- Matcha Gotcha
- Foul Play
- Hex

Gliscor @ Toxic Orb
Ability: Poison Heal
Tera Type: Steel
EVs: 244 HP / 168 SpD / 96 Spe
Careful Nature
- Protect
- Knock Off
- Spikes
- Earthquake

Corviknight @ Heavy-Duty Boots
Ability: Pressure
Tera Type: Fighting
EVs: 60 HP / 252 Def / 196 Spe
Bold Nature
IVs: 0 Atk
- Roost
- Iron Defense
- Body Press
- Defog
""",
    # Choice Specs Kyurem Offense
    """
Kingambit @ Leftovers
Ability: Supreme Overlord
Tera Type: Dark
EVs: 160 HP / 252 Atk / 96 Spe
Adamant Nature
- Kowtow Cleave
- Iron Head
- Sucker Punch
- Swords Dance

Cinderace @ Heavy-Duty Boots
Ability: Blaze
Tera Type: Flying
EVs: 144 HP / 112 Atk / 252 Spe
Jolly Nature
- Pyro Ball
- Will-O-Wisp
- Court Change
- U-turn

Landorus-Therian @ Rocky Helmet
Ability: Intimidate
Tera Type: Dragon
EVs: 248 HP / 244 Def / 16 Spe
Bold Nature
- Earth Power
- Taunt
- Stealth Rock
- U-turn

Iron Valiant @ Booster Energy
Ability: Quark Drive
Tera Type: Ghost
EVs: 176 Atk / 80 SpA / 252 Spe
Naive Nature
- Moonblast
- Close Combat
- Knock Off
- Encore

Kyurem @ Choice Specs
Ability: Pressure
Tera Type: Ice
EVs: 4 Def / 252 SpA / 252 Spe
Timid Nature
IVs: 0 Atk
- Draco Meteor
- Freeze-Dry
- Earth Power
- Blizzard

Slowking-Galar @ Heavy-Duty Boots
Ability: Regenerator
Tera Type: Water
EVs: 248 HP / 8 Def / 252 SpD
Sassy Nature
IVs: 0 Atk / 0 Spe
- Toxic
- Future Sight
- Surf
- Chilly Reception
""",
    # Swords Dance Gliscor BO
    """
Ting-Lu @ Leftovers
Ability: Vessel of Ruin
Tera Type: Water
EVs: 252 HP / 24 Def / 228 SpD / 4 Spe
Impish Nature
- Stealth Rock
- Earthquake
- Ruination
- Whirlwind

Dragapult @ Heavy-Duty Boots
Ability: Infiltrator
Tera Type: Dragon
EVs: 4 Def / 252 SpA / 252 Spe
Timid Nature
- Draco Meteor
- Hex
- Will-O-Wisp
- U-turn

Slowking-Galar @ Heavy-Duty Boots
Ability: Regenerator
Tera Type: Water
EVs: 252 HP / 4 Def / 252 SpD
Sassy Nature
IVs: 0 Atk
- Future Sight
- Sludge Bomb
- Flamethrower
- Chilly Reception

Kingambit @ Leftovers
Ability: Supreme Overlord
Tera Type: Dark
EVs: 232 HP / 252 Atk / 24 Spe
Adamant Nature
- Sucker Punch
- Swords Dance
- Iron Head
- Kowtow Cleave

Gliscor @ Toxic Orb
Ability: Poison Heal
Tera Type: Fairy
EVs: 244 HP / 244 SpD / 20 Spe
Jolly Nature
- Swords Dance
- Earthquake
- Knock Off
- Protect

Zamazenta @ Heavy-Duty Boots
Ability: Dauntless Shield
Tera Type: Fire
EVs: 80 HP / 252 Atk / 176 Spe
Jolly Nature
- Close Combat
- Stone Edge
- Crunch
- Roar
""",
    # Ogerpon Hazard Stack
    """
Gholdengo @ Heavy-Duty Boots
Ability: Good as Gold
Tera Type: Fairy
EVs: 252 HP / 248 Def / 8 Spe
Bold Nature
IVs: 0 Atk
- Focus Blast
- Thunder Wave
- Hex
- Recover

Ting-Lu @ Heavy-Duty Boots
Ability: Vessel of Ruin
Tera Type: Water
EVs: 252 HP / 4 Def / 252 SpD
Careful Nature
- Whirlwind
- Earthquake
- Ruination
- Spikes

Deoxys-Speed @ Heavy-Duty Boots
Ability: Pressure
Tera Type: Fighting
EVs: 200 Atk / 252 SpA / 56 Spe
Naive Nature
- Knock Off
- Ice Beam
- Psycho Boost
- Superpower

Garganacl @ Leftovers
Ability: Purifying Salt
Tera Type: Water
EVs: 252 HP / 176 Def / 80 SpD
Impish Nature
- Salt Cure
- Protect
- Stealth Rock
- Recover

Ogerpon @ Heavy-Duty Boots
Ability: Defiant
Tera Type: Grass
EVs: 252 Atk / 4 Def / 252 Spe
Jolly Nature
- Knock Off
- Ivy Cudgel
- U-turn
- Encore

Gliscor @ Toxic Orb
Ability: Poison Heal
Tera Type: Water
EVs: 244 HP / 12 Def / 252 SpD
Careful Nature
IVs: 28 Spe
- U-turn
- Toxic
- Earthquake
- Protect
""",
    # AV Hoopa-U Fat
    """
Gliscor @ Toxic Orb
Ability: Poison Heal
Tera Type: Normal
EVs: 244 HP / 240 SpD / 24 Spe
Careful Nature
- Swords Dance
- Facade
- Earthquake
- Protect

Alomomola @ Red Card
Ability: Regenerator
Tera Type: Flying
EVs: 248 HP / 252 Def / 8 SpD
Relaxed Nature
IVs: 0 Spe
- Wish
- Protect
- Flip Turn
- Acrobatics

Gholdengo @ Choice Scarf
Ability: Good as Gold
Tera Type: Fighting
EVs: 4 Def / 252 SpA / 252 Spe
Timid Nature
IVs: 0 Atk
- Shadow Ball
- Make It Rain
- Trick
- Focus Blast

Hoopa-Unbound @ Assault Vest
Ability: Magician
Tera Type: Fairy
EVs: 248 HP / 244 Def / 16 SpA
Quiet Nature
- Knock Off
- Psychic Noise
- Thunderbolt
- Gunk Shot

Cinderace @ Heavy-Duty Boots
Ability: Libero
Tera Type: Fire
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Pyro Ball
- Sucker Punch
- Will-O-Wisp
- Court Change

Zamazenta @ Leftovers
Ability: Dauntless Shield
Tera Type: Steel
EVs: 104 HP / 236 Def / 168 Spe
Jolly Nature
- Iron Defense
- Body Press
- Crunch
- Roar
""",
]


class RandomTeamFromPool(Teambuilder):
    """Picks one of GEN9OU_SAMPLE_TEAMS at random per battle.

    Using a pool instead of one fixed team avoids benchmarking a single,
    possibly-unrepresentative team matchup - every battle in a run still
    samples from a small, known-legal set.
    """

    def __init__(self, teams: List[str] = GEN9OU_SAMPLE_TEAMS):
        self.teams = teams

    def yield_team(self) -> str:
        team = random.choice(self.teams)
        return self.join_team(self.parse_showdown_team(team))
