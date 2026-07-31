"""A small pool of real, legal gen9ou teams for benchmarking.

gen9ou (unlike gen9randombattle) requires each player to submit a
constructed team - poke-env has no built-in team generator for it. These 5
teams are copied verbatim from Smogon's "SV OU Sample Teams" forum thread
(https://www.smogon.com/forums/threads/sv-ou-sample-teams-new-samples-added-post-spl-and-tera-blast-ban.3712513/,
fetched 2026-07-31), a community-vetted resource specifically meant to be
known-legal and competitively reasonable - not hand-written.

Even so, don't trust the forum post's legality over the local server's own
opinion: two of the originally-fetched teams failed `node pokemon-showdown
validate-team gen9ou` on this checkout (one used the since-banned Tera Blast
twice, another used Baxcalibur - tagged Uber here - and Spore, banned by this
ruleset's Sleep Moves Clause). Both were swapped for validator-clean
replacements from the same thread. Re-run the validator
(`node pokemon-showdown validate-team gen9ou < packed_team`, packed via
Teambuilder.join_team(Teambuilder.parse_showdown_team(...))) after editing
this pool or updating the pokemon-showdown checkout - the banlist isn't
static.

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
