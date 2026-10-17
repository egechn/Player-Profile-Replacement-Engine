One YAML file per season, named after the season (for example `2026-27.yaml`).
Each line maps a StatsLook team name to a league name from `config/league_strength.yaml`
and overrides every other mapping for that season only. Use it for promoted and
relegated clubs:

```yaml
"Leeds United": Premier League
"Southampton": EFL Championship
```
