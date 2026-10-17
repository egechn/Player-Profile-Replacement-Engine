# Scheduled update

`weekly_update.sh` fetches the season set in `config/statslook_fetch.yaml`
(all competitions and the league-only variant) and rebuilds every season, so
the current clubs stay in sync. It takes about 20 minutes and writes a log to
`data/logs/`. The dashboard picks the new data up after a restart.

Run it by hand:

```bash
PYTHON=/path/to/python scripts/weekly_update.sh
```

To run it every Monday at 07:00 on macOS, add a crontab line (`crontab -e`):

```text
0 7 * * 1 PYTHON=/Users/<you>/anaconda3/bin/python3 "/path/to/Player-Profile-Replacement-Engine/scripts/weekly_update.sh"
```

The computer has to be on (and not asleep) at that time.
