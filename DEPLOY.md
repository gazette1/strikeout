# Deploy — daily Discord picks in the cloud (GitHub Actions)

Runs every morning whether your PC is on or not. Self-skips on days with no MLB games.

## One-time setup

1. **Create an empty GitHub repo** (no README/license) at https://github.com/new — e.g. `strikeout-bot`.

2. **Push this repo** (run from `mlb-k-predictor/`):
   ```bash
   git remote add origin https://github.com/<your-username>/strikeout-bot.git
   git push -u origin main
   ```
   (First push opens a browser to authenticate.)

3. **Add the Discord webhook as a secret:**
   Repo → **Settings → Secrets and variables → Actions → New repository secret**
   - Name: `DISCORD_WEBHOOK_URL`
   - Value: your `https://discord.com/api/webhooks/...` URL

4. **Test it:** Repo → **Actions → "Daily K Picks to Discord" → Run workflow**.
   The first run backfills 2024–2026 Statcast (~5–8 min); later runs use the cache (~2 min).

## Schedule
- Defined in `.github/workflows/daily_k_discord.yml`: `cron: '0 13 * * *'` = 13:00 UTC ≈ 9 AM ET.
- To change the time, edit that cron (it's in UTC) and push.

## After cloud is confirmed working
Disable the local Windows task so you don't get double posts:
```powershell
Disable-ScheduledTask -TaskName "StrikeOutBot Daily K Picks"
```

## Updating the model
Retrain locally, then commit the refreshed `data/models/production/*.txt`:
```bash
python tools/backfill_statcast.py 2024 2025 2026
python tools/build_dataset.py
python tools/train_model.py
git commit -am "retrain model" && git push
```
