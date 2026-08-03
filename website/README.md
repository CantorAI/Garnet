# Garnet Models website

Static marketing site for `garnetmodel.ai`.

## Local preview

```powershell
python -m http.server 4174 --directory D:\CantorAI\Garnet\website
```

Open `http://127.0.0.1:4174`.

## Production

The site is designed to be served directly by Nginx from
`/var/www/garnetmodel.ai`. The production server block is in
`deploy/garnetmodel.nginx`.

Authentication remains on Manifold through the Garnet-branded same-origin alias.
All sign-in and package-access links point to
`https://app.garnetmodel.ai/login?next=/portal`.

The model names, supported backends, benchmark numbers, and maturity labels are
derived from the Garnet repository documentation and model manifests. Update
them as release status changes.
